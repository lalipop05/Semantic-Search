package storage

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"mySearchEngine/internal/config"
	"mySearchEngine/internal/utils"
	"os"
	"strconv"

	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	_ "github.com/mattn/go-sqlite3"
)

type StorageManager struct {
	db    *sql.DB
	Cache *CrawledPagesCache
}

func SetUpDataBase(driver string) (*StorageManager, error) {
	// Enables the sqlite_vec extension for all functions
	sqlite_vec.Auto()

	// Make a directory to store the database
	err := os.MkdirAll(config.DATABASE_DIR_PATH, 0755)
	if err != nil {
		return nil, fmt.Errorf("falied to create directory at %s\n%w", config.DATABASE_DIR_PATH, err)
	}

	// Connect to or create a database
	db, err := sql.Open(driver, config.DATABASE_DIR_PATH+"/"+config.DATABASE_NAME)
	if err != nil {
		return nil, fmt.Errorf("falied to open database at %s\n%w", config.DATABASE_DIR_PATH+config.DATABASE_NAME, err)
	}

	// Get vecVersion to ensure proper integration between sqlite3 and the vec library
	var vecVersion string
	err = db.QueryRow("select vec_version()").Scan(&vecVersion)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("queryRow to database to get vecVersion failed\n%w", err)
	}
	utils.InfoLogger.Printf("vec_version=%s\n", vecVersion)

	// Enable foreign keys to help link vectors with their associated web pages
	_, err = db.Exec("PRAGMA foreign_keys = ON;")
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to enable foreign keys: %w", err)
	}

	err = createTablesInDatabase(db)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create tables for storage in db\n%w", err)
	}

	// stores webpages visited in this run of the crawler to prevent duplicate entires in the database
	cache := make(CrawledPagesCache)

	storageManager := StorageManager{db, &cache}

	return &storageManager, nil
}

func (storageManager StorageManager) InsertEntriesIntoDB(dbEntries []*PagesDBEntry) error {

	// sql statements to capture the insertion behaviour of various resources
	insertMetaDataSQL := `INSERT OR REPLACE INTO pages_meta_data
	(url, title, description, keywords, heading, hash, modified_time, language, content_length)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id`

	insertVectorMetaDataSQL := `INSERT INTO page_vectors (page_id, vector_type) VALUES (?, ?)`

	insertVectorIdxSQL := `INSERT INTO page_vectors_idx (rowid, embedding) VALUES (?, ?)`

	// insertion transation built specifically for our insertion case
	tx, err := newInsertionTransaction(storageManager.db, insertMetaDataSQL, insertVectorMetaDataSQL, insertVectorIdxSQL)
	if err != nil {
		return fmt.Errorf("could not initialize a new insertion transaction\n%w", err)
	}
	defer tx.Rollback()

	// for each database entry we get, we insert its meta data and the embeddings
	for _, dbEntry := range dbEntries {
		metaData := dbEntry.MetaData

		if storageManager.Cache.Exists(metaData) {
			utils.CrawlLogger.Println("Not adding: ", metaData.URL)
			continue
		}

		pageId, err := tx.insertPageMetaData(dbEntry)
		if err != nil {
			return err
		}

		err = tx.insertPageEmbeddings(dbEntry, pageId)
		if err != nil {
			return fmt.Errorf("failed to insert page embeddings\n%w", err)
		}
	}
	// commit confirms the changes
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit insertion transaction, rolling back...\n%w", err)
	}
	return nil
}

func (storageManager StorageManager) QueryDatabase(query []float32) ([]*DistanceMetric, error) {
	serializedQuery, err := sqlite_vec.SerializeFloat32(query)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize query: %w", err)
	}

	// sql statement gets the rowid and distance of the closest 30 vector embeddings along with 
	// the url of the web page associated with each embeddings
	sqlQuery := `
        SELECT
            idx.rowid,
            idx.distance,
            meta.url
        FROM
            page_vectors_idx AS idx
        JOIN
            page_vectors AS vec ON idx.rowid = vec.id
        JOIN
            pages_meta_data AS meta ON vec.page_id = meta.id
        WHERE
            idx.embedding MATCH ? AND k = 30
        ORDER BY
            idx.distance
    `

	// rows stores the 30 entries we requested
	rows, err := storageManager.db.Query(sqlQuery, serializedQuery)
	if err != nil {
		utils.ErrorLogger.Printf("Failed to execute vector search query: %v", err)
		return nil, err
	}
	defer rows.Close()

	var results []*DistanceMetric

	// store each of the rows to return to our search
	for rows.Next() {
		metric := &DistanceMetric{}
		err := rows.Scan(&metric.Rowid, &metric.Distance, &metric.URL)
		if err != nil {
			utils.ErrorLogger.Printf("Failed to scan search result row: %v", err)
			return nil, err
		}
		results = append(results, metric)
	}

	// Check for any error that occurred during iteration
	if err = rows.Err(); err != nil {
		return nil, fmt.Errorf("error during result set iteration: %v", err)
	}

	return results, nil
}

func (storageManager *StorageManager) Close() {
	// close connection to the db
	storageManager.db.Close()
}

func createTablesInDatabase(db *sql.DB) error {
	// Table to store the metadata of webpages
	pagesMetaDataSQL := `CREATE TABLE IF NOT EXISTS pages_meta_data (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	url TEXT UNIQUE NOT NULL,
	title TEXT,
	content TEXT,
	description TEXT,
	keywords TEXT,
	heading TEXT,
	hash TEXT,
	crawl_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	modified_time TIMESTAMP,
	language TEXT,
	content_length INTEGER);`
	_, err := db.Exec(pagesMetaDataSQL)

	if err != nil {
		return fmt.Errorf("falied to create pages_meta_data table\n%w", err)
	}

	// page_vectors table to store an entry for each embedding produced
	vectorTableSQL := `CREATE TABLE IF NOT EXISTS page_vectors (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		page_id INTEGER NOT NULL,
		vector_type TEXT NOT NULL,
		FOREIGN KEY(page_id) REFERENCES pages_meta_data(id) ON DELETE CASCADE
	);`
	if _, err := db.Exec(vectorTableSQL); err != nil {
		return fmt.Errorf("falied to create page_vectors table\n%w", err)
	}

	// page_vectors_idx creates an entry having the same rowId as the id of the associated entry in
	// page_vectors. The table where the vectors are stored and knn search is implemented
	vectorIndexSQL := `
    CREATE VIRTUAL TABLE IF NOT EXISTS page_vectors_idx USING 
	vec0(embedding float[` + strconv.Itoa(config.MODEL_OUTPUT_VECTOR_LEN) + `]);`

	if _, err := db.Exec(vectorIndexSQL); err != nil {
		return fmt.Errorf("falied to create page_vectors_idx table\n%w", err)
	}

	return nil
}

type InsertionTransaction struct {
	transaction           *sql.Tx
	pageMetaDataStmt      *sql.Stmt
	embeddingMetaDataStmt *sql.Stmt
	embeddingStmt         *sql.Stmt
}

func newInsertionTransaction(db *sql.DB, insertMetaDataSQL, insertVectorMetaDataSQL, insertVectorIdxSQL string) (*InsertionTransaction, error) {
	// transaction: all or nothing behaviour. If one of the insertions fail, none of the insetions goes through
	transaction, err := db.Begin()
	if err != nil {
		return nil, fmt.Errorf("could not begin db transaction\n%w", err)
	}

	// statment responsible for insertion of web page meta data
	pageMetaDataStmt, err := transaction.Prepare(insertMetaDataSQL)
	if err != nil {
		return nil, fmt.Errorf("could not prepare insertion statement for transaction\n%w", err)
	}

	// statement responsbile for insertion of embedding meta data
	embeddingMetaDataStmt, err := transaction.Prepare(insertVectorMetaDataSQL)
	if err != nil {
		return nil, fmt.Errorf("could not prepare insertion statement for transaction\n%w", err)
	}

	// statement responsbile for insertion of embedding in the virtual table
	embeddingStmt, err := transaction.Prepare(insertVectorIdxSQL)
	if err != nil {
		return nil, fmt.Errorf("could not prepare insertion statement for transaction\n%w", err)
	}

	return &InsertionTransaction{
		transaction,
		pageMetaDataStmt,
		embeddingMetaDataStmt,
		embeddingStmt,
	}, nil
}

func (t *InsertionTransaction) insertPageMetaData(dbEntry *PagesDBEntry) (int64, error) {
	// Insert the web page meta data into a table
	metaData := dbEntry.MetaData
	headingsJSON, err := json.Marshal(metaData.Headings)
	if err != nil {
		return 0, fmt.Errorf("failed to marshal metaData headings\n%w", err)
	}

	// Get the id of the inserted web page metadata
	var pageId int64
	err = t.pageMetaDataStmt.QueryRow(
		metaData.URL,
		metaData.Title,
		metaData.MetaDescription,
		metaData.MetaKeyWords,
		string(headingsJSON),
		metaData.Hash,
		metaData.ModifiedTime,
		metaData.Language,
		metaData.ContentLength,
	).Scan(&pageId)
	if err != nil {
		return 0, fmt.Errorf("failed to insert into pageMetaDataStmt\n%w", err)
	}

	return pageId, nil
}

func (t *InsertionTransaction) insertPageEmbeddings(dbEntry *PagesDBEntry, pageId int64) error {
	// Insert each embedding individual though a transaction
	// First insert the embedding meta data into a table and get the rowId
	// then insert the embedding in the virtual table with the id matching the rowId
	// to map both entries in to the same embedding
	for i, embedding := range dbEntry.Embeddings {
		// Which content chunk is the current embedding
		vectorType := fmt.Sprintf("content_chunk_%d", i+1)

		// insert metadata and get rowId
		res, err := t.embeddingMetaDataStmt.Exec(pageId, vectorType)
		if err != nil {
			return fmt.Errorf("could not insert embeddings meta data into db\n%w", err)
		}
		embeddingRowId, err := res.LastInsertId()
		if err != nil {
			return fmt.Errorf("could not fetch rowId of meta data in db\n%w", err)
		}

		serializedEmbedding, err := sqlite_vec.SerializeFloat32(embedding)
		if err != nil {
			return fmt.Errorf("could not serialized embeddings\n%w", err)
		}

		// insert into the virtual table
		_, err = t.embeddingStmt.Exec(embeddingRowId, serializedEmbedding)
		if err != nil {
			return fmt.Errorf("could not insert embeddings into virtual table in db\n%w", err)
		}
	}
	return nil
}

func (t *InsertionTransaction) Rollback() {
	// Rollback transaction so that all or nothing behaviour of the transaction is enforced
	t.transaction.Rollback()
}

func (t *InsertionTransaction) Commit() error {
	// Commit means all the changes went through
	return t.transaction.Commit()
}

func (t *InsertionTransaction) Close() {
	// Close all statements
	t.pageMetaDataStmt.Close()
	t.embeddingMetaDataStmt.Close()
	t.embeddingStmt.Close()
}

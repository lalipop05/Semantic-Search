package storage

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"mySearchEngine/utils"
	"os"
	"strconv"
	"strings"

	"github.com/mattn/go-sqlite3"
)

const VEC_DIM = 512

type StorageManager struct {
	db    *sql.DB
	Cache *CrawledPagesCache
}

func SetUpDataBase(driver string) (*StorageManager, error) {

	err := os.MkdirAll("./database", 0755)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	sql.Register("sqlite3_with_extensions",
		&sqlite3.SQLiteDriver{
			Extensions: []string{
				"vec0",
			},
		})

	db, err := sql.Open("sqlite3_with_extensions", "./database/crawler.db")

	//dsn := "file:./database/crawler.db?_extension_functions"

	//db, err := sql.Open(driver, dsn)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	// _, err = db.Exec("SELECT load_extension('./vec0.dll')")
	// if err != nil {
	// 	utils.ErrorLogger.Println(err)
	// 	return nil, fmt.Errorf("failed to load extension: %w", err)
	// }

	_, err = db.Exec("PRAGMA foreign_keys = ON;")
	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, fmt.Errorf("failed to enable foreign keys: %w", err)
	}

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
	_, err = db.Exec(pagesMetaDataSQL)

	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	vectorTableSQL := `CREATE TABLE IF NOT EXISTS page_vectors (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		page_id INTEGER NOT NULL,
		vector_type TEXT NOT NULL,
		FOREIGN KEY(page_id) REFERENCES pages_meta_data(id) ON DELETE CASCADE
	);`
	if _, err := db.Exec(vectorTableSQL); err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	vectorIndexSQL := `
    CREATE VIRTUAL TABLE IF NOT EXISTS page_vectors_idx USING vec0(
        embedding_vec(`+ strconv.Itoa(VEC_DIM) + `));`

	if _, err := db.Exec(vectorIndexSQL); err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	cache := make(CrawledPagesCache)

	storageManager := StorageManager{db, &cache}

	return &storageManager, nil
}

func (storageManager StorageManager) InsertEntriesIntoDB(dbEntries []*PagesDBEntry) (uint64, error) {
	var pagesInserted uint64 = 0

	db := storageManager.db

	insertMetaDataSQL := `INSERT OR REPLACE INTO pages_meta_data
	(url, title, content, description, keywords, heading, hash, modified_time, language, content_length)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id`

	insertVectorMetaDataSQL := `INSERT INTO page_vectors (page_id, vector_type) VALUES (?, ?)`

	insertVectorIdxSQL := `INSERT INTO page_vectors_idx (rowid, embedding_vec) VALUES (?, ?)`

	transaction, err := db.Begin()
	if err != nil {
		utils.ErrorLogger.Println(err)
		return 0, err
	}
	defer transaction.Rollback()

	pageMetaDataStmt, err := transaction.Prepare(insertMetaDataSQL)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return 0, err
	}
	defer pageMetaDataStmt.Close()

	embeddingMetaDataStmt, err := transaction.Prepare(insertVectorMetaDataSQL)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return 0, err
	}
	defer embeddingMetaDataStmt.Close()

	embeddingStmt, err := transaction.Prepare(insertVectorIdxSQL)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return 0, err
	}
	defer embeddingStmt.Close()

	for _, dbEntry := range dbEntries {
		metaData := dbEntry.MetaData

		if storageManager.Cache.Exists(metaData) {
			utils.CrawlLogger.Println("Not adding: ", metaData.URL)
			continue
		}

		headingsJSON, err := json.Marshal(metaData.Headings)
		if err != nil {
			utils.ErrorLogger.Println(err)
			return 0, err
		}

		var pageId int64
		err = pageMetaDataStmt.QueryRow(
			metaData.URL,
			metaData.Title,
			metaData.Content,
			metaData.MetaDescription,
			metaData.MetaKeyWords,
			string(headingsJSON),
			metaData.Hash,
			metaData.ModifiedTime,
			metaData.Language,
			metaData.ContentLength,
		).Scan(&pageId)

		if err != nil {
			utils.ErrorLogger.Println(err)
			return 0, err
		}

		for i, embedding := range dbEntry.Embeddings {
			vectorType := fmt.Sprintf("content_chunk_%d", i)

			res, err := embeddingMetaDataStmt.Exec(pageId, vectorType)
			if err != nil {
				utils.ErrorLogger.Println(err)
				return 0, err
			}
			embeddingRowId, err := res.LastInsertId()
			if err != nil {
				utils.ErrorLogger.Println(err)
				return 0, err
			}

			embeddingJSON, err := json.Marshal(embedding)
			if err != nil {
				utils.ErrorLogger.Println(err)
				return 0, err
			}

			_, err = embeddingStmt.Exec(embeddingRowId, embeddingJSON)
			if err != nil {
				utils.ErrorLogger.Println(err)
				return 0, err
			}
		}
		pagesInserted++
	}
	if err := transaction.Commit(); err != nil {
		utils.ErrorLogger.Println(err)
		return 0, err
	}
	return pagesInserted, nil
}

func (storageManager StorageManager) InsertMetaDataIntoDB(data *PagesMetaData) (uint64, error) {

	if storageManager.Cache.Exists(data) {
		utils.CrawlLogger.Println("Not adding: ", data.URL)
		return 0, nil
	}

	db := storageManager.db

	headingsStr := strings.Join(data.Headings, "|")

	insertSQL := `INSERT OR REPLACE INTO pages_meta_data
	(url, title, content, description, keywords, heading, hash, modified_time, language, content_length)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	result, err := db.Exec(insertSQL,
		data.URL,
		data.Title,
		data.Content,
		data.MetaDescription,
		data.MetaKeyWords,
		headingsStr,
		data.Hash,
		data.ModifiedTime,
		data.Language,
		data.ContentLength,
	)

	if err != nil {
		utils.ErrorLogger.Println(err)
		return 0, err
	}

	lastIndex, err := result.LastInsertId()

	if err != nil {
		utils.ErrorLogger.Println(err)
		return 0, err
	}

	return uint64(lastIndex), err
}

func (storageManager StorageManager) PrintLastFiveRows() {

	err := os.MkdirAll("./debug", 0755)
	if err != nil {
		utils.ErrorLogger.Fatal(err)
	}
	file, err := os.OpenFile("./debug/database.txt", os.O_CREATE|os.O_WRONLY, 0666)
	if err != nil {
		utils.ErrorLogger.Fatal(err)
	}
	defer file.Close()

	db := storageManager.db

	rows, err := db.Query("SELECT url, title, content, description, hash, content_length FROM pages_meta_data ORDER BY id DESC LIMIT 5")
	file.WriteString("Printing rows")
	if err != nil {
		s := fmt.Sprintf("Query error: %v", err)
		file.WriteString(s)
		return
	}
	defer rows.Close()

	file.WriteString("Recent entries:")
	for rows.Next() {
		var url, title, content, description, hash string
		var len int32
		rows.Scan(&url, &title, &content, &description, &hash, &len)
		s := fmt.Sprintf("URL: %s\nTitle: %s\nContent: %.50s...\nDescription: %s\n Lenght: %d Hash: %s\n\n",
			url, title, content, description, len, hash)
		file.WriteString(s)
	}
}

func (storageManager *StorageManager) Close() {
	storageManager.db.Close()
}

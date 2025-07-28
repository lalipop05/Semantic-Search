package storage

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"mySearchEngine/internal/utils"
	"os"
	"strconv"

	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	_ "github.com/mattn/go-sqlite3"
)

const VEC_DIM = 384

const DATABASE_DIR_PATH = "./../database"
const DATABASE_NAME = "tester.db"

type StorageManager struct {
	db    *sql.DB
	Cache *CrawledPagesCache
}

func SetUpDataBase(driver string) (*StorageManager, error) {
	sqlite_vec.Auto()
	err := os.MkdirAll(DATABASE_DIR_PATH, 0755)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	db, err := sql.Open("sqlite3", DATABASE_DIR_PATH+"/"+DATABASE_NAME)
	if err != nil {
		utils.ErrorLogger.Fatal(err)
	}

	var vecVersion string
	err = db.QueryRow("select vec_version()").Scan(&vecVersion)
	if err != nil {
		utils.ErrorLogger.Fatal(err)
	}
	utils.InfoLogger.Printf("vec_version=%s\n", vecVersion)

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
    CREATE VIRTUAL TABLE IF NOT EXISTS page_vectors_idx USING 
	vec0(embedding float[` + strconv.Itoa(VEC_DIM) + `]);`

	if _, err := db.Exec(vectorIndexSQL); err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	cache := make(CrawledPagesCache)

	storageManager := StorageManager{db, &cache}

	return &storageManager, nil
}

func (storageManager StorageManager) InsertEntriesIntoDB(dbEntries []*PagesDBEntry) error {
	db := storageManager.db

	transaction, err := db.Begin()
	if err != nil {
		utils.ErrorLogger.Println(err)
		return err
	}
	defer transaction.Rollback()

	insertMetaDataSQL := `INSERT OR REPLACE INTO pages_meta_data
	(url, title, content, description, keywords, heading, hash, modified_time, language, content_length)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id`

	insertVectorMetaDataSQL := `INSERT INTO page_vectors (page_id, vector_type) VALUES (?, ?)`

	insertVectorIdxSQL := `INSERT INTO page_vectors_idx (rowid, embedding) VALUES (?, ?)`

	pageMetaDataStmt, err := transaction.Prepare(insertMetaDataSQL)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return err
	}
	defer pageMetaDataStmt.Close()

	embeddingMetaDataStmt, err := transaction.Prepare(insertVectorMetaDataSQL)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return err
	}
	defer embeddingMetaDataStmt.Close()

	embeddingStmt, err := transaction.Prepare(insertVectorIdxSQL)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return err
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
			return err
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
			return err
		}

		for i, embedding := range dbEntry.Embeddings {
			vectorType := fmt.Sprintf("content_chunk_%d", i+1)

			res, err := embeddingMetaDataStmt.Exec(pageId, vectorType)
			if err != nil {
				utils.ErrorLogger.Println(err)
				return err
			}
			embeddingRowId, err := res.LastInsertId()
			if err != nil {
				utils.ErrorLogger.Println(err)
				return err
			}

			serializedEmbedding, err := sqlite_vec.SerializeFloat32(*embedding)
			if err != nil {
				utils.ErrorLogger.Println(err)
				return err
			}

			_, err = embeddingStmt.Exec(embeddingRowId, serializedEmbedding)
			if err != nil {
				utils.ErrorLogger.Println(err)
				return err
			}
		}
	}
	if err := transaction.Commit(); err != nil {
		utils.ErrorLogger.Println(err)
		return err
	}
	return nil
}

func (storageManager StorageManager) QueryDatabase(query []float32) ([]*DistanceMetric, error) {
    serializedQuery, err := sqlite_vec.SerializeFloat32(query)
    if err != nil {
        utils.ErrorLogger.Printf("Failed to serialize query vector: %v", err)
        return nil, fmt.Errorf("failed to serialize query: %w", err)
    }

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
            idx.embedding MATCH ? AND k = 5
        ORDER BY
            idx.distance
    `

    rows, err := storageManager.db.Query(sqlQuery, serializedQuery)
    if err != nil {
        utils.ErrorLogger.Printf("Failed to execute vector search query: %v", err)
        return nil, err
    }
    defer rows.Close() 

    var results []*DistanceMetric

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
        utils.ErrorLogger.Printf("Error during result set iteration: %v", err)
        return nil, err
    }

    return results, nil
}

func (storageManager *StorageManager) Close() {
	storageManager.db.Close()
}

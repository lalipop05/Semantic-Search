package storage

import (
	"database/sql"
	"fmt"
	"mySearchEngine/utils"
	"os"
	"strings"
)

type StorageManager struct {
	db    *sql.DB
	Cache *CrawledPagesCache
}

func SetUpDataBase(driver string) (*StorageManager, error) {

	err := os.MkdirAll("./database", 0755)
	if err != nil {
		utils.Error(err)
		return nil, err
	}

	db, err := sql.Open(driver, "./database/crawler.db")

	if err != nil {
		utils.Error(err)
		return nil, err
	}

	cache := make(CrawledPagesCache)

	storageManager := StorageManager{db, &cache}

	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS pages_meta_data (
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
	content_length INTEGER)`)

	if err != nil {
		utils.Error(err)
		return &storageManager, err
	}
	return &storageManager, nil
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

	if (err != nil) {
		utils.ErrorLogger.Println(err)
		return 0, err
	}

	lastIndex, err := result.LastInsertId()

	if (err != nil) {
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

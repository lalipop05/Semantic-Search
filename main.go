package main

import (
	"database/sql"
	"fmt"
	"os"
	"strings"
	"time"

	"crawler/storage"

	"github.com/gocolly/colly/v2"
	_ "github.com/mattn/go-sqlite3"
)

func crawl(url string, ch *chan storage.PagesMetaData) uint32 {
	var count int = 0;
	c := colly.NewCollector(colly.MaxDepth(2), colly.Async(true))

	c.Limit(&colly.LimitRule{
		DomainGlob:  "*",
		Parallelism: 4,  // Number of concurrent requests
	})

	// Find and visit all links
	c.OnHTML("a[href]", func(e *colly.HTMLElement) {
		e.Request.Visit(e.Attr("href"))
		
	})

	c.OnHTML("html", func(e *colly.HTMLElement) {
		metaData :=  storage.PagesMetaData{
			URL: e.Request.URL.String(), 
		}
		*ch <- metaData
	})

	c.OnRequest(func(r *colly.Request) {
		count++;
		s:=""
		for i := 0; i < r.Depth-1; i++ {
			s += "  "
		}
		s += fmt.Sprintf("%d", r.Depth) + "- Visiting %v" + r.URL.String()
		fmt.Println(s)
	})

	c.Visit(url)
	c.Wait();


	return uint32(count)
}

func main() {

	var ch = make(chan storage.PagesMetaData, 16)

	start := time.Now()

	go func () {
		fmt.Println(crawl("https://personal.utdallas.edu/~vince/cs4365-honors/index.html", &ch))
		close(ch)
	}()
	
	db, err := setUpDataBase("sqlite3")

	if (err != nil) {
		fmt.Println(err)
	}
	defer db.Close()

	counter := 0
	for value := range ch {
		insertMetaDataIntoDB(db, &value)
		counter++
		if (counter == 5) {
			counter = 0
			printLastFiveRows(db)

		}
	}
	
	end := time.Now()
	diff := end.Sub(start)
	fmt.Println(diff)
	
}


func insertMetaDataIntoDB(db *sql.DB, data *storage.PagesMetaData) error{
	headingsStr := strings.Join(data.Headings, "|")
	
	insertSQL := `INSERT OR REPLACE INTO pages_meta_data
	(url, title, content, description, keywords, heading, hash, modified_time, language, content_length)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	_, err := db.Exec(insertSQL,
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

	return err;
}

func setUpDataBase(driver string) (*sql.DB, error) {
	err := os.MkdirAll("./database", 0755)
	if err != nil {
		fmt.Println(err)
		return nil, err
	}
	
	db, err := sql.Open(driver, "./database/test.db")
	
	if (err != nil) {
		fmt.Println(err)
		return nil, err
	}

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

	if (err != nil) {
		fmt.Println(err)
		return db, err
	}
	return db, nil
}

func printLastFiveRows(db *sql.DB) {
	rows, err := db.Query("SELECT url, title, content, description FROM pages_meta_data ORDER BY id DESC LIMIT 5")
	fmt.Println("Printing rows")
	if err != nil {
		fmt.Printf("Query error: %v", err)
		return
	}
	defer rows.Close()

	fmt.Println("Recent entries:")
	for rows.Next() {
		var url, title, content, description string
		rows.Scan(&url, &title, &content, &description)
		fmt.Printf("URL: %s\nTitle: %s\nContent: %.50s...\nDescription: %s\n\n", 
			url, title, content, description)
	}

}
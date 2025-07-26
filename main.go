package main

import (
	"fmt"
	"time"

	"mySearchEngine/crawler"
	"mySearchEngine/embeddings"
	"mySearchEngine/storage"
	"mySearchEngine/utils"

	_ "github.com/tursodatabase/libsql-client-go/libsql"
)

const DBDRIVER string = "sqlite3"

func main() {

	var ch = make(chan storage.PagesMetaData, 16)

	start := time.Now()

	go func() {
		crawler.Crawl("https://personal.utdallas.edu/~vince/cs4365-honors/index.html", &ch)
		close(ch)
	}()

	storageManager, err := storage.SetUpDataBase(DBDRIVER)
	if err != nil {
		if storageManager != nil {
			storageManager.Close()
		}
		utils.ErrorLogger.Println(err)
		panic(err)
	}
	defer storageManager.Close()

	embeddingService, err := embeddings.NewEmbeddingService()
	if err != nil {
		utils.ErrorLogger.Println(err)
		panic(err)
	}

	for value := range ch {
		dbEntry, err := embeddingService.GenerateBatchEmbeddings([]*storage.PagesMetaData{&value})
		if err != nil {
			utils.ErrorLogger.Println(err)
			panic(err)
		}
		inserted, err := storageManager.InsertEntriesIntoDB(dbEntry)

		if err != nil {
			utils.ErrorLogger.Println(err)
			panic(err)
		}
		if inserted != 1 {
			utils.CrawlLogger.Println(dbEntry[0].MetaData.URL)
			panic("Not inserted")
		}

	}

	end := time.Now()
	diff := end.Sub(start)
	fmt.Println("TIME TAKEN: ", diff)

}

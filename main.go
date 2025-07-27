package main

import (
	"fmt"
	"time"

	"mySearchEngine/crawler"
	"mySearchEngine/embeddings"
	"mySearchEngine/storage"
	"mySearchEngine/utils"
)

const DBDRIVER string = "sqlite3"

func main() {

	var ch = make(chan storage.PagesMetaData, 16)

	start := time.Now()

	go func() {
		crawler.Crawl("https://www.wikipedia.org", &ch)
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
	defer embeddingService.Destroy()
	
	for value := range ch {
		fmt.Println(value.URL)
		dbEntry, err := embeddingService.GenerateBatchEmbeddings([]*storage.PagesMetaData{&value})
		if err != nil {
			utils.ErrorLogger.Println(err)
			panic(err)
		}
		err = storageManager.InsertEntriesIntoDB(dbEntry)

		if err != nil {
			utils.ErrorLogger.Println(err)
			panic(err)
		}
	}

	end := time.Now()
	diff := end.Sub(start)
	fmt.Println("TIME TAKEN: ", diff)

}

package main

import (
	"fmt"
	"time"

	"mySearchEngine/internal/crawler"
	"mySearchEngine/internal/embeddings"
	"mySearchEngine/internal/storage"
	"mySearchEngine/internal/utils"
)

const DBDRIVER string = "sqlite3"

func main() {

	var ch = make(chan storage.PagesMetaData, 16)

	start := time.Now()

	go func() {
		crawler.Crawl("https://personal.utdallas.edu/~vince/cs4365-honors/index.html", &ch)
		close(ch)
		fmt.Println("Web pages crawled: ", crawler.Count)
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

	count := 0

	for value := range ch {
		count++;
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

		fmt.Println(count)
	}

	end := time.Now()
	diff := end.Sub(start)
	fmt.Println("TIME TAKEN: ", diff)

}

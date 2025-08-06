package main

import (
	"fmt"
	"time"

	"mySearchEngine/internal/config"
	"mySearchEngine/internal/crawler"
	"mySearchEngine/internal/embeddings"
	"mySearchEngine/internal/storage"
	"mySearchEngine/internal/utils"
)

func main() {

	var ch = make(chan storage.PagesMetaData, 16)

	start := time.Now()

	go func() {
		urlsToCrawl := []string{
			"https://personal.utdallas.edu/~vince/cs4365-honors/index.html",
		}

		allowDomains := [][]string {
			{},
		}

		crawler.CrawlUrls(urlsToCrawl, allowDomains, 0, 4, &ch)
		close(ch)
		utils.InfoLogger.Println("Crawling finished")
	}()

	storageManager, err := storage.SetUpDataBase(config.DBDRIVER)
	if err != nil {
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

		utils.InfoLogger.Println(count)
		
	}
	utils.InfoLogger.Println(count)

	end := time.Now()
	diff := end.Sub(start)
	fmt.Println("TIME TAKEN: ", diff)

}

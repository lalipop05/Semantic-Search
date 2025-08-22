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

		crawler.CrawlUrls(urlsToCrawl, allowDomains, 0, 3, &ch)
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

	go func(sm *storage.StorageManager, es *embeddings.EmbeddingService) {
		for dbEntry := range *es.OutputChan {
			err := sm.InsertEntriesIntoDB([]*storage.PagesDBEntry{dbEntry})

			if err != nil {
				utils.ErrorLogger.Println(err)
				panic(err)
			}
			count++;
			fmt.Println(count)
		}
	}(storageManager, embeddingService)

	for value := range ch {
		err := embeddingService.AsyncGenerateBatchEmbeddings([]*storage.PagesMetaData{&value})
		if err != nil {
			utils.ErrorLogger.Println(err)
			panic(err)
		}
		
	}
	embeddingService.Wait()
	utils.InfoLogger.Println(count)

	end := time.Now()
	diff := end.Sub(start)
	fmt.Println("TIME TAKEN: ", diff)

}

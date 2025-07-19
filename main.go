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
	fmt.Println("------------------------------------")
	if err != nil {
		utils.ErrorLogger.Println(err)
		panic(err)
	}
	defer embeddingService.Close()
	batch := []*storage.PagesMetaData{}

	for value := range ch {
		storageManager.InsertMetaDataIntoDB(&value)
		batch = append(batch, &value)
		//fmt.Println(len(batch))
		if len(batch) == 10 {
			_, errs := embeddingService.GetBatchEmbeddings(batch)
			batch = []*storage.PagesMetaData{}
			if len(errs) != 0 {
				for _, err := range errs {
					utils.ErrorLogger.Println(err)
				}
				panic("Problem with embeddingService.GetBatchEmbeddings()")
			}

		}
	}

	end := time.Now()
	diff := end.Sub(start)
	fmt.Println("TIME TAKEN: ", diff)

}

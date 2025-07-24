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
	if (err != nil) {
		panic(utils.Error(err))
	}
	

	

	for value := range ch {
		idx, err := storageManager.InsertMetaDataIntoDB(&value)
		if (err != nil) {
			panic(utils.Error(err))
		}
		embeddedPages, err := embeddingService.GenerateBatchEmbeddings([]uint64{idx}, []*storage.PagesMetaData{&value})
		if (err != nil) {
			panic(utils.Error(err))
		}
		fmt.Println(embeddedPages)
		
	}

	end := time.Now()
	diff := end.Sub(start)
	fmt.Println("TIME TAKEN: ", diff)

}

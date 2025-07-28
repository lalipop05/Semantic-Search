package main

import (
	"fmt"
	"mySearchEngine/internal/embeddings"
	"mySearchEngine/internal/storage"
	"mySearchEngine/internal/utils"
)

const DBDRIVER = "sqlite3"

func main() {
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

	var input string = ""
	for {
		fmt.Println("What do you want to search for: ")
		fmt.Scan(&input)
		
		singleEmbedding, err := embeddingService.GenerateQueryEmbedding(input)

		if (err != nil) {
			if (err.Error() == "input too long") {
				fmt.Println("Please shorten query and search again")
				continue
			} else {
				utils.ErrorLogger.Println(err)
				panic(err)
			}
		}
		fmt.Println(len(singleEmbedding[0]))
		pages, err := storageManager.QueryDatabase(singleEmbedding[0])
		if (err != nil) {
			utils.ErrorLogger.Println(err)
			panic(err)
		}
		
		for _, d := range pages {
			fmt.Println("URL: ", d.URL)
			fmt.Println("Distance: ", d.Distance)
		}		
	}
}
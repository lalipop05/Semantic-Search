package main

import (
	"bufio"
	"fmt"
	"mySearchEngine/internal/embeddings"
	"mySearchEngine/internal/storage"
	"mySearchEngine/internal/utils"
	"os"
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
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("What do you want to search for: ")
		
		if !scanner.Scan() {
			break
		}

		input = scanner.Text()

		if (input == "") {
			break
		}
		
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
		
		pages, err := storageManager.QueryDatabase(singleEmbedding[0])
		if (err != nil) {
			utils.ErrorLogger.Println(err)
			panic(err)
		}
		
		for _, d := range pages {
			fmt.Println("URL: ", d.URL)
			fmt.Println("Distance: ", d.Distance)
			fmt.Println("Rowid: ", d.Rowid)
		}		
	}
}
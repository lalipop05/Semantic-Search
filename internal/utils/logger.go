package utils

import (
	"log"
	"os"
)

const LOGS_PATH = "./../logs"
const LOGS_FILENAME = "logs.txt"

var (
	InfoLogger    *log.Logger
	WarningLogger *log.Logger
	ErrorLogger   *log.Logger
	CrawlLogger   *log.Logger
)

func init() {
	err := os.MkdirAll(LOGS_PATH, 0755)
	if err != nil {
		log.Fatal(err)
	}
	file, err := os.OpenFile(LOGS_PATH+"/"+LOGS_FILENAME, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0666)
	if err != nil {
		log.Fatal(err)
	}

	CrawlLogger = log.New(file, "CRAWL: ", log.Ldate|log.Ltime|log.Lshortfile)
	WarningLogger = log.New(file, "WARNING: ", log.Ldate|log.Ltime|log.Lshortfile)
	ErrorLogger = log.New(file, "ERROR: ", log.Ldate|log.Ltime|log.Lshortfile)

	InfoLogger = log.New(os.Stdout, "INFO: ", log.Ltime|log.Lshortfile)
}
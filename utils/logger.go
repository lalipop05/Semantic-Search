package utils

import (
    "log"
    "os"
)

var (
    InfoLogger    *log.Logger
    WarningLogger *log.Logger
    ErrorLogger   *log.Logger
	CrawlLogger *log.Logger
)

func init() {
	err := os.MkdirAll("./utils/logs", 0755)
	if err != nil {
        log.Fatal(err)
    }
    file, err := os.OpenFile("./utils/logs/logs.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0666)
    if err != nil {
        log.Fatal(err)
    }
    
    InfoLogger = log.New(file, "INFO: ", log.Ldate|log.Ltime|log.Lshortfile)
    WarningLogger = log.New(file, "WARNING: ", log.Ldate|log.Ltime|log.Lshortfile)
    ErrorLogger = log.New(file, "ERROR: ", log.Ldate|log.Ltime|log.Lshortfile)

	CrawlLogger = log.New(os.Stdout, "CRAWL: ", log.Ltime|log.Lshortfile)
}


func Error(err error) error {
    ErrorLogger.Println(err)
    return err
}

func Warning(err error) error {
    WarningLogger.Println(err)
    return err
}


package storage

import (
	"time"
)

type PagesMetaData struct {
	URL             string
	Title           string
	Content         string
	MetaDescription string
	MetaKeyWords    string
	Headings        []string
	Hash            string
	ModifiedTime    time.Time
	Language        string
	ContentLength   int64
}

type PagesDBEntry struct {
	MetaData *PagesMetaData
	Embeddings []*[]float32
}


type InMemoryMetaData struct {
	DataBaseID   int32
	CrawledTime  time.Time
	ContentHash  string
}

type CrawledPagesCache map[string]InMemoryMetaData

func (cache CrawledPagesCache) Exists(data *PagesMetaData) bool {
	metaData, exists := cache[data.URL]

	if (!exists) {
		return false
	}

	if (data.Hash == metaData.ContentHash) {
		return true
	} else {
		return false
	}
}

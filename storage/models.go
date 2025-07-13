package storage

import "time"

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
	ContentLength int64
}

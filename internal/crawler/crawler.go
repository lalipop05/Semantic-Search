package crawler

import (
	"fmt"

	"mySearchEngine/internal/storage"
	"mySearchEngine/internal/utils"

	"github.com/gocolly/colly/v2"
)

// Called on every <a> tag encountered while crawling a web page
var onATagCallBack = func(e *colly.HTMLElement) {
	e.Request.Visit(e.Attr("href"))
}

// Called everytime a HTTP request is sent
var onRequestCallBack = func(r *colly.Request) {
	s := ""
	for i := 0; i < r.Depth-1; i++ {
		s += "  "
	}
	s += fmt.Sprintf("%d - Visiting %v", r.Depth, r.URL.String())
	utils.CrawlLogger.Println(s)
}

var onErrorCallBack = func(r *colly.Response, err error) {
	utils.WarningLogger.Printf("Request URL: %s failed with error: %v\n", r.Request.URL, err)
}

// Called on every HTML tag
// All of the pages data is sent though a channel
func onHTMLTagCallBack(ch *chan storage.PagesMetaData) func(e *colly.HTMLElement) {
	return func(e *colly.HTMLElement) {
		metaData := storage.PagesMetaData{
			URL:             e.Request.URL.String(),
			Title:           e.ChildText("title"),
			Content:         utils.Normalize(e.Text),
			MetaDescription: e.ChildAttr("meta[name='description']", "content"),
			MetaKeyWords:    e.ChildAttr("meta[name='keywords']", "content"),
			Headings:        e.ChildTexts("h1, h2, h3, h4, h5, h6"),
			Language:        e.ChildAttr("html", "lang"),
			Hash:            utils.NormalizeAndHash(e.Text),
			ContentLength:   int64(len(e.Text)),
		}
		*ch <- metaData
	}
}

// urls contains the urls to be visited
// allowedDomains contains a slice of allowed domains for each url
// outputChannel is the channel though which you will recieve the pages meta data
func CrawlUrls(urls []string, allowedDomains [][]string, parallelism int, maxDepth int, outputChannel *chan storage.PagesMetaData) {
	if len(urls) != len(allowedDomains) {
		utils.ErrorLogger.Fatal("len(url) is not equal to len(allowedDomains)")
	}

	isAsync := parallelism != 0
	limitRule := &colly.LimitRule{
		DomainGlob:  "*",
		Parallelism: parallelism,
	}

	// create a new collector of each url and visit each url individually
	for i := range urls {
		opts := []colly.CollectorOption{
			colly.MaxDepth(maxDepth),
			colly.Async(isAsync),
		}
		if (len(allowedDomains[i]) > 0) {
			opts = append(opts, colly.AllowedDomains(allowedDomains[i]...))
		}
		c := colly.NewCollector(
			opts...,
		)
		
		if (isAsync) {
			c.Limit(limitRule)
		}

		c.OnHTML("a[href]", onATagCallBack)

		c.OnHTML("html", onHTMLTagCallBack(outputChannel))

		c.OnRequest(onRequestCallBack)

		c.OnError(onErrorCallBack)

		c.Visit(urls[i])
		c.Wait()
	}
}

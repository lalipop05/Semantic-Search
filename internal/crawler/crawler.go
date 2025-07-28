package crawler

import (
	"fmt"

	"mySearchEngine/internal/storage"
	"mySearchEngine/internal/utils"

	"github.com/gocolly/colly/v2"
)

const MAXDEPTH = 1
const PARALLELISM = 4
func Crawl(url string, ch *chan storage.PagesMetaData) {

	c := initCrawler(MAXDEPTH, PARALLELISM)

	c.OnHTML("a[href]", onATagCallBack())

	c.OnHTML("html", onHTMLTagCallBack(ch))

	c.OnRequest(onRequestCallBack())

	c.Visit(url)
	c.Wait()
}

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

func onATagCallBack() func(e *colly.HTMLElement) {
	return func(e *colly.HTMLElement) {
		e.Request.Visit(e.Attr("href"))
	}
}

func initCrawler(maxDepth int, parallelism int) *colly.Collector {
	c := colly.NewCollector(
		colly.MaxDepth(maxDepth), 
		colly.Async(true),
		colly.AllowedDomains("www.geeksforgeeks.org"),
	)

	c.Limit(&colly.LimitRule{
		DomainGlob:  "*",
		Parallelism: parallelism,
	})

	return c
}

func onRequestCallBack() func(*colly.Request) {
	return func(r *colly.Request) {
		s := ""
		for i := 0; i < r.Depth-1; i++ {
			s += "  "
		}
		s += fmt.Sprintf("%d - Visiting %v", r.Depth, r.URL.String())
		utils.CrawlLogger.Println(s)
	}
}

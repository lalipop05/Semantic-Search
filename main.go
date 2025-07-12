package main

import (
	"fmt"
	"time"

	"github.com/gocolly/colly"
)

func crawl(url string, ch *chan string) uint32 {

	var count int = 0;
	c := colly.NewCollector(colly.MaxDepth(4), colly.Async(true))

	c.Limit(&colly.LimitRule{
		DomainGlob:  "*",
		Parallelism: 4,  // Number of concurrent requests
	})

	// Find and visit all links
	c.OnHTML("a[href]", func(e *colly.HTMLElement) {
		count++;
		e.Request.Visit(e.Attr("href"))
	})

	c.OnRequest(func(r *colly.Request) {
		for i := 0; i < r.Depth-1; i++ {
			fmt.Print("  ")
		}
		s := fmt.Sprintf("%d", r.Depth) + "- Visiting %v" + r.URL.String()
		*ch <- s
	})

	c.Visit(url)
	c.Wait();

	return uint32(count)
}

func main() {

	var ch = make(chan string, 100)

	start := time.Now()

	
	go func () {
		crawl("https://personal.utdallas.edu/~vince/cs4365-honors/index.html", &ch)
		close(ch)
	}()

	for values := range ch {
		fmt.Println(values)
	}
	
	end := time.Now()
	diff := end.Sub(start)
	fmt.Println(diff)
}

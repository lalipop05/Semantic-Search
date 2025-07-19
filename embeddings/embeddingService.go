package embeddings

import (
	"fmt"
	"mySearchEngine/storage"
	"mySearchEngine/utils"
	"strings"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
)

const BATCHSIZEINBYTES = 1e5
const MODELPATH = "sentence-transformers/all-MiniLM-L6-v2"

type EmbeddingService struct {
	session *hugot.Session
	fePipeline *pipelines.FeatureExtractionPipeline
}

func NewEmbeddingService() (*EmbeddingService, error) {
	session, err := hugot.NewGoSession()
	if (err != nil) {
		utils.ErrorLogger.Println("Failed to create hugot GoSession: ", err)
		return nil, err
	}
	downloadOptions := hugot.NewDownloadOptions()
	downloadOptions.OnnxFilePath = "onnx/model.onnx"  

	

	downloadedPath, err := hugot.DownloadModel(
		MODELPATH,
		"./models",
		downloadOptions,
	)
	fmt.Println("------------------------------------")
	if (err != nil) {
		session.Destroy()
		utils.ErrorLogger.Println("Failed to download model: ", err)
		return nil, err
	}

	fmt.Println("------------------------------------")

	config := hugot.FeatureExtractionConfig {
		ModelPath: downloadedPath,
		Name: "embeddings",
	}

	fePipeline, err := hugot.NewPipeline(session, config)
	if (err != nil) {
		session.Destroy()
		utils.ErrorLogger.Println("Failed to set up pipeline: ", err)
		return nil, err
	}

	return &EmbeddingService{
		session: session,
		fePipeline: fePipeline,
	}, err

}

func (es *EmbeddingService) GetBatchEmbeddings(metaData []*storage.PagesMetaData) ([][]float32, []error) {
	embeddings := [][]float32{}
	errs := []error{}

	batch := []string{}
	var batchSize int = 0

	for _, data := range metaData {
		s := generateEmbeddingString(data)
		batch = append(batch, s)
		batchSize += len(s)
		if (batchSize >= BATCHSIZEINBYTES) {
			responseEmbeddings, err := es.processBatch(batch)
			if (err != nil) {
				errs = append(errs, err)
			} else {
				embeddings = append(embeddings, responseEmbeddings...)
			}
			batch = []string{}
			batchSize = 0
		}
	}
	if (batchSize != 0) {
		responseEmbeddings, err := es.processBatch(batch)
		if (err != nil) {
			errs = append(errs, err)
		} else {
			embeddings = append(embeddings, responseEmbeddings...)
		}
	}

	return embeddings, errs
}

func (es *EmbeddingService) processBatch(texts []string) ([][]float32, error) {
	output, err := es.fePipeline.RunPipeline(texts)
	if (err != nil) {
		utils.ErrorLogger.Println("Could not run pipeline: ", err)
		return nil, err
	}

	embs := output.Embeddings
	for i, emb := range embs {
		fmt.Println(i, emb)
	}

	return embs, nil
}

func (es *EmbeddingService) Close() {
	es.session.Destroy()
}

func generateEmbeddingString(data *storage.PagesMetaData) string {
	var builder strings.Builder
	if (data.Title != "") {
		builder.WriteString(data.Title)
		builder.WriteString(". ")
	}
	if (data.MetaDescription != "") {
		builder.WriteString(data.MetaDescription)
		builder.WriteString(". ")
	}
	if (data.MetaKeyWords != "") {
		builder.WriteString(data.MetaKeyWords)
		builder.WriteString(". ")
	}
	if (data.Headings != nil) {
		for _, heading := range data.Headings {
			builder.WriteString(heading)
			builder.WriteString(",")
		}
		builder.WriteString(". ")
	}
	builder.WriteString(data.Content)
	return builder.String()
}
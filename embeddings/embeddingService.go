package embeddings

import (
	"mySearchEngine/storage"
	"mySearchEngine/utils"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

const MODELPATH = "C:/College_UTD/Summer_2024/Models/Qdrant_bge-small-en-v1.5-onnx-Q/"
const TOKENIZER_FILE_NAME = "tokenizer.json"
const MODEL_FILENAME = "model_optimized.onnx"

const ONNX_RUNTIME_PATH = "C:/College_UTD/Summer_2024/onnxruntime-win-x64-1.22.1/lib/onnxruntime.dll"

type EmbeddingService struct {
	tokenizer *tokenizer.Tokenizer
}

func NewEmbeddingService() (*EmbeddingService, error) {
	tk, err := pretrained.FromFile(MODELPATH + TOKENIZER_FILE_NAME)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	paddingParams := getPaddingParams()
	tk.WithPadding(paddingParams)

	ort.SetSharedLibraryPath(ONNX_RUNTIME_PATH)

	err = ort.InitializeEnvironment()
	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	return &EmbeddingService{
		tokenizer: tk,
	}, err
}

func getPaddingParams() *tokenizer.PaddingParams {
	paddingStrat := tokenizer.NewPaddingStrategy(tokenizer.WithFixed(MODEL_SEQUENCE_LEN))
	paddingParams := tokenizer.PaddingParams{
		Strategy:  *paddingStrat,
		Direction: tokenizer.Right,
	}
	return &paddingParams
}

func (es *EmbeddingService) Destroy() {
	defer ort.DestroyEnvironment()
}

func (es *EmbeddingService) GenerateBatchEmbeddings(data []*storage.PagesMetaData) ([]*storage.PagesDBEntry, error) {
	tokenizedPages, err := es.GetBatchTokens(data)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	for _, tokenizedPage := range tokenizedPages {
		for _, embedding := range tokenizedPage.Tokens {
			utils.CrawlLogger.Println("tokenized len/input len: ", embedding.Len())
		}
	}

	embeddedPages, err := es.ProduceEmbeddings(tokenizedPages)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}
	for _, embeddedPage := range embeddedPages {
		utils.CrawlLogger.Println(len(embeddedPage.Embeddings))
		for _, embedding := range embeddedPage.Embeddings {
			utils.CrawlLogger.Println("Embedded len/output len: ", len(embedding))
		}
		
	}

	dbEntries := make([]*storage.PagesDBEntry, 0, len(data))
	for i := range data {
		dbEntry := storage.PagesDBEntry{
			MetaData:   data[i],
			Embeddings: embeddedPages[i].getPointersToEmbeddings(),
		}
		dbEntries = append(dbEntries, &dbEntry)
	}
	return dbEntries, nil
}

type TokenizedPage struct {
	Tokens []*tokenizer.Encoding
}

type EmbeddedPage struct {
	Embeddings [][]float32
}

func (eb *EmbeddedPage) getPointersToEmbeddings() []*[]float32 {
	res := make([]*[]float32, 0, len(eb.Embeddings))
	for i := range eb.Embeddings {
		res = append(res, &eb.Embeddings[i])
	}
	return res
}

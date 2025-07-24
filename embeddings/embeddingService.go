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
	tk, err := pretrained.FromFile(MODELPATH+TOKENIZER_FILE_NAME)
	if (err != nil) {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	paddingParams := getPaddingParams()
    tk.WithPadding(paddingParams)

	ort.SetSharedLibraryPath(ONNX_RUNTIME_PATH)

	err = ort.InitializeEnvironment()
	if err != nil {
		utils.Error(err)
		return nil, err
	}


	return &EmbeddingService{
		tokenizer: tk,
	}, err
}

func getPaddingParams() *tokenizer.PaddingParams {
	paddingStrat := tokenizer.NewPaddingStrategy(tokenizer.WithFixed(MODEL_SEQUENCE_LEN))
    paddingParams := tokenizer.PaddingParams {
        Strategy: *paddingStrat,
        Direction: tokenizer.Right,
    }
	return &paddingParams
}

func (es *EmbeddingService) Destroy() {
	defer ort.DestroyEnvironment()
}

func (es *EmbeddingService) GenerateBatchEmbeddings(idx []uint64, data []*storage.PagesMetaData) ([]*EmbeddedPage, error) {
	tokenizedPages, err := es.GetBatchTokens(idx, data)
	if (err != nil) {
		utils.Error(err)
		return nil, err
	}
	embeddedPages, err := es.ProduceEmbeddings(tokenizedPages)
	if (err != nil) {
		utils.Error(err)
		return nil, err
	}
	return embeddedPages, nil
}


type TokenizedPage struct {
	Idx uint64
	Tokens []*tokenizer.Encoding
}

type EmbeddedPage struct {
	Idx uint64
	Embeddings [][]float32
}


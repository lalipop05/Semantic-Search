package embeddings

import (
	"fmt"
	"mySearchEngine/internal/config"
	"mySearchEngine/internal/storage"
	"mySearchEngine/internal/utils"
	"sync"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

type EmbeddingService struct {
	tokenizer *tokenizer.Tokenizer
	ortObj    *OrtObject
	wg		  sync.WaitGroup
}

func NewEmbeddingService() (*EmbeddingService, error) {
	// load tokenizer config file from path
	tk, err := pretrained.FromFile(config.MODELPATH + config.TOKENIZER_FILE_NAME)
	if err != nil {
		return nil, fmt.Errorf("could not load tokenizer config from file path %s\n%w", config.MODELPATH+config.TOKENIZER_FILE_NAME, err)
	}

	// padding params to make sure that output is always the same length of config.MODEL_SEQUENCE_LEN
	paddingParams := getPaddingParams()
	tk.WithPadding(paddingParams)

	inputShape := ort.NewShape(config.MODEL_BATCH_SIZE, config.MODEL_SEQUENCE_LEN)
	outputShape := ort.NewShape(config.MODEL_BATCH_SIZE, config.MODEL_SEQUENCE_LEN, config.MODEL_OUTPUT_VECTOR_LEN)

	err = SetUpOrtEnv()
	if (err != nil) {
		return nil, fmt.Errorf("failed to set up ort environment\n%w", err)
	}

	// var wg sync.WaitGroup
	// wg.Add(config.MODEL_INSTANCES)
	// SetUpAsyncEmbeddingProd(&wg, config.MODEL_INSTANCES)

	// handles inputs and outputs to the onnx model and running it
	ortObj := NewOrtObject(&inputShape, &outputShape)

	return &EmbeddingService{
		tokenizer: tk,
		ortObj:    ortObj,
	}, err
}

func getPaddingParams() *tokenizer.PaddingParams {
	// pads output until output is of length config.MODEL_SEQUENCE_LEN
	paddingStrat := tokenizer.NewPaddingStrategy(tokenizer.WithFixed(config.MODEL_SEQUENCE_LEN))
	paddingParams := tokenizer.PaddingParams{
		Strategy:  *paddingStrat,
		Direction: tokenizer.Right,
	}
	return &paddingParams
}

func (es *EmbeddingService) Destroy() {
	es.ortObj.Destroy()
	ort.DestroyEnvironment()
}

func (es *EmbeddingService) GenerateBatchEmbeddings(data []*storage.PagesMetaData) ([]*storage.PagesDBEntry, error) {

	tokenizedPages, err := es.GetBatchTokens(data)
	if err != nil {
		return nil, fmt.Errorf("failed to get batch tokens\n%w", err)
	}

	embeddedPages, err := es.ProduceEmbeddings(tokenizedPages)
	if err != nil {
		return nil, fmt.Errorf("failed to get bathc embeddings\n%w", err)
	}

	dbEntries := make([]*storage.PagesDBEntry, 0, len(data))
	for i := range data {
		dbEntry := storage.PagesDBEntry{
			MetaData:   data[i],
			Embeddings: embeddedPages[i].Embeddings,
		}
		dbEntries = append(dbEntries, &dbEntry)
	}
	return dbEntries, nil
}

func (es *EmbeddingService) GenerateQueryEmbedding(query string) ([][]float32, error) {
	if query == "" {
		return nil, fmt.Errorf("empty string")
	}

	tokens, err := es.GetQueryTokens(query)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}
	if len(tokens) > 1 {
		return nil, fmt.Errorf("input too long")
	}

	embeddings, err := es.ProduceQueryEmbeddings(tokens)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	return embeddings, nil
}

type TokenizedPage struct {
	Tokens []*tokenizer.Encoding
	Chunks int
}

type EmbeddedPage struct {
	Embeddings [][]float32
	Chunks     int
}

type OrtObject struct {
	InputTensor       *ort.Tensor[int64]
	AttentionTensor   *ort.Tensor[int64]
	TokenTypeIdTensor *ort.Tensor[int64]
	OutputTensor      *ort.Tensor[float32]
	Session           *ort.AdvancedSession
}

func SetUpOrtEnv() error {
	// set up onnx runtime library by providing the downloaded onnxruntime.dll path
	ort.SetSharedLibraryPath(config.ONNX_RUNTIME_PATH)

	// Create enviornment
	err := ort.InitializeEnvironment()
	return err
}

// func SetUpAsyncEmbeddingProd(wg *sync.WaitGroup, n int, onError func(error)) {
// 	go func() {
		
// 	}()
// }


func NewOrtObject(inputShape *ort.Shape, outputShape *ort.Shape) *OrtObject {
	// input tensors
	inTensor, attTensor, tokenTypeTensor, err := getInputTensors(*inputShape)
	if err != nil {
		utils.ErrorLogger.Fatal(err)
	}
	// output tensor
	outTensor, err := getOutputTensor(*outputShape)
	if err != nil {
		utils.ErrorLogger.Fatal(err)
	}

	// session to run the loaded model
	session, err := ort.NewAdvancedSession(config.MODELPATH+config.MODEL_FILENAME,
		config.MODEL_INPUT_NAMES, config.MODEL_OUTPUT_NAMES,
		[]ort.ArbitraryTensor{inTensor, attTensor, tokenTypeTensor}, []ort.ArbitraryTensor{outTensor}, nil)
	if err != nil {
		utils.ErrorLogger.Fatalf("Failed to create inference session: %v", err)
	}

	return &OrtObject{
		inTensor,
		attTensor,
		tokenTypeTensor,
		outTensor,
		session,
	}
}

func (ortObj *OrtObject) Destroy() {
	ortObj.InputTensor.Destroy()
	ortObj.AttentionTensor.Destroy()
	ortObj.TokenTypeIdTensor.Destroy()
	ortObj.OutputTensor.Destroy()
	ortObj.Session.Destroy()
}

// Create and return 3 tensors specified by the shape passed in
func getInputTensors(shape ort.Shape) (*ort.Tensor[int64], *ort.Tensor[int64], *ort.Tensor[int64], error) {
	inputTensor, err := ort.NewEmptyTensor[int64](shape)
	if err != nil {
		utils.ErrorLogger.Fatalf("Failed to create input tensor: %v", err)
		return nil, nil, nil, err
	}

	attentionTensor, err := ort.NewEmptyTensor[int64](shape)
	if err != nil {
		utils.ErrorLogger.Fatalf("Failed to create attention tensor: %v", err)
		return nil, nil, nil, err
	}

	tokenTypeIdTensor, err := ort.NewEmptyTensor[int64](shape)
	if err != nil {
		utils.ErrorLogger.Fatalf("Failed to create type id tensor: %v", err)
		return nil, nil, nil, err
	}

	return inputTensor, attentionTensor, tokenTypeIdTensor, nil
}

// Create and return 1 output tensor defined by the shape passed in
func getOutputTensor(shape ort.Shape) (*ort.Tensor[float32], error) {
	outputTensor, err := ort.NewEmptyTensor[float32](shape)
	if err != nil {
		utils.ErrorLogger.Fatalf("Failed to create output tensor: %v", err)
		return nil, err
	}
	return outputTensor, err
}

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

type Pair[U any, V any] struct {
	First  U
	Second V
}

type EmbeddingService struct {
	tokenizer     *tokenizer.Tokenizer
	wg            *sync.WaitGroup
	embeddingChan *chan *Pair[*TokenizedPage, *storage.PagesMetaData]
	OutputChan    *chan *storage.PagesDBEntry
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

	err = SetUpOrtEnv()
	if err != nil {
		return nil, fmt.Errorf("failed to set up ort environment\n%w", err)
	}

	var wg sync.WaitGroup

	embeddingChan := make(chan *Pair[*TokenizedPage, *storage.PagesMetaData], 16)

	outputChan := make(chan *storage.PagesDBEntry, 4)

	es := &EmbeddingService{
		tokenizer:     tk,
		wg:            &wg,
		embeddingChan: &embeddingChan,
		OutputChan:    &outputChan,
	}

	es.SetUpAsyncEmbeddingProd(config.MODEL_INSTANCES, func(err error) {
		utils.ErrorLogger.Println("problem with async embedding\n%w", err)
	})

	return es, nil

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
	ort.DestroyEnvironment()
}

func (es *EmbeddingService) AsyncGenerateBatchEmbeddings(data []*storage.PagesMetaData) error {
	tokenizedPages, err := es.GetBatchTokens(data)
	if err != nil {
		return fmt.Errorf("failed to get batch tokens\n%w", err)
	}

	for i := range tokenizedPages {
		*es.embeddingChan <- &Pair[*TokenizedPage, *storage.PagesMetaData]{tokenizedPages[i], data[i]}
	}

	return nil
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

	inputShape := ort.NewShape(config.MODEL_BATCH_SIZE, config.MODEL_SEQUENCE_LEN)
	outputShape := ort.NewShape(config.MODEL_BATCH_SIZE, config.MODEL_SEQUENCE_LEN, config.MODEL_OUTPUT_VECTOR_LEN)
	ortObject := NewOrtObject(&inputShape, &outputShape)

	embeddings, err := es.ProduceQueryEmbeddings(tokens, ortObject)
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

func (es *EmbeddingService) SetUpAsyncEmbeddingProd(n int, onError func(error)) {
	for ; n > 0; n-- {
		i := n
		utils.InfoLogger.Println("Setting up ", i)
		go func() {
			inputShape := ort.NewShape(config.MODEL_BATCH_SIZE, config.MODEL_SEQUENCE_LEN)
			outputShape := ort.NewShape(config.MODEL_BATCH_SIZE, config.MODEL_SEQUENCE_LEN, config.MODEL_OUTPUT_VECTOR_LEN)
			ortObject := NewOrtObject(&inputShape, &outputShape)
			es.wg.Add(1)
			for pair := range *es.embeddingChan {
				tokenizedPage := pair.First
				metaData := pair.Second
				pageEmbeddings, err := es.producePageEmbeddings(tokenizedPage, ortObject)
				if err != nil {
					onError(err)
				}
				embeddedPage := EmbeddedPage{
					Embeddings: pageEmbeddings,
					Chunks:     tokenizedPage.Chunks,
				}
				utils.InfoLogger.Println("DONE BY: ", i)
				*es.OutputChan <- &storage.PagesDBEntry{
					MetaData:   metaData,
					Embeddings: embeddedPage.Embeddings,
					Chunks:     embeddedPage.Chunks,
				}
			}
			ortObject.Destroy()
			es.wg.Done()
		}()
	}

}

func (es *EmbeddingService) Wait() {
	close(*es.embeddingChan)
	es.wg.Wait()
}

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

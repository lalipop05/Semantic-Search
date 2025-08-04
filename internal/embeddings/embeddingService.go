package embeddings

import (
	"fmt"
	"mySearchEngine/internal/storage"
	"mySearchEngine/internal/utils"

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
	ortObj    *OrtObject
}

func NewEmbeddingService() (*EmbeddingService, error) {
	tk, err := pretrained.FromFile(MODELPATH + TOKENIZER_FILE_NAME)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	paddingParams := getPaddingParams()
	tk.WithPadding(paddingParams)

	inputShape := ort.NewShape(1, MODEL_SEQUENCE_LEN)
	outputShape := ort.NewShape(1, MODEL_SEQUENCE_LEN, MODEL_OUTPUT_VECTOR_LEN)

	ortObj := NewOrtObject(&inputShape, &outputShape)

	return &EmbeddingService{
		tokenizer: tk,
		ortObj: ortObj,
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
	es.ortObj.Destroy()
}

func (es *EmbeddingService) GenerateBatchEmbeddings(data []*storage.PagesMetaData) ([]*storage.PagesDBEntry, error) {
	tokenizedPages, err := es.GetBatchTokens(data)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}
	fmt.Println("Tokens Produced")
	embeddedPages, err := es.ProduceEmbeddings(tokenizedPages)
	if err != nil {
		utils.ErrorLogger.Println(err)
		return nil, err
	}
	fmt.Println("Embeddings Produced")
	dbEntries := make([]*storage.PagesDBEntry, 0, len(data))
	for i := range data {
		dbEntry := storage.PagesDBEntry{
			MetaData:   data[i],
			Embeddings: embeddedPages[i].getPointersToEmbeddings(),
		}
		dbEntries = append(dbEntries, &dbEntry)
	}
	fmt.Println("Returning...")
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

type OrtObject struct {
	InputTensor       *ort.Tensor[int64]
	AttentionTensor   *ort.Tensor[int64]
	TokenTypeIdTensor *ort.Tensor[int64]
	OutputTensor      *ort.Tensor[float32]
	Session           *ort.AdvancedSession
}

func NewOrtObject(inputShape *ort.Shape, outputShape *ort.Shape) *OrtObject {

	ort.SetSharedLibraryPath(ONNX_RUNTIME_PATH)

	err := ort.InitializeEnvironment()
	if err != nil {
		utils.ErrorLogger.Fatal(err)
	}

	inTensor, attTensor, tokenTypeTensor, err := getInputTensors(*inputShape)
	if err != nil {
		utils.ErrorLogger.Fatal(err)
	}

	outTensor, err := getOutputTensor(*outputShape)
	if err != nil {
		utils.ErrorLogger.Fatal(err)
	}

	session, err := ort.NewAdvancedSession(MODELPATH+MODEL_FILENAME,
		MODEL_INPUT_NAMES, MODEL_OUTPUT_NAMES,
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
	ort.DestroyEnvironment()
}

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

	tokenTypeIdTensor, err := ort.NewTensor(shape, make([]int64, MODEL_SEQUENCE_LEN))
	if err != nil {
		utils.ErrorLogger.Fatalf("Failed to create type id tensor: %v", err)
		return nil, nil, nil, err
	}

	return inputTensor, attentionTensor, tokenTypeIdTensor, nil
}

func getOutputTensor(shape ort.Shape) (*ort.Tensor[float32], error) {
	outputTensor, err := ort.NewEmptyTensor[float32](shape)
	if err != nil {
		utils.ErrorLogger.Fatalf("Failed to create output tensor: %v", err)
		return nil, err
	}
	return outputTensor, err
}

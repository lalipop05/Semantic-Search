package embeddings

import (
	"mySearchEngine/internal/utils"

	"github.com/sugarme/tokenizer"
	ort "github.com/yalue/onnxruntime_go"
)

const MODEL_SEQUENCE_LEN = 512
const MODEL_OUTPUT_VECTOR_LEN = 384

var MODEL_INPUT_NAMES = []string{"input_ids", "attention_mask", "token_type_ids"}
var MODEL_OUTPUT_NAMES = []string{"last_hidden_state"}

func populateTensor(tensor *ort.Tensor[int64], values []int) {
	if len(values) != int(tensor.GetShape()[1]) {
		utils.WarningLogger.Println("Tensor shape and values shape does not match up!")
		return
	}
	newSlice := tensor.GetData()
	for i, value := range values {
		newSlice[i] = int64(value)
	}
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

func (es *EmbeddingService) ProduceQueryEmbeddings(tokens []*tokenizer.Encoding) ([][]float32, error) {
	var response [][]float32

	inputShape := ort.NewShape(1, MODEL_SEQUENCE_LEN)

	inputTensor, attentionTensor, tokenTypeIdTensor, err := getInputTensors(inputShape)
	if (err != nil) {
		utils.ErrorLogger.Println(err)
		return nil, err
	}
	defer inputTensor.Destroy()
	defer attentionTensor.Destroy()
	defer tokenTypeIdTensor.Destroy()

	outputShape := ort.NewShape(1, MODEL_SEQUENCE_LEN, MODEL_OUTPUT_VECTOR_LEN)

	outputTensor, err := getOutputTensor(outputShape)
	if (err != nil) {
		utils.ErrorLogger.Println(err)
		return nil, err
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(MODELPATH+MODEL_FILENAME,
		MODEL_INPUT_NAMES, MODEL_OUTPUT_NAMES,
		[]ort.Value{inputTensor, attentionTensor, tokenTypeIdTensor}, []ort.Value{outputTensor}, nil)
	if err != nil {
		utils.ErrorLogger.Fatalf("Failed to create inference session: %v", err)
		return nil, err
	}
	defer session.Destroy()

	for _, token := range tokens {
		populateTensor(inputTensor, token.Ids)
		populateTensor(attentionTensor, token.AttentionMask)

		err = session.Run()
		if (err != nil) {
			utils.ErrorLogger.Println(err)
			return nil, err
		}
		outputData := outputTensor.GetData()
		response = append(response, outputData[:MODEL_OUTPUT_VECTOR_LEN])
	}
	return response, nil
}

func (es *EmbeddingService) ProduceEmbeddings(tokenizedPages []*TokenizedPage) ([]*EmbeddedPage, error) {
	inputShape := ort.NewShape(1, MODEL_SEQUENCE_LEN)

	inputTensor, attentionTensor, tokenTypeIdTensor, err := getInputTensors(inputShape)
	if (err != nil) {
		utils.ErrorLogger.Println(err)
		return nil, err
	}
	defer inputTensor.Destroy()
	defer attentionTensor.Destroy()
	defer tokenTypeIdTensor.Destroy()

	outputShape := ort.NewShape(1, MODEL_SEQUENCE_LEN, MODEL_OUTPUT_VECTOR_LEN)

	outputTensor, err := getOutputTensor(outputShape)
	if (err != nil) {
		utils.ErrorLogger.Println(err)
		return nil, err
	}
	defer outputTensor.Destroy()

	embeddedPages := make([]*EmbeddedPage, 0, len(tokenizedPages))

	session, err := ort.NewAdvancedSession(MODELPATH+MODEL_FILENAME,
		MODEL_INPUT_NAMES, MODEL_OUTPUT_NAMES,
		[]ort.Value{inputTensor, attentionTensor, tokenTypeIdTensor}, []ort.Value{outputTensor}, nil)

	if err != nil {
		utils.ErrorLogger.Fatalf("Failed to create inference session: %v", err)
		return nil, err
	}
	defer session.Destroy()

	for _, tokenizedPage := range tokenizedPages {
		pageEmbeddings := make([][]float32, 0, len(tokenizedPage.Tokens))
		for _, tokens := range tokenizedPage.Tokens {
			populateTensor(inputTensor, tokens.Ids)
			populateTensor(attentionTensor, tokens.AttentionMask)

			err = session.Run()
			if err != nil {
				utils.ErrorLogger.Println(err)
				return nil, err
			}

			outputData := outputTensor.GetData()
			pageEmbeddings = append(pageEmbeddings, outputData[:MODEL_OUTPUT_VECTOR_LEN])
		}
		embeddedPage := EmbeddedPage{
			Embeddings: pageEmbeddings,
		}
		embeddedPages = append(embeddedPages, &embeddedPage)
	}

	return embeddedPages, nil
}

package embeddings

import (
	"mySearchEngine/utils"

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

func (es *EmbeddingService) ProduceEmbeddings(tokenizedPages []*TokenizedPage) ([]*EmbeddedPage, error) {
	inputShape := ort.NewShape(1, MODEL_SEQUENCE_LEN)

	inputTensor, err := ort.NewEmptyTensor[int64](inputShape)
	if err != nil {
		utils.ErrorLogger.Fatalf("Failed to create input tensor: %v", err)
	}
	defer inputTensor.Destroy()

	attentionTensor, err := ort.NewEmptyTensor[int64](inputShape)
	if err != nil {
		utils.ErrorLogger.Fatalf("Failed to create attention tensor: %v", err)
		return nil, err
	}
	defer attentionTensor.Destroy()

	tokenTypeIdTensor, err := ort.NewTensor(inputShape, make([]int64, MODEL_SEQUENCE_LEN))
	if err != nil {
		utils.ErrorLogger.Fatalf("Failed to create type id tensor: %v", err)
		return nil, err
	}
	defer tokenTypeIdTensor.Destroy()

	outputShape := ort.NewShape(1, MODEL_SEQUENCE_LEN, MODEL_OUTPUT_VECTOR_LEN)

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		utils.ErrorLogger.Fatalf("Failed to create output tensor: %v", err)
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

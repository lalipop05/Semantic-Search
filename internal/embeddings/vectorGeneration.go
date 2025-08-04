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

func (es *EmbeddingService) ProduceQueryEmbeddings(tokens []*tokenizer.Encoding) ([][]float32, error) {
	var response [][]float32

	ortObject := es.ortObj

	for _, token := range tokens {
		populateTensor(ortObject.InputTensor, token.Ids)
		populateTensor(ortObject.AttentionTensor, token.AttentionMask)

		err := ortObject.Session.Run()
		if (err != nil) {
			utils.ErrorLogger.Println(err)
			return nil, err
		}
		outputData := ortObject.OutputTensor.GetData()

		temp := make([]float32, MODEL_OUTPUT_VECTOR_LEN)
		copy(temp, outputData[:MODEL_OUTPUT_VECTOR_LEN])
		response = append(response, temp)
	}
	return response, nil
}

func (es *EmbeddingService) ProduceEmbeddings(tokenizedPages []*TokenizedPage) ([]*EmbeddedPage, error) {
	
	embeddedPages := make([]*EmbeddedPage, 0, len(tokenizedPages))

	ortObject := es.ortObj
	for _, tokenizedPage := range tokenizedPages {
		pageEmbeddings := make([][]float32, 0, len(tokenizedPage.Tokens))
		for _, tokens := range tokenizedPage.Tokens {
			populateTensor(ortObject.InputTensor, tokens.Ids)
			populateTensor(ortObject.AttentionTensor, tokens.AttentionMask)


			err := ortObject.Session.Run()
			if err != nil {
				utils.ErrorLogger.Println(err)
				return nil, err
			}

			outputData := ortObject.OutputTensor.GetData()
			temp := make([]float32, MODEL_OUTPUT_VECTOR_LEN)
			copy(temp, outputData)
			pageEmbeddings = append(pageEmbeddings, temp)
		}
		embeddedPage := EmbeddedPage{
			Embeddings: pageEmbeddings,
		}
		embeddedPages = append(embeddedPages, &embeddedPage)
	}

	return embeddedPages, nil
}

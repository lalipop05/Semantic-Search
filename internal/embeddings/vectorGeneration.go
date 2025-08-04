package embeddings

import (
	"mySearchEngine/internal/utils"

	"github.com/sugarme/tokenizer"
	ort "github.com/yalue/onnxruntime_go"
)

const MODEL_SEQUENCE_LEN = 512
const MODEL_OUTPUT_VECTOR_LEN = 384
const MODEL_BATCH_SIZE = 1

var MODEL_INPUT_NAMES = []string{"input_ids", "attention_mask", "token_type_ids"}
var MODEL_OUTPUT_NAMES = []string{"last_hidden_state"}

func populateTensor(tensor *ort.Tensor[int64], tokenizedPage *TokenizedPage, classifier func(*tokenizer.Encoding) []int, index int) {
	tensorData := tensor.GetData()
	
	end := min(len(tokenizedPage.Tokens), index+int(MODEL_BATCH_SIZE))

	for ; index < end; index++ {
		currentEncoding := tokenizedPage.Tokens[index]
		if len(classifier(currentEncoding)) != int(tensor.GetShape()[1]) {
			utils.ErrorLogger.Fatal("Tensor shape and values shape does not match up!")
		}
		for i, value := range classifier(currentEncoding) {
			tensorData[i] = int64(value)
		}
	}
}

func getIds(tp *tokenizer.Encoding) []int {
	return tp.Ids
}

func getAttMask(tp *tokenizer.Encoding) []int {
	return tp.AttentionMask
}

// func (es *EmbeddingService) ProduceQueryEmbeddings(tokens []*tokenizer.Encoding) ([][]float32, error) {
// 	var response [][]float32

// 	ortObject := es.ortObj

// 	for _, token := range tokens {
// 		populateTensor(ortObject.InputTensor, TokenizedPage)
// 		populateTensor(ortObject.AttentionTensor, token.AttentionMask)

// 		err := ortObject.Session.Run()
// 		if (err != nil) {
// 			utils.ErrorLogger.Println(err)
// 			return nil, err
// 		}
// 		outputData := ortObject.OutputTensor.GetData()

// 		temp := make([]float32, MODEL_OUTPUT_VECTOR_LEN)
// 		copy(temp, outputData[:MODEL_OUTPUT_VECTOR_LEN])
// 		response = append(response, temp)
// 	}
// 	return response, nil
// }

func (es *EmbeddingService) producePageEmbeddings(tokenizedPage *TokenizedPage) [][]float32 {
	ortObject := es.ortObj
	chunks := len(tokenizedPage.Tokens)
	pageEmbeddings := make([][]float32, 0, chunks)

	for i := 0; i < chunks; i+=int(MODEL_BATCH_SIZE) {
		populateTensor(ortObject.InputTensor, tokenizedPage, getIds, i)
		populateTensor(ortObject.AttentionTensor, tokenizedPage, getAttMask, i)

		err := ortObject.Session.Run()
		if err != nil {
			utils.ErrorLogger.Fatal(err)
		}

		outputData := ortObject.OutputTensor.GetData()
		increment := MODEL_OUTPUT_VECTOR_LEN*MODEL_SEQUENCE_LEN
		end := min(chunks-i, int(MODEL_BATCH_SIZE))
		for j := 0; j < end; j++ {
			temp := make([]float32, MODEL_OUTPUT_VECTOR_LEN)
			start := j*increment
			copy(temp, outputData[start:start+MODEL_OUTPUT_VECTOR_LEN])
			pageEmbeddings = append(pageEmbeddings, temp)
		}

	}

	return pageEmbeddings
}

func (es *EmbeddingService) ProduceEmbeddings(tokenizedPages []*TokenizedPage) ([]*EmbeddedPage, error) {
	embeddedPages := make([]*EmbeddedPage, 0, len(tokenizedPages))

	for _, tokenizedPage := range tokenizedPages {
		pageEmbeddings := es.producePageEmbeddings(tokenizedPage)
		embeddedPage := EmbeddedPage{
			Embeddings: pageEmbeddings,
		}
		embeddedPages = append(embeddedPages, &embeddedPage)
	}

	return embeddedPages, nil
}

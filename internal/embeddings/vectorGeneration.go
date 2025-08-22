package embeddings

import (
	"fmt"
	"mySearchEngine/internal/config"
	"mySearchEngine/internal/utils"

	"github.com/sugarme/tokenizer"
	ort "github.com/yalue/onnxruntime_go"
)

func populateTensor(tensor *ort.Tensor[int64], tokenizedPage *TokenizedPage, classifier func(*tokenizer.Encoding) []int, index int) {
	tensorData := tensor.GetData()
	
	end := min(len(tokenizedPage.Tokens), index+int(config.MODEL_BATCH_SIZE))

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

func (es *EmbeddingService) ProduceQueryEmbeddings(tokens []*tokenizer.Encoding, ortObject *OrtObject) ([][]float32, error) {
	chunks := len(tokens)
	var response [][]float32


	for i := 0; i < chunks; i++ {
		populateTensor(ortObject.InputTensor, &TokenizedPage{tokens, len(tokens)}, getIds, i)
		populateTensor(ortObject.AttentionTensor, &TokenizedPage{tokens, len(tokens)}, getAttMask, i)

		err := ortObject.Session.Run()
		if err != nil {
			utils.ErrorLogger.Fatal(err)
		}

		outputData := ortObject.OutputTensor.GetData()
		increment := config.MODEL_OUTPUT_VECTOR_LEN*config.MODEL_SEQUENCE_LEN
		end := min(chunks-i, int(config.MODEL_BATCH_SIZE))
		for j := range end {
			temp := make([]float32, config.MODEL_OUTPUT_VECTOR_LEN)
			start := j*increment
			copy(temp, outputData[start:start+config.MODEL_OUTPUT_VECTOR_LEN])
			response = append(response, temp)
		}
	}
	return response, nil
}

func (es *EmbeddingService) producePageEmbeddings(tokenizedPage *TokenizedPage, ortObject *OrtObject) ([][]float32, error) {
	chunks := len(tokenizedPage.Tokens)
	pageEmbeddings := make([][]float32, 0, chunks)

	for i := 0; i < chunks; i+=int(config.MODEL_BATCH_SIZE) {
		populateTensor(ortObject.InputTensor, tokenizedPage, getIds, i)
		populateTensor(ortObject.AttentionTensor, tokenizedPage, getAttMask, i)

		err := ortObject.Session.Run()
		if err != nil {
			return nil, fmt.Errorf("could not run ort session\n%w", err)
		}

		outputData := ortObject.OutputTensor.GetData()
		increment := config.MODEL_OUTPUT_VECTOR_LEN*config.MODEL_SEQUENCE_LEN
		end := min(chunks-i, int(config.MODEL_BATCH_SIZE))
		for j := range end {
			temp := make([]float32, config.MODEL_OUTPUT_VECTOR_LEN)
			start := j*increment
			copy(temp, outputData[start:start+config.MODEL_OUTPUT_VECTOR_LEN])
			pageEmbeddings = append(pageEmbeddings, temp)
		}

	}

	return pageEmbeddings, nil
}

func (es *EmbeddingService) ProduceEmbeddings(tokenizedPages []*TokenizedPage) ([]*EmbeddedPage, error) {
	embeddedPages := make([]*EmbeddedPage, 0, len(tokenizedPages))
	inputShape := ort.NewShape(config.MODEL_BATCH_SIZE, config.MODEL_SEQUENCE_LEN)
	outputShape := ort.NewShape(config.MODEL_BATCH_SIZE, config.MODEL_SEQUENCE_LEN, config.MODEL_OUTPUT_VECTOR_LEN)
	ortObject := NewOrtObject(&inputShape, &outputShape)
	defer ortObject.Destroy()
	for _, tokenizedPage := range tokenizedPages {
		pageEmbeddings, err := es.producePageEmbeddings(tokenizedPage, ortObject)
		if (err != nil) {
			return nil, fmt.Errorf("coutl not produce page embeddings")
		}
		embeddedPage := EmbeddedPage{
			Embeddings: pageEmbeddings,
			Chunks: tokenizedPage.Chunks,
		}
		embeddedPages = append(embeddedPages, &embeddedPage)
	}
	
	return embeddedPages, nil
}

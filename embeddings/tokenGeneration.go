package embeddings

import (
	"mySearchEngine/storage"
	"mySearchEngine/utils"
	"strings"

	"github.com/sugarme/tokenizer"
)

const TOKENIZER_STRIDE_LEN = 64

func (es *EmbeddingService) GetBatchTokens(idx []uint64, data []*storage.PagesMetaData) ([]*TokenizedPage, error) {
	n := len(idx)
	if (n == 0) {
		return nil, nil
	}
	batchEncodings, err := es.generateBatchTokens(data)
	if (err != nil) {
		utils.ErrorLogger.Println(err)
		return nil, err
	}

	tokenizedPages := make([]*TokenizedPage, 0, n)

	for i, enc := range batchEncodings {
		processed := es.ProcessEncoding(enc)
		if (processed != nil) {
			tokenizedPages = append(tokenizedPages, &TokenizedPage{idx[i], processed})
		}
	}
	return tokenizedPages, nil
}

func (es *EmbeddingService) ProcessEncoding(encoding tokenizer.Encoding) []*tokenizer.Encoding {
	const startToken int = 101
	const endToken int = 102
	n := encoding.Len()
	if (n == MODEL_SEQUENCE_LEN) {
		return []*tokenizer.Encoding{&encoding}
	} else {
		splitEncodings := make([]*tokenizer.Encoding, 0)
		
		start := 1
		for ; start+MODEL_SEQUENCE_LEN-2 < n-1; {
			temp := tokenizer.Encoding{
				Ids: []int{startToken},
				AttentionMask: []int{1},
			}
			temp.Ids = append(temp.Ids, encoding.Ids[start:start+MODEL_SEQUENCE_LEN-2]...)
			temp.AttentionMask = append(temp.AttentionMask, encoding.AttentionMask[start:start+MODEL_SEQUENCE_LEN-2]...)
			temp.Ids = append(temp.Ids, endToken)
			temp.AttentionMask = append(temp.AttentionMask, 1)
			splitEncodings = append(splitEncodings, &temp)
			
			start += MODEL_SEQUENCE_LEN-2-TOKENIZER_STRIDE_LEN
		}
		temp := tokenizer.Encoding{
			Ids: []int{startToken},
			AttentionMask: []int{1},
		}
		temp.Ids = append(temp.Ids, encoding.Ids[start:]...)
		temp.AttentionMask = append(temp.AttentionMask, encoding.AttentionMask[start:]...)
		
		temp = tokenizer.PadEncodings([]tokenizer.Encoding{temp}, *getPaddingParams())[0]

		return append(splitEncodings, &temp)
	}
}


func (es *EmbeddingService) generateBatchTokens(data []*storage.PagesMetaData) ([]tokenizer.Encoding, error) {
	var encodingInput []tokenizer.EncodeInput = make([]tokenizer.EncodeInput, 0, len(data))
	for i, pageData := range data {
		if (pageData == nil) {
			utils.WarningLogger.Printf("pageData %v is nil", i)
			continue
		}
		text := ToText(pageData)
		inputSeq := tokenizer.NewInputSequence(text)
		encodingInput = append(encodingInput, tokenizer.NewSingleEncodeInput(inputSeq))
	}
	encoding, err := es.tokenizer.EncodeBatch(encodingInput, true)
	if (err != nil) {
		utils.ErrorLogger.Println(err)
		return nil, err
	}
	return encoding, nil
}

func ToText(page *storage.PagesMetaData) string {
	var parts []string

	if page.Title != "" {
		parts = append(parts, page.Title)
	}
	if page.MetaDescription != "" {
		parts = append(parts, page.MetaDescription)
	}
	if page.MetaKeyWords != "" {
		parts = append(parts, page.MetaKeyWords)
	}
	if len(page.Headings) > 0 {
		parts = append(parts, strings.Join(page.Headings, ", "))
	}
	metaText := strings.Join(parts, ". ")

	var builder strings.Builder
	builder.WriteString(metaText)

	if metaText != "" && page.Content != "" {
		builder.WriteString(". ")
	}
	builder.WriteString(page.Content)

	return builder.String()

}


package utils

import (
	"fmt"
	"regexp"
	"strings"

	"hash/fnv"
)

func NormalizeAndHash(content string) string {
	normalized := Normalize(content)
	return contentHash(normalized)
}

// Produces a hash for the input to check for changes in content
func contentHash(content string) string {
	hashFunc := fnv.New64a()
	hashFunc.Write([]byte(content))
	return fmt.Sprintf("%x", hashFunc.Sum64())
}

// Remove whitespace
func Normalize(content string) string {
	normalized := strings.TrimSpace(content)
	normalized = regexp.MustCompile(`\s+`).ReplaceAllString(normalized, " ")
	return normalized
}


package config

const MODELPATH = "C:/College_UTD/Summer_2024/Models/Qdrant_bge-small-en-v1.5-onnx-Q/"
const TOKENIZER_FILE_NAME = "tokenizer.json"
const MODEL_FILENAME = "model_optimized.onnx"

const ONNX_RUNTIME_PATH = "C:/College_UTD/Summer_2024/onnxruntime-win-x64-1.22.1/lib/onnxruntime.dll"

const TOKENIZER_STRIDE_LEN = 64

const MODEL_SEQUENCE_LEN = 512
const MODEL_OUTPUT_VECTOR_LEN = 384
const MODEL_BATCH_SIZE = 1

var MODEL_INPUT_NAMES = []string{"input_ids", "attention_mask", "token_type_ids"}
var MODEL_OUTPUT_NAMES = []string{"last_hidden_state"}

const DATABASE_DIR_PATH = "./../database"
const DATABASE_NAME = "tester.db"

const DBDRIVER string = "sqlite3"

const MODEL_INSTANCES = 6
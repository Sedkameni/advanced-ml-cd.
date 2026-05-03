"""
Sentiment Analysis API using ONNX model.
FastAPI application with async support for concurrent request handling.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global model session
model_session: Optional[ort.InferenceSession] = None
tokenizer_vocab: Optional[dict] = None

# Base directory (resolved relative to this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.onnx")

# -------------------------------------------------------------------
# Simple whitespace tokenizer + vocab (mirrors lab activity approach)
# -------------------------------------------------------------------

def build_simple_vocab():
    """Build a small fixed vocabulary for demonstration purposes."""
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    positive_words = [
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "love", "best", "awesome", "perfect", "happy", "positive", "brilliant",
        "outstanding", "superb", "magnificent", "delightful", "pleasant",
    ]
    negative_words = [
        "bad", "terrible", "awful", "horrible", "worst", "hate", "disgusting",
        "poor", "disappointing", "dreadful", "negative", "pathetic", "lousy",
        "miserable", "atrocious", "unpleasant",
    ]
    neutral_words = [
        "the", "a", "an", "is", "was", "are", "were", "it", "this", "that",
        "movie", "film", "book", "product", "service", "experience", "very",
        "not", "no", "yes", "but", "and", "or", "so", "quite", "rather",
        "somewhat", "extremely", "absolutely", "totally", "really", "just",
    ]
    all_words = special + positive_words + negative_words + neutral_words
    return {word: idx for idx, word in enumerate(all_words)}


def tokenize(text: str, vocab: dict, max_length: int = 128) -> List[int]:
    """Simple whitespace tokenizer returning token IDs."""
    tokens = text.lower().split()
    unk_id = vocab.get("[UNK]", 1)
    ids = [vocab.get(tok, unk_id) for tok in tokens]
    ids = ids[:max_length]
    ids += [vocab.get("[PAD]", 0)] * (max_length - len(ids))
    return ids


def create_dummy_onnx_model():
    """Create a minimal ONNX model for sentiment analysis when no model file exists."""
    try:
        import onnx
        from onnx import helper, TensorProto

        input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None, 128])
        output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 2])

        cast_node = helper.make_node("Cast", inputs=["input_ids"], outputs=["cast_out"],
                                     to=TensorProto.FLOAT)
        mean_node = helper.make_node("ReduceMean", inputs=["cast_out"], outputs=["mean_out"],
                                     axes=[1])

        weight_data = np.random.randn(128, 2).astype(np.float32) * 0.01
        bias_data = np.array([0.1, -0.1], dtype=np.float32)

        weight_init = helper.make_tensor("weight", TensorProto.FLOAT, [128, 2],
                                         weight_data.flatten().tolist())
        bias_init = helper.make_tensor("bias", TensorProto.FLOAT, [2],
                                       bias_data.tolist())

        gemm_node = helper.make_node("Gemm", inputs=["mean_out", "weight", "bias"],
                                     outputs=["logits"])

        graph = helper.make_graph(
            [cast_node, mean_node, gemm_node],
            "sentiment_graph",
            [input_ids],
            [output],
            initializer=[weight_init, bias_init],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7

        # Use relative path instead of hardcoded /app
        os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)
        onnx.save(model, MODEL_PATH)
        logger.info("Dummy ONNX model created successfully at %s.", MODEL_PATH)
        return True
    except Exception as exc:
        logger.error("Could not create dummy ONNX model: %s", exc)
        return False


# -------------------------------------------------------------------
# Lifespan (startup / shutdown)
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_session, tokenizer_vocab
    logger.info("Loading ONNX model from %s...", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        logger.warning("Model file not found at %s. Creating dummy model.", MODEL_PATH)
        create_dummy_onnx_model()
    try:
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 4
        sess_options.intra_op_num_threads = 4
        model_session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options)
        logger.info("ONNX model loaded. Inputs: %s", [i.name for i in model_session.get_inputs()])
    except Exception as exc:
        logger.error("Failed to load ONNX model: %s", exc)
        model_session = None
    tokenizer_vocab = build_simple_vocab()
    logger.info("Tokenizer vocabulary built (%d tokens).", len(tokenizer_vocab))
    yield  # Application runs here
    logger.info("Shutting down — releasing model resources.")
    model_session = None


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(
    title="Sentiment Analysis API",
    description="ML-powered sentiment analysis using an ONNX model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                      description="Text to analyse for sentiment.")

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace only.")
        return v


class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=32,
                              description="List of texts (max 32).")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        for i, t in enumerate(v):
            if not t or not t.strip():
                raise ValueError(f"texts[{i}] must not be blank.")
            if len(t) > 10000:
                raise ValueError(f"texts[{i}] exceeds maximum length of 10,000 characters.")
        return v


class SentimentResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    positive_score: float
    negative_score: float
    processing_time_ms: float


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResult]
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def run_inference(texts: List[str]):
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()

    input_ids = np.array(
        [tokenize(t, tokenizer_vocab) for t in texts], dtype=np.int64
    )

    pad_id = tokenizer_vocab.get("[PAD]", 0)
    attention_mask = (input_ids != pad_id).astype(np.int64)

    input_names = [inp.name for inp in model_session.get_inputs()]

    inputs = {}
    if "input_ids" in input_names:
        inputs["input_ids"] = input_ids
    if "attention_mask" in input_names:
        inputs["attention_mask"] = attention_mask

    logits = model_session.run(None, inputs)[0]

    probs = softmax(logits)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    results = []
    per_item_ms = elapsed_ms / len(texts)

    for i, text in enumerate(texts):
        neg_score, pos_score = float(probs[i, 0]), float(probs[i, 1])
        sentiment = "positive" if pos_score >= neg_score else "negative"
        confidence = max(pos_score, neg_score)

        results.append(
            SentimentResult(
                text=text,
                sentiment=sentiment,
                confidence=round(confidence, 4),
                positive_score=round(pos_score, 4),
                negative_score=round(neg_score, 4),
                processing_time_ms=round(per_item_ms, 3),
            )
        )

    return results, elapsed_ms


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health-check endpoint for liveness / readiness probes."""
    return HealthResponse(
        status="healthy" if model_session is not None else "degraded",
        model_loaded=model_session is not None,
        version="1.0.0",
    )


@app.get("/", tags=["System"])
async def root():
    return {"message": "Sentiment Analysis API — visit /docs for the interactive UI."}


@app.post("/predict", response_model=SentimentResult, tags=["Inference"])
async def predict(request: SentimentRequest, req: Request):
    """Analyse the sentiment of a single text."""
    logger.info("Single inference request from %s", req.client.host if req.client else "unknown")
    results, _ = run_inference([request.text])
    return results[0]


@app.post("/predict/batch", response_model=BatchSentimentResponse, tags=["Inference"])
async def predict_batch(request: BatchSentimentRequest, req: Request):
    """Analyse sentiment for a batch of texts (max 32)."""
    logger.info("Batch inference request (%d texts) from %s",
                len(request.texts), req.client.host if req.client else "unknown")
    results, elapsed = run_inference(request.texts)
    return BatchSentimentResponse(
        results=results,
        total_processing_time_ms=round(elapsed, 3),
    )
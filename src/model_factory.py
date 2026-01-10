import os

from config import (
    GEMINI_API_KEY_ENV,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    GROQ_API_KEY_ENV,
    GROQ_BASE_URL,
    GROQ_MODEL,
    HF_MODEL_NAME,
    MODEL_BACKEND,
    RANDOM_SEED,
    VLLM_BASE_URL,
    VLLM_MODEL,
)
from model_interface import (
    GeminiModel,
    GroqModel,
    HFLocalModel,
    MockModel,
    ModelInterface,
    VLLMModel,
)


def get_model() -> ModelInterface:
    if MODEL_BACKEND == "mock":
        return MockModel(seed=RANDOM_SEED)
    if MODEL_BACKEND == "hf-local":
        return HFLocalModel(model_name=HF_MODEL_NAME)
    if MODEL_BACKEND == "vllm":
        return VLLMModel(base_url=VLLM_BASE_URL, model=VLLM_MODEL)
    if MODEL_BACKEND == "groq":
        api_key = os.getenv(GROQ_API_KEY_ENV, "")
        if not api_key:
            raise RuntimeError(f"Missing API key in env var {GROQ_API_KEY_ENV}")
        return GroqModel(base_url=GROQ_BASE_URL, model=GROQ_MODEL, api_key=api_key)
    if MODEL_BACKEND == "gemini":
        api_key = os.getenv(GEMINI_API_KEY_ENV, "")
        if not api_key:
            raise RuntimeError(f"Missing API key in env var {GEMINI_API_KEY_ENV}")
        return GeminiModel(base_url=GEMINI_BASE_URL, model=GEMINI_MODEL, api_key=api_key)
    raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND}")

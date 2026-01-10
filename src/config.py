import os

DATASET_DIR = "MedQA_Reason"
TRACE_COUNT = 3
MODEL_BACKEND = "hf-local"  # "mock", "hf-local", "vllm", "groq", "gemini"
RANDOM_SEED = 7
HF_MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
# Other SmolLM options:
# - HuggingFaceTB/SmolLM-360M
# - HuggingFaceTB/SmolLM-1.7B
VLLM_BASE_URL = "http://localhost:8000"
VLLM_MODEL = "smollm"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
THROTTLE_SECONDS = 1.0
DEBUG_DIR = "debug"
GENERATED_TRACES_PATH = os.path.join(DEBUG_DIR, "generated_traces.json")
OPTIMIZATION_DEBUG_PATH = os.path.join(DEBUG_DIR, "optimization_debug.json")
LOG_FILE = os.path.join(DEBUG_DIR, "run_logs.txt")
DUMMY_ACCEPT_FIRST_N = True

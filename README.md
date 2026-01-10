# MedQA RLHF Bench

## Run the UI
1) Activate the venv:
```
source .venv/bin/activate
```
2) Launch Streamlit:
```
python3 -m streamlit run src/app.py
```

Artifacts and logs are written under `debug/`.

## Config knobs
All settings live in `src/config.py`.

Key parameters:
- `MODEL_BACKEND`: `"hf-local"`, `"mock"`, `"vllm"`, `"groq"`, `"gemini"`.
- `HF_MODEL_NAME`: Hugging Face model ID for local HF runs.
- `TRACE_COUNT`: number of traces shown and ranked.
- `THROTTLE_SECONDS`: sleep between trace generation calls.
- `DUMMY_ACCEPT_FIRST_N`: if true, accept the first `TRACE_COUNT` traces (no verification).
- `DEBUG_DIR`: directory for logs/artifacts.
- `GENERATED_TRACES_PATH`: cached traces path (default: `debug/generated_traces.json`).
- `OPTIMIZATION_DEBUG_PATH`: optimizer debug dump (default: `debug/optimization_debug.json`).
- `LOG_FILE`: optional log file path (default: `debug/run_logs.txt`).

## Prompt template
The prompt template lives in `src/prompt.txt`. Update it to change the instructions
passed to the model.

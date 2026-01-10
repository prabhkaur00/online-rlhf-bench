import json
import os
import re
import time
from typing import Dict, List, Sequence

from config import (
    DATASET_DIR,
    DUMMY_ACCEPT_FIRST_N,
    GENERATED_TRACES_PATH,
    THROTTLE_SECONDS,
    TRACE_COUNT,
)
from dataset import Sample, load_samples
from model_factory import get_model
from model_interface import ModelInterface


ANSWER_RE = re.compile(
    r'Final\s+Answer\s*:\s*\{\s*"answer"\s*:\s*"([A-D])"\s*\}',
    re.IGNORECASE
)


def extract_json_answer(text: str) -> str:
    """
    Extract answer from the final line format: Final Answer: {"answer":"A"}
    
    Returns:
        The letter (A/B/C/D) if found in correct format, empty string otherwise
    """
    match = ANSWER_RE.search(text)
    return match.group(1) if match else ""


def build_prompt(sample: Sample) -> str:
    return (
        f"You are a medical expert analyzing a clinical case. Read the following question carefully:\n\n"
        f"{sample.question}\n\n"
        "Please provide your analysis using the following structure:\n"
        "1. **Key Clinical Information**: Identify the most relevant symptoms, signs, and patient characteristics\n"
        "2. **Differential Diagnosis**: Consider possible conditions and explain your reasoning\n"
        "3. **Evidence Evaluation**: Analyze which findings support or refute each possibility\n"
        "4. **Clinical Reasoning**: Explain step-by-step how you arrive at your conclusion\n"
        "5. **Final Determination**: State your final answer with brief justification\n\n"
        "After your complete reasoning, you MUST end with exactly this format on the final line:\n"
        'Final Answer: {"answer":"X"}\n\n'
        "Where X is one of: A, B, C, or D\n"
        "Do not add any text after the final answer line."
    )


def rejection_sample_traces(
    model: ModelInterface,
    sample: Sample,
    num_traces: int,
    log_fn=None,
    throttle_seconds: float = THROTTLE_SECONDS,
    dummy_accept_first_n: bool = DUMMY_ACCEPT_FIRST_N,
) -> List[str]:
    traces: List[str] = []
    seen = set()
    attempts = 0
    options = ["A", "B", "C", "D"]

    while len(traces) < num_traces:
        attempts += 1
        if log_fn:
            log_fn(f"{sample.sample_id}: request {attempts}")
        result = model.generate_trace(build_prompt(sample), options)
        if log_fn:
            log_fn(f"{sample.sample_id}: raw trace {attempts} - {result.trace}")
        if dummy_accept_first_n:
            traces.append(result.trace)
            if log_fn:
                log_fn(f"{sample.sample_id}: dummy accept {attempts}")
            if len(traces) >= num_traces:
                break
            if throttle_seconds > 0:
                time.sleep(throttle_seconds)
            continue
        answer = extract_json_answer(result.trace)
        verified = answer == sample.answer
        if log_fn:
            status = "verified" if verified else "not verified"
            answer_label = answer or "none"
            log_fn(
                f"{sample.sample_id}: received answer {attempts} - {answer_label} - {status}"
            )
        if not verified:
            if throttle_seconds > 0:
                time.sleep(throttle_seconds)
            continue
        if result.trace in seen:
            if throttle_seconds > 0:
                time.sleep(throttle_seconds)
            continue
        seen.add(result.trace)
        traces.append(result.trace)
        if throttle_seconds > 0:
            time.sleep(throttle_seconds)

    return traces


def generate_traces_for_samples(
    samples: Sequence[Sample],
    model: ModelInterface,
    num_traces: int = TRACE_COUNT,
    log_fn=None,
    throttle_seconds: float = THROTTLE_SECONDS,
) -> Dict[str, List[str]]:
    trace_map: Dict[str, List[str]] = {}
    for sample in samples:
        if log_fn:
            log_fn(f"Starting {sample.sample_id}")
        trace_map[sample.sample_id] = rejection_sample_traces(
            model,
            sample,
            num_traces,
            log_fn=log_fn,
            throttle_seconds=throttle_seconds,
        )
    return trace_map


def load_or_generate_traces(
    traces_path: str,
    samples: Sequence[Sample],
    model: ModelInterface,
    log_fn=None,
    throttle_seconds: float = THROTTLE_SECONDS,
) -> Dict[str, List[str]]:
    if os.path.exists(traces_path):
        if log_fn:
            log_fn(f"Loaded cached traces from {traces_path}")
        with open(traces_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    traces_dir = os.path.dirname(traces_path)
    if traces_dir:
        os.makedirs(traces_dir, exist_ok=True)
    traces = generate_traces_for_samples(
        samples,
        model,
        TRACE_COUNT,
        log_fn=log_fn,
        throttle_seconds=throttle_seconds,
    )
    with open(traces_path, "w", encoding="utf-8") as handle:
        json.dump(traces, handle, indent=2)
    return traces


def main() -> None:
    samples = load_samples(DATASET_DIR)
    model = get_model()
    traces = generate_traces_for_samples(samples, model, TRACE_COUNT)
    traces_path = GENERATED_TRACES_PATH
    traces_dir = os.path.dirname(traces_path)
    if traces_dir:
        os.makedirs(traces_dir, exist_ok=True)
    with open(traces_path, "w", encoding="utf-8") as handle:
        json.dump(traces, handle, indent=2)


if __name__ == "__main__":
    main()

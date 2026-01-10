import json
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

from config import DATASET_DIR


OPTION_RE = re.compile(r"^[A-D]\)")


@dataclass
class Sample:
    sample_id: str
    question: str
    answer: str
    reasoning: str


def load_samples(dataset_dir: str = DATASET_DIR) -> List[Sample]:
    samples: List[Sample] = []
    for name in sorted(os.listdir(dataset_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(dataset_dir, name)
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        samples.append(
            Sample(
                sample_id=name,
                question=data["Questions"],
                answer=data["Answer"],
                reasoning=data.get("Reasoning", ""),
            )
        )
    return samples


def split_question(text: str) -> Tuple[str, List[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    options = [line for line in lines if OPTION_RE.match(line)]
    stem_lines = [line for line in lines if line not in options]
    stem = " ".join(stem_lines).strip()
    return stem, options

import json
import random
import re
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\\sA-Za-z0-9]")


class SimpleTokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab or {"<pad>": 0, "<unk>": 1}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def build_from_texts(self, texts: Sequence[str]) -> None:
        for text in texts:
            for token in TOKEN_RE.findall(text):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        tokens = TOKEN_RE.findall(text)
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]

    def batch_encode(
        self, texts: Sequence[str], pad_to_max: bool = True
    ) -> Dict[str, torch.Tensor]:
        encoded = [self.encode(t) for t in texts]
        max_len = max(len(seq) for seq in encoded) if pad_to_max else None
        padded = []
        attention = []
        for seq in encoded:
            if pad_to_max:
                pad_len = max_len - len(seq)
                padded.append(seq + [self.vocab["<pad>"]] * pad_len)
                attention.append([1] * len(seq) + [0] * pad_len)
            else:
                padded.append(seq)
                attention.append([1] * len(seq))
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


class ToyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits)


@dataclass
class TraceResult:
    trace: str
    answer: str


class ModelInterface(ABC):
    @abstractmethod
    def generate_trace(
        self, prompt: str, answer_choices: Sequence[str], forced_answer: Optional[str] = None
    ) -> TraceResult:
        pass

    @abstractmethod
    def batch_encode(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def build_torch_model(self, vocab_size: int) -> nn.Module:
        pass


class MockModel(ModelInterface):
    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self.tokenizer = SimpleTokenizer()
        self.templates = [
            "Key clues point to option {ans}. The travel history and neuro deficit fit embolic risk.",
            "Matching symptoms to the options leaves {ans} as the only consistent answer.",
            "The pattern is most compatible with {ans} when weighing the options.",
            "Given the differential, {ans} best explains the findings.",
            "The most likely diagnosis among choices is {ans} based on the presentation.",
        ]

    def generate_trace(
        self, prompt: str, answer_choices: Sequence[str], forced_answer: Optional[str] = None
    ) -> TraceResult:
        answer = forced_answer or self.rng.choice(list(answer_choices))

        template = self.rng.choice(self.templates)
        reasoning = template.format(ans=answer)
        trace = f"{reasoning}\nFinal Answer: {json.dumps({'answer': answer})}"
        return TraceResult(trace=trace, answer=answer)

    def get_tokenizer(self) -> SimpleTokenizer:
        return self.tokenizer

    def batch_encode(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        if len(self.tokenizer.vocab) <= 2:
            self.tokenizer.build_from_texts(texts)
        return self.tokenizer.batch_encode(texts, pad_to_max=True)

    def build_torch_model(self, vocab_size: int) -> nn.Module:
        return ToyCausalLM(vocab_size=vocab_size)


class HFLocalModel(ModelInterface):
    def __init__(self, model_name: str):
        try:
            from transformers.models.auto.modeling_auto import (  # type: ignore
                AutoModelForCausalLM,
            )
            from transformers.models.auto.tokenization_auto import (  # type: ignore
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError("transformers is required for hf-local backend") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=None, low_cpu_mem_usage=False
        )
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.device = torch.device("cpu")

    def generate_trace(
        self, prompt: str, answer_choices: Sequence[str], forced_answer: Optional[str] = None
    ) -> TraceResult:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = forced_answer or ""
        trace = f"{text}\nFinal Answer: {json.dumps({'answer': answer})}"
        return TraceResult(trace=trace, answer=answer)

    def get_tokenizer(self) -> SimpleTokenizer:
        raise RuntimeError("HFLocalModel uses its own tokenizer; not supported in this demo")

    def batch_encode(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    def build_torch_model(self, vocab_size: int) -> nn.Module:
        return self.model


def _post_json(url: str, payload: Dict[str, object], headers: Dict[str, str]) -> Dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


class APIModelBase(ModelInterface):
    def __init__(self):
        self.tokenizer = SimpleTokenizer()

    def batch_encode(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        if len(self.tokenizer.vocab) <= 2:
            self.tokenizer.build_from_texts(texts)
        return self.tokenizer.batch_encode(texts, pad_to_max=True)

    def build_torch_model(self, vocab_size: int) -> nn.Module:
        return ToyCausalLM(vocab_size=vocab_size)


class VLLMModel(APIModelBase):
    def __init__(self, base_url: str, model: str):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate_trace(
        self, prompt: str, answer_choices: Sequence[str], forced_answer: Optional[str] = None
    ) -> TraceResult:
        instruction = (
            "Return your reasoning and end with JSON like {\"answer\":\"A\"}."
        )
        if forced_answer:
            instruction += f" Use answer {forced_answer}."
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": f"{prompt}\n{instruction}"}],
            "temperature": 0.7,
            "max_tokens": 200,
        }
        headers = {"Content-Type": "application/json"}
        response = _post_json(f"{self.base_url}/v1/chat/completions", payload, headers)
        content = response["choices"][0]["message"]["content"]
        answer = forced_answer or ""
        trace = (
            f"{content}\nFinal Answer: {json.dumps({'answer': answer})}"
            if forced_answer
            else content
        )
        return TraceResult(trace=trace, answer=answer)


class GroqModel(APIModelBase):
    def __init__(self, base_url: str, model: str, api_key: str):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def generate_trace(
        self, prompt: str, answer_choices: Sequence[str], forced_answer: Optional[str] = None
    ) -> TraceResult:
        instruction = (
            "Return your reasoning and end with JSON like {\"answer\":\"A\"}."
        )
        if forced_answer:
            instruction += f" Use answer {forced_answer}."
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": f"{prompt}\n{instruction}"}],
            "temperature": 0.7,
            "max_tokens": 200,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = _post_json(f"{self.base_url}/chat/completions", payload, headers)
        content = response["choices"][0]["message"]["content"]
        answer = forced_answer or ""
        trace = (
            f"{content}\nFinal Answer: {json.dumps({'answer': answer})}"
            if forced_answer
            else content
        )
        return TraceResult(trace=trace, answer=answer)


class GeminiModel(APIModelBase):
    def __init__(self, base_url: str, model: str, api_key: str):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def generate_trace(
        self, prompt: str, answer_choices: Sequence[str], forced_answer: Optional[str] = None
    ) -> TraceResult:
        instruction = (
            "Return your reasoning and end with JSON like {\"answer\":\"A\"}."
        )
        if forced_answer:
            instruction += f" Use answer {forced_answer}."
        payload = {
            "contents": [{"role": "user", "parts": [{"text": f"{prompt}\n{instruction}"}]}]
        }
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        response = _post_json(url, payload, headers)
        content = response["candidates"][0]["content"]["parts"][0]["text"]
        answer = forced_answer or ""
        trace = (
            f"{content}\nFinal Answer: {json.dumps({'answer': answer})}"
            if forced_answer
            else content
        )
        return TraceResult(trace=trace, answer=answer)

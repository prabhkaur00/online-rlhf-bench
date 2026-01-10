import builtins
import copy
import importlib
import json
import os
import typing
from typing import Dict, List, Sequence, Any

import torch

from config import OPTIMIZATION_DEBUG_PATH
from model_interface import ModelInterface


def build_batch(model: ModelInterface, traces: Sequence[str]) -> Dict[str, torch.Tensor]:
    encoded = model.batch_encode(traces)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_trace_inputs(model: ModelInterface, traces: Sequence[str]) -> List[Dict[str, torch.Tensor]]:
    inputs: List[Dict[str, torch.Tensor]] = []
    for trace in traces:
        encoded = model.batch_encode([trace])
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        inputs.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }
        )
    return inputs


def run_update(
    model: ModelInterface, traces: Sequence[str], ranking: Sequence[int]
) -> Dict[str, float]:
    if len(traces) != 3 or len(ranking) != 3:
        raise ValueError("Expected exactly 3 traces and 3 ranking indices")

    def tensor_to_list(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        return value

    print("DPO input traces:")
    for idx, trace in enumerate(traces):
        print(f"  trace[{idx}] = {trace}")
    print(f"DPO ranking indices: {list(ranking)}")

    batch = build_batch(model, traces)
    print(
        "DPO batch shapes:",
        {
            "input_ids": tuple(batch["input_ids"].shape),
            "attention_mask": tuple(batch["attention_mask"].shape),
            "labels": tuple(batch["labels"].shape),
        },
    )
    vocab_size = int(batch["input_ids"].max().item()) + 1
    policy_model = model.build_torch_model(vocab_size=vocab_size)
    reference_model = copy.deepcopy(policy_model)
    policy_model.train()
    yw_idx = torch.tensor([ranking[0]], dtype=torch.long)
    yl_idx = torch.tensor([ranking[2]], dtype=torch.long)
    print(f"DPO yw_idx={yw_idx.tolist()} yl_idx={yl_idx.tolist()}")

    # optimizer.py uses Optional in annotations without importing it.
    if not hasattr(builtins, "Optional"):
        builtins.Optional = typing.Optional
    optimizer_mod = importlib.import_module("optimizer")
    dpo = optimizer_mod.DPOOptimizer()
    dpo_losses, _, dpo_metrics = dpo.compute_dpo_loss(
        policy_model=policy_model,
        reference_model=reference_model,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        yw_idxs=yw_idx,
        yl_idxs=yl_idx,
    )

    rewards = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float)
    reward_order = torch.zeros_like(rewards)
    reward_order[ranking[0]] = rewards[0]
    reward_order[ranking[1]] = rewards[1]
    reward_order[ranking[2]] = rewards[2]

    grpo = optimizer_mod.GRPOOptimizer()
    trace_inputs = build_trace_inputs(model, traces)
    trace_lengths = [int(inp["input_ids"].shape[1]) for inp in trace_inputs]
    print(f"GRPO rewards: {reward_order.tolist()}")
    print(f"GRPO trace lengths: {trace_lengths}")
    grpo_loss, grpo_metrics, _ = grpo.compute_grpo_loss(
        policy_model=policy_model,
        reference_model=reference_model,
        traces_inputs=trace_inputs,
        rewards=reward_order,
        old_token_logps_list=None,
    )

    # Detailed debug dump for manual verification.
    pi_logps = dpo.get_log_probs(
        policy_model, batch["input_ids"], batch["attention_mask"], batch["labels"]
    )
    with torch.no_grad():
        ref_logps = dpo.get_log_probs(
            reference_model, batch["input_ids"], batch["attention_mask"], batch["labels"]
        )
    pi_yw_logps = pi_logps[yw_idx]
    pi_yl_logps = pi_logps[yl_idx]
    ref_yw_logps = ref_logps[yw_idx]
    ref_yl_logps = ref_logps[yl_idx]
    pi_logratios = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps
    dpo_logits = dpo.config.beta * (pi_logratios - ref_logratios)
    if dpo.config.label_smoothing > 0:
        dpo_losses_detail = (
            -torch.nn.functional.logsigmoid(dpo_logits) * (1 - dpo.config.label_smoothing)
            - torch.nn.functional.logsigmoid(-dpo_logits) * dpo.config.label_smoothing
        )
    else:
        dpo_losses_detail = -torch.nn.functional.logsigmoid(dpo_logits)
    dpo_rewards = dpo.config.beta * (pi_logps - ref_logps)

    rewards_mean = reward_order.mean()
    rewards_std = reward_order.std()
    if rewards_std < 1e-8:
        advantages = torch.zeros_like(reward_order)
    else:
        advantages = (reward_order - rewards_mean) / (rewards_std + 1e-8)

    grpo_traces_debug = []
    for i, trace_input in enumerate(trace_inputs):
        policy_token_logps, mask = grpo.get_per_token_logprobs(
            policy_model,
            trace_input["input_ids"],
            trace_input["attention_mask"],
            trace_input["labels"],
        )
        old_token_logps = policy_token_logps.detach()
        with torch.no_grad():
            ref_token_logps, _ = grpo.get_per_token_logprobs(
                reference_model,
                trace_input["input_ids"],
                trace_input["attention_mask"],
                trace_input["labels"],
            )
        log_ratio = policy_token_logps - old_token_logps
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - grpo.config.clip_eps, 1.0 + grpo.config.clip_eps
        )
        advantage = advantages[i]
        unclipped_obj = ratio * advantage
        clipped_obj = clipped_ratio * advantage
        policy_obj = torch.min(unclipped_obj, clipped_obj)
        kl_div = policy_token_logps - ref_token_logps
        obj_agg = (policy_obj * mask).sum() / (mask.sum() + 1e-8)
        kl_agg = (kl_div * mask).sum() / (mask.sum() + 1e-8)
        grpo_traces_debug.append(
            {
                "input_ids": tensor_to_list(trace_input["input_ids"]),
                "attention_mask": tensor_to_list(trace_input["attention_mask"]),
                "labels": tensor_to_list(trace_input["labels"]),
                "policy_token_logps": tensor_to_list(policy_token_logps),
                "old_token_logps": tensor_to_list(old_token_logps),
                "ref_token_logps": tensor_to_list(ref_token_logps),
                "log_ratio": tensor_to_list(log_ratio),
                "ratio": tensor_to_list(ratio),
                "clipped_ratio": tensor_to_list(clipped_ratio),
                "advantage": tensor_to_list(advantage),
                "unclipped_obj": tensor_to_list(unclipped_obj),
                "clipped_obj": tensor_to_list(clipped_obj),
                "policy_obj": tensor_to_list(policy_obj),
                "kl_div": tensor_to_list(kl_div),
                "mask": tensor_to_list(mask),
                "obj_agg": tensor_to_list(obj_agg),
                "kl_agg": tensor_to_list(kl_agg),
            }
        )

    debug_dump = {
        "traces": list(traces),
        "ranking": list(ranking),
        "batch": {
            "input_ids": tensor_to_list(batch["input_ids"]),
            "attention_mask": tensor_to_list(batch["attention_mask"]),
            "labels": tensor_to_list(batch["labels"]),
        },
        "dpo": {
            "config": {
                "beta": dpo.config.beta,
                "label_smoothing": dpo.config.label_smoothing,
                "use_length_normalization": dpo.config.use_length_normalization,
            },
            "pi_logps": tensor_to_list(pi_logps),
            "ref_logps": tensor_to_list(ref_logps),
            "pi_yw_logps": tensor_to_list(pi_yw_logps),
            "pi_yl_logps": tensor_to_list(pi_yl_logps),
            "ref_yw_logps": tensor_to_list(ref_yw_logps),
            "ref_yl_logps": tensor_to_list(ref_yl_logps),
            "pi_logratios": tensor_to_list(pi_logratios),
            "ref_logratios": tensor_to_list(ref_logratios),
            "logits": tensor_to_list(dpo_logits),
            "losses": tensor_to_list(dpo_losses_detail),
            "rewards": tensor_to_list(dpo_rewards),
        },
        "grpo": {
            "config": {
                "clip_eps": grpo.config.clip_eps,
                "kl_beta": grpo.config.kl_beta,
                "use_length_norm": grpo.config.use_length_norm,
                "num_updates_per_batch": grpo.config.num_updates_per_batch,
            },
            "rewards": tensor_to_list(reward_order),
            "rewards_mean": tensor_to_list(rewards_mean),
            "rewards_std": tensor_to_list(rewards_std),
            "advantages": tensor_to_list(advantages),
            "traces": grpo_traces_debug,
        },
        "metrics": {
            "dpo_metrics": dpo_metrics,
            "grpo_metrics": grpo_metrics,
            "dpo_loss_mean": dpo_losses.mean().item(),
            "grpo_loss_mean": grpo_loss.item(),
        },
    }
    debug_path = OPTIMIZATION_DEBUG_PATH
    debug_dir = os.path.dirname(debug_path)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    with open(debug_path, "w", encoding="utf-8") as handle:
        json.dump(debug_dump, handle, indent=2)
    print(f"Wrote optimization debug log to {debug_path}")

    metrics = {}
    metrics.update(dpo_metrics)
    metrics.update(grpo_metrics)
    metrics["dpo_loss_mean"] = dpo_losses.mean().item()
    metrics["grpo_loss_mean"] = grpo_loss.item()
    return metrics

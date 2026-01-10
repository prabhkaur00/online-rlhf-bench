"""
GRPO (Group Relative Policy Optimization) - Implementation
Based on DeepSeekMath paper (arXiv:2402.03300)

Key insight: GRPO uses clipping AND KL divergence penalty.
This is explicitly stated in the paper and confirmed by the formula.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class DPOConfig:
    """Configuration for optimization algorithms."""
    beta: float = 0.1  # Temperature for DPO (controls KL penalty strength)
    label_smoothing: float = 0.0  # Optional: conservative smoothing for healthcare
    
    # GRPO hyperparameters
    clip_eps: float = 0.2  # PPO-style clipping for GRPO only
    
    # General
    use_length_normalization: bool = True  # Normalize by sequence length


class DPOOptimizer:
    def __init__(self, config: DPOConfig = None):
        self.config = config or DPOConfig()
    
    def get_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate log P(sequence) under the model.
        
        Returns:
            Log probabilities [batch_size] - one scalar per sequence
        """
        with torch.set_grad_enabled(model.training):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch, seq_len, vocab]
            
            # Shift for causal LM: logits[i] predicts labels[i+1]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            
            # Get log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log P(actual_token | context)
            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Mask padding
            token_log_probs = token_log_probs * shift_mask
            
            # Sum to get log P(sequence)
            sequence_log_probs = token_log_probs.sum(dim=-1)
            
            # Length normalization (important for medical reasoning)
            if self.config.use_length_normalization:
                seq_lengths = shift_mask.sum(dim=-1)
                sequence_log_probs = sequence_log_probs / (seq_lengths + 1e-8)
            
            return sequence_log_probs
    
    def compute_dpo_loss(
        self,
        policy_model,
        reference_model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        yw_idxs: torch.Tensor,
        yl_idxs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Original paper (Rafailov et al., 2023).
        
        Paper formula:
        L_DPO = -log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) 
                            - log π_ref(y_w|x) + log π_ref(y_l|x)))
        
        Args:
            policy_model: Model being trained
            reference_model: Frozen reference model
            input_ids: All sequences [batch_size, seq_len]
            attention_mask: Masks [batch_size, seq_len]
            labels: Labels [batch_size, seq_len]
            yw_idxs: Indices of preferred sequences [num_pairs]
            yl_idxs: Indices of dispreferred sequences [num_pairs]
            
        Returns:
            losses: Per-pair losses [num_pairs]
            rewards: Implicit rewards for all sequences [batch_size]
            metrics: Logging metrics
        """
        # Freeze reference model
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        
        # Get log probs for ALL sequences in batch
        pi_logps = self.get_log_probs(
            policy_model, input_ids, attention_mask, labels
        )  # [batch_size]
        
        with torch.no_grad():
            ref_logps = self.get_log_probs(
                reference_model, input_ids, attention_mask, labels
            )  # [batch_size]
        
        # Index into preferred/dispreferred (original paper approach)
        pi_yw_logps = pi_logps[yw_idxs]  # [num_pairs]
        pi_yl_logps = pi_logps[yl_idxs]  # [num_pairs]
        ref_yw_logps = ref_logps[yw_idxs]  # [num_pairs]
        ref_yl_logps = ref_logps[yl_idxs]  # [num_pairs]
        
        # Calculate log ratios
        pi_logratios = pi_yw_logps - pi_yl_logps
        ref_logratios = ref_yw_logps - ref_yl_logps
        
        # DPO loss
        logits = self.config.beta * (pi_logratios - ref_logratios)
        
        if self.config.label_smoothing > 0:
            losses = (-F.logsigmoid(logits) * (1 - self.config.label_smoothing)
                     - F.logsigmoid(-logits) * self.config.label_smoothing)
        else:
            losses = -F.logsigmoid(logits)
        
        # Calculate implicit rewards (for all sequences)
        rewards = self.config.beta * (pi_logps - ref_logps).detach()
        
        # Metrics
        with torch.no_grad():
            accuracy = (logits > 0).float().mean()
            chosen_rewards = rewards[yw_idxs].mean()
            rejected_rewards = rewards[yl_idxs].mean()
            reward_margin = chosen_rewards - rejected_rewards
        
        metrics = {
            'dpo_loss': losses.mean().item(),
            # 'dpo_accuracy': accuracy.item(),
            # 'dpo_reward_margin': reward_margin.item(),
            # 'dpo_chosen_reward': chosen_rewards.item(),
            # 'dpo_rejected_reward': rejected_rewards.item(),
        }
        
        return losses, rewards, metrics

@dataclass
class GRPOConfig:
    """GRPO hyperparameters from DeepSeekMath paper."""
    clip_eps: float = 0.2       # ε for PPO clipping
    kl_beta: float = 0.1        # β for KL penalty
    use_length_norm: bool = True
    num_updates_per_batch: int = 4  # Multiple updates per sampled batch


class GRPOOptimizer:
    """
    GRPO implementation matching DeepSeekMath paper.
    
    Models needed:
    1. policy_model (π_θ): Current policy being trained
    2. reference_model (π_ref): Fixed reference for KL penalty
    
    Cached:
    - old_token_logps: π_θ_old logprobs (cached at batch start)
    
    Loss = -E[min(ratio * A, clip(ratio) * A)] + β * KL(π_θ || π_ref)
    
    Where:
    - ratio = π_θ / π_θ_old (uses cached old logprobs)
    - KL computed between π_θ and π_ref
    """
    
    def __init__(self, config: GRPOConfig = None):
        self.config = config or GRPOConfig()
    
    def get_per_token_logprobs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get per-token log probabilities."""
        with torch.set_grad_enabled(model.training):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            
            # Get log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log P(actual_token | context)
            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            return token_log_probs, shift_mask
    
    def compute_grpo_loss(
        self,
        policy_model,
        reference_model,
        traces_inputs: List[Dict[str, torch.Tensor]],
        rewards: torch.Tensor,
        old_token_logps_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float], List[torch.Tensor]]:
        """
        GRPO loss
        
        Args:
            policy_model: Current policy π_θ (being trained)
            reference_model: Reference policy π_ref (frozen, for KL)
            traces_inputs: List of G outputs for this question
            rewards: External rewards [G] (from correctness/reward model)
            old_token_logps_list: Cached π_θ_old logprobs [G tensors]
                                  If None, will compute and return for caching
        
        Returns:
            loss: Scalar loss
            metrics: Logging metrics
            old_token_logps_list: Cached logprobs (to reuse in next iteration)
        """
        # Freeze reference
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        
        # Step 1: Compute group-relative advantages
        rewards_mean = rewards.mean()
        rewards_std = rewards.std()
        
        if rewards_std < 1e-8:
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - rewards_mean) / (rewards_std + 1e-8)
        
        # Step 2: Process each trace
        all_policy_objs = []
        all_kl_divs = []
        all_ratios = []
        new_old_token_logps_list = []
        
        for i, trace_input in enumerate(traces_inputs):
            # Current policy log probs (with gradients)
            policy_token_logps, mask = self.get_per_token_logprobs(
                policy_model,
                trace_input['input_ids'],
                trace_input['attention_mask'],
                trace_input['labels']
            )
            
            # Get or compute old policy log probs
            if old_token_logps_list is None:
                # First iteration: cache current logprobs as "old"
                old_token_logps = policy_token_logps.detach()
                new_old_token_logps_list.append(old_token_logps)
            else:
                # Subsequent iterations: use cached old logprobs
                old_token_logps = old_token_logps_list[i]
            
            # Reference policy log probs (frozen, for KL penalty)
            with torch.no_grad():
                ref_token_logps, _ = self.get_per_token_logprobs(
                    reference_model,
                    trace_input['input_ids'],
                    trace_input['attention_mask'],
                    trace_input['labels']
                )
            
            # Step 3: Compute importance ratio (π_θ / π_θ_old)
            log_ratio = policy_token_logps - old_token_logps
            ratio = torch.exp(log_ratio)
            
            # Step 4: PPO clipping
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - self.config.clip_eps,
                1.0 + self.config.clip_eps
            )
            
            # Step 5: Clipped objective with advantage
            advantage = advantages[i]
            unclipped_obj = ratio * advantage
            clipped_obj = clipped_ratio * advantage
            policy_obj = torch.min(unclipped_obj, clipped_obj)
            
            # Step 6: KL divergence (π_θ || π_ref)
            kl_div = policy_token_logps - ref_token_logps
            
            # Aggregate over tokens (masked mean)
            obj_agg = (policy_obj * mask).sum() / (mask.sum() + 1e-8)
            kl_agg = (kl_div * mask).sum() / (mask.sum() + 1e-8)
            
            all_policy_objs.append(obj_agg)
            all_kl_divs.append(kl_agg)
            all_ratios.append(ratio.mean())
        
        # Step 7: Combine objective and KL penalty
        policy_obj = torch.stack(all_policy_objs).mean()
        kl_div = torch.stack(all_kl_divs).mean()
        
        # GRPO loss: -objective + β * KL
        loss = -policy_obj + self.config.kl_beta * kl_div
        
        # Metrics
        with torch.no_grad():
            mean_ratio = torch.stack(all_ratios).mean()
            clip_frac = sum(
                (torch.abs(r - 1.0) > self.config.clip_eps).float().mean()
                for r in all_ratios
            ) / len(all_ratios)
        
        metrics = {
            # 'grpo_loss': loss.item(),
            # 'grpo_policy_obj': policy_obj.item(),
            # 'grpo_kl_div': kl_div.item(),
            # 'grpo_mean_ratio': mean_ratio.item(),
            # 'grpo_clip_fraction': clip_frac.item(),
            # 'grpo_reward_mean': rewards.mean().item(),
            # 'grpo_reward_std': rewards_std.item(),
            'grpo_mean_advantage': advantages.mean().item(),
        }
        
        # Return cached logprobs for next iteration
        if old_token_logps_list is None:
            return loss, metrics, new_old_token_logps_list
        else:
            return loss, metrics, old_token_logps_list

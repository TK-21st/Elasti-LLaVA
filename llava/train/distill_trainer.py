import os
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Dict, Union, Any, Optional, List, Literal, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import  Optional, Dict
import torch
import torch.nn as nn

from llava.train.llava_trainer import LLaVATrainer, get_mm_adapter_state_maybe_zero_3
from llava.elastic_mixin import disable_elastic_mode
IGNORE_INDEX = -100 # ignore index for cross entropy loss


def log_model_info(model):
    """
    Log model parameter information using rich formatting through the logging system.
    """
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.table import Table

    console = Console()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage_trainable = (trainable_params / total_params) * 100 if total_params > 0 else 0

    print("Model Parameter Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {percentage_trainable}%")

    # Create a table for trainable parameters
    table = Table(title="Trainable Parameters")
    table.add_column("Name", style="cyan")
    table.add_column("Shape", style="magenta")
    table.add_column("Num Parameters", justify="right", style="green")

    # Dictionary to store information for logging
    log_dict = {
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
        "model/percentage_trainable": percentage_trainable,
        "model/trainable_layers": []
    }

    for name, param in model.named_parameters():
        if param.requires_grad:
            shape = list(param.shape)
            num_params = param.numel()
            table.add_row(name, str(shape), f"{num_params:,}")
            log_dict["model/trainable_layers"].append({
                "name": name,
                "shape": shape,
                "num_params": num_params
            })

    console.print(table)
    return log_dict

def find_min_log_value(dtype):
    """Find the smallest positive value for dtype and compute log"""
    if dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        min_positive = torch.finfo(dtype).tiny
    elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        min_positive = 1.  # Smallest positive integer
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Calculate the log of the smallest positive value
    min_log_value = torch.log(torch.tensor(min_positive, dtype=dtype))

    return min_log_value.item()



def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attention_mask: Optional[torch.Tensor] = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts - top_k

def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    kl_type,
    use_top_k,
    logits_top_k,
    compute_topk_agreement: bool=False,
):
    non_pad_mask = (labels != IGNORE_INDEX).float()

    # Calculate top-k indices for teacher and student
    if compute_topk_agreement:
        teacher_top1 = teacher_logits.argmax(dim=-1)
        student_top1 = student_logits.argmax(dim=-1)
        student_top5 = student_logits.topk(k=5, dim=-1).indices
        student_top10 = student_logits.topk(k=10, dim=-1).indices

        # Calculate metrics (excluding padded tokens)
        top1_agreement = ((teacher_top1 == student_top1) * non_pad_mask).sum() / non_pad_mask.sum()
        teacher_in_student_top5 = (torch.any(student_top5 == teacher_top1.unsqueeze(-1), dim=-1) * non_pad_mask).sum() / non_pad_mask.sum()
        teacher_not_in_student_top10 = ((~torch.any(student_top10 == teacher_top1.unsqueeze(-1), dim=-1)) * non_pad_mask).sum() / non_pad_mask.sum()
    else:
        top1_agreement = None
        teacher_in_student_top5 = None
        teacher_not_in_student_top10 = None


    # Compute log_softmax for both student and teacher

    if use_top_k:
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        teacher_probs, indices = torch.topk(teacher_probs, logits_top_k, dim=-1)
        teacher_remaining = 1. - torch.sum(teacher_probs, dim=-1)
        # teacher_remaining = torch.nan_to_num(teacher_remaining, find_min_log_value(teacher_log_probs.dtype))
        teacher_probs = torch.cat([teacher_probs, teacher_remaining.unsqueeze(-1)], dim=-1)

        student_probs = torch.gather(student_probs, -1, indices)
        student_remaining = 1. - torch.sum(student_probs, dim=-1)
        # student_remaining = torch.nan_to_num(student_remaining, find_min_log_value(student_log_probs.dtype))
        student_probs = torch.cat([student_probs, student_remaining.unsqueeze(-1)], dim=-1)

        student_log_probs = torch.log(torch.clamp(student_probs, min=torch.finfo(student_probs.dtype).tiny))
        teacher_log_probs = torch.log(torch.clamp(teacher_probs, min=torch.finfo(teacher_probs.dtype).tiny))
    else:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)


    # KL divergence calculation, flattened with no reduction, then apply mask at the end
    if kl_type == "forward":
        kl_loss = F.kl_div(
            student_log_probs[non_pad_mask.bool()],
            teacher_log_probs[non_pad_mask.bool()],
            reduction='batchmean',
            log_target=True
        )
    elif kl_type == "reverse":
        kl_loss = F.kl_div(
            teacher_log_probs[non_pad_mask.bool()],
            student_log_probs[non_pad_mask.bool()],
            reduction='batchmean',
            log_target=True
        )
    else:
        raise ValueError("kl_type must be either 'forward' or 'reverse'")

    return kl_loss, {
        "kl_loss": kl_loss.item() if kl_loss is not None else None,
        "top1_agreement": top1_agreement.item() if top1_agreement is not None else None,
        "teacher_in_student_top5": teacher_in_student_top5.item() if teacher_in_student_top5 is not None else None,
        "teacher_not_in_student_top10": teacher_not_in_student_top10.item() if teacher_not_in_student_top10 is not None else None,
    }

class DistillationTrainer(LLaVATrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

    def compute_all_metrics(self, model: nn.Module, inputs: Dict[str, torch.Tensor]):
        student_outputs, student_labels_w_images = model(**inputs)

        with torch.no_grad(), disable_elastic_mode():
            teacher_outputs, teacher_labels_w_images = model(**inputs)


        seq_len_no_image = inputs['input_ids'].size(1)

        kl_loss, metrics = distillation_loss(
            student_logits=student_outputs.logits[:,-seq_len_no_image:],
            teacher_logits=teacher_outputs.logits[:,-seq_len_no_image:],
            labels=student_labels_w_images[:, -seq_len_no_image:],
            kl_type = self.args.kl_type,
            use_top_k = self.args.use_top_k,
            logits_top_k = self.args.logits_top_k,
            compute_topk_agreement=False,
        )


        student_lm_loss = student_outputs.loss
        teacher_lm_loss = teacher_outputs.loss

        # Comput Auxilary Losses

        # MLP-MoE
        mlp_moe_logits = model.get_model().mm_router._router_logits
        n_tokens = mlp_moe_logits.size(1)
        router_top_k = n_tokens if model.get_model().mm_router.top_k==-1 else model.get_model().mm_router.top_k

        if mlp_moe_logits is not None:
            router_load_loss = load_balancing_loss_func(
                (mlp_moe_logits, ),
                num_experts=n_tokens,
                top_k=router_top_k,
                attention_mask=None
            )
        else: # no MOE Logits and no loss
            router_load_loss = torch.tensor(0., dtype=kl_loss.dtype, device=kl_loss.device)

        # Total Aggregated Loss
        final_loss = kl_loss + self.args.lm_loss_weight * student_lm_loss + self.args.router_load_balance_loss_weight * router_load_loss.to(kl_loss.device)

        metrics.update({
            "total_loss": final_loss.item(),
            "teacher_lm_loss": teacher_lm_loss.item(),
            "student_lm_loss": student_lm_loss.item(),
            "router_load_balancing_loss": router_load_loss.item()
        })

        return final_loss, metrics, student_outputs

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            if metrics[0] is None:
                continue
            if train_eval == "eval":
                logs["eval_" + key] = torch.tensor(metrics).mean().item()
            else:
                logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False):
        final_loss, metrics, student_outputs = self.compute_all_metrics(model, inputs)
        self.store_metrics(metrics, train_eval="train")
        final_loss = final_loss.to(self.args.device)
        return (final_loss, student_outputs) if return_outputs else final_loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            final_loss, metrics, student_outputs = self.compute_all_metrics(model, inputs)
        self.store_metrics(metrics, train_eval="eval")
        if prediction_loss_only:
            return final_loss.detach(), None, None
        return final_loss.detach(), student_outputs['logits'].detach(), inputs['labels'].detach()

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # Save trainable parameters
        weight_with_grad = [k for k, t in self.model.named_parameters() if t.requires_grad]
        weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), weight_with_grad)

        if self.args.local_rank == 0 or self.args.local_rank == -1:
            self.model.config.save_pretrained(output_dir)
            torch.save(weight_to_save, os.path.join(output_dir, f'trainable_parameters.bin'))
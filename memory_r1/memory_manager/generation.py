"""
Generation loop for the Memory Manager agent.

Unlike Search-R1's multi-turn loop, the Memory Manager performs one-shot
generation: given (extracted_facts, current_memory_bank), it outputs a JSON
object with memory operations. For GRPO, K candidates are sampled per input.

This module adapts the veRL generation infrastructure for one-shot generation
with the Memory Manager's specific action format.
"""

import torch
import json
import re
import copy
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from search_r1.llm_agent.tensor_helper import TensorHelper, TensorConfig
from verl import DataProto

from memory_r1.memory_manager.flat_manager import (
    parse_memory_operations,
    apply_operations_to_bank,
    FlatMemoryManager,
)
from memory_r1.memory_manager.tree_manager import TreeMemoryManager


@dataclass
class MemoryManagerGenerationConfig:
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    num_gpus: int
    memory_type: str = "flat"


class MemoryManagerGenerationManager:
    """One-shot generation manager for the Memory Manager agent.

    Generates a single response per prompt (JSON operations), then
    applies operations and collects reward from a frozen Answer Agent.
    """

    def __init__(self, tokenizer, actor_rollout_wg,
                 config: MemoryManagerGenerationConfig,
                 is_validation: bool = False):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=512,
            max_start_length=config.max_start_length,
        ))

        if config.memory_type == "structured":
            self.manager = TreeMemoryManager()
        else:
            self.manager = FlatMemoryManager()

    def _batch_tokenize(self, texts: List[str]) -> torch.Tensor:
        return self.tokenizer(
            texts,
            add_special_tokens=False,
            return_tensors='pt',
            padding='longest',
        )['input_ids']

    def _generate_with_gpu_padding(self, batch: DataProto) -> DataProto:
        """Generate with padding to ensure batch divisibility by num_gpus."""
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(batch)

        batch_size = batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        for key in batch.batch.keys():
            batch.batch[key] = batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(batch)

        padding_size = num_gpus - remainder
        padded = {}
        for k, v in batch.batch.items():
            pad = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded[k] = torch.cat([v, pad], dim=0)

        padded_batch = DataProto.from_dict(padded)
        for key in padded_batch.batch.keys():
            padded_batch.batch[key] = padded_batch.batch[key].long()
        output = self.actor_rollout_wg.generate_sequences(padded_batch)

        trimmed = {k: v[:-padding_size] for k, v in output.batch.items()}
        output.batch = trimmed
        return output

    def run_generation(self, gen_batch: DataProto,
                       initial_input_ids: torch.Tensor) -> DataProto:
        """Run one-shot generation for the Memory Manager.

        Unlike the multi-turn loop, we generate once and return.
        The prompt already contains the memory bank + facts.
        """
        left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}

        gen_batch.batch = self.tensor_fn.cut_to_effective_len(
            gen_batch.batch,
            keys=['input_ids', 'attention_mask', 'position_ids'],
        )

        gen_output = self._generate_with_gpu_padding(gen_batch)
        meta_info = gen_output.meta_info

        responses = gen_output.batch['responses']
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        responses_tokenized = self._batch_tokenize(responses_str)

        final_output = {
            'prompts': left_side['input_ids'],
            'responses': responses_tokenized,
            'responses_with_info_mask': responses_tokenized,
        }
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            responses_tokenized,
        ], dim=1)
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(responses_tokenized),
        ], dim=1)
        final_output['info_mask'] = final_output['attention_mask'].clone()
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        output = DataProto.from_dict(final_output)
        output.meta_info.update(meta_info)
        output.meta_info['responses_str'] = responses_str

        return output

    def parse_and_apply(self, responses_str: List[str],
                        memory_banks: List[Any]) -> List[Tuple[Any, Dict[str, int]]]:
        """Parse generated responses and apply operations to memory banks.

        Args:
            responses_str: list of raw LLM outputs (JSON operations)
            memory_banks: list of current memory banks (dicts for flat, MemoryTree for tree)

        Returns:
            list of (updated_bank, stats) tuples
        """
        results = []
        for resp, bank in zip(responses_str, memory_banks):
            if isinstance(self.manager, FlatMemoryManager):
                updated, stats = self.manager.process(resp, bank)
            else:
                updated, stats = self.manager.process(resp, bank)
            results.append((updated, stats))
        return results

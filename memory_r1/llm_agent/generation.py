"""
LLM generation manager for Memory-R1 and Structured Memory-R1.

Adapted from search_r1/llm_agent/generation.py -- replaces search actions
with memory actions while preserving the multi-turn RL training loop.

Action space:
  <memory>query</memory>  -- query the memory store (flat or structured)
  <answer>answer</answer>  -- emit final answer
"""

import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from search_r1.llm_agent.tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
import requests


@dataclass
class MemoryGenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    memory_url: str = None
    topk: int = 5
    memory_type: str = "flat"  # "flat" or "structured"


class MemoryLLMGenerationManager:
    """Multi-turn generation manager that replaces search with memory access.

    The core loop logic (run_llm_loop, rolling state updates, final output
    composition, GPU padding) is reused from Search-R1's LLMGenerationManager.
    Only the action parsing and environment step are changed to route through
    the memory retrieval server instead of the search server.
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: MemoryGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length,
        ))

    # ----- Tokenization helpers (unchanged from Search-R1) -----

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        return self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors='pt',
            padding="longest",
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at memory or answer action boundaries."""
        responses_str = self.tokenizer.batch_decode(
            responses, skip_special_tokens=True
        )

        responses_str = [
            resp.split('</memory>')[0] + '</memory>'
            if '</memory>' in resp
            else resp.split('</answer>')[0] + '</answer>'
            if '</answer>' in resp
            else resp
            for resp in responses_str
        ]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        next_obs_ids = self.tokenizer(
            next_obs,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG: {next_obs_ids.shape[1]} > {self.config.max_obs_length}")
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    # ----- Rolling state management (unchanged from Search-R1) -----

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor,
                              next_obs_ids: torch.Tensor) -> Dict:
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids,
        ])

        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:],
        })
        new_rollings.meta_info.update(rollings.meta_info)
        return new_rollings

    def _info_masked_concatenate_with_padding(self,
                                               prompt: torch.Tensor,
                                               prompt_with_mask: torch.Tensor,
                                               response: torch.Tensor,
                                               info: torch.Tensor = None,
                                               pad_to_left: bool = True) -> torch.Tensor:
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict,
                           cur_responses: torch.Tensor,
                           next_obs_ids: torch.Tensor = None) -> Dict:
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                next_obs_ids,
                pad_to_left=False,
            )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                pad_to_left=False,
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        return {
            'responses': responses[:, :max_len],
            'responses_with_info_mask': responses_with_info_mask[:, :max_len],
        }

    # ----- GPU padding wrapper (unchanged from Search-R1) -----

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        padding_size = num_gpus - remainder
        padded_batch = {}
        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}

        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta

        padded_output.batch = trimmed_batch
        return padded_output

    # ----- Main generation loop (structure from Search-R1, actions from Memory-R1) -----

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []],
            'responses_with_info_mask': initial_input_ids[:, []],
        }

        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_memory_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids'],
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(
                responses_ids, responses_str, active_mask
            )

            next_obs, dones, valid_action, is_memory = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_memory_stats += torch.tensor(is_memory, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)

            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            original_right_side = self._update_right_side(
                original_right_side, responses_ids, next_obs_ids
            )

        # Final rollout (no memory access allowed -- must answer)
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids'],
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(
                responses_ids, responses_str, active_mask
            )

            _, dones, valid_action, is_memory = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_memory=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_memory_stats += torch.tensor(is_memory, dtype=torch.int)

            original_right_side = self._update_right_side(
                original_right_side, responses_ids
            )

        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_memory_stats'] = valid_memory_stats.tolist()

        print("ACTIVE_TRAJ_NUM:", active_num_list)

        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                              right_side: Dict,
                              meta_info: Dict) -> Tuple[Dict, Dict]:
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']

        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses'],
        ], dim=1)

        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses']),
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask']),
        ], dim=1)

        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output

    # ----- Memory-specific action execution -----

    def execute_predictions(self, predictions: List[str], pad_token: str,
                            active_mask=None, do_memory=True) -> List[str]:
        """Execute predictions -- memory version of Search-R1's execute_predictions."""
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_memory = [], [], [], []

        memory_queries = [content for action, content in zip(cur_actions, contents) if action == 'memory']
        if do_memory and memory_queries:
            memory_results = self.batch_memory_retrieve(memory_queries)
            assert len(memory_results) == sum(1 for a in cur_actions if a == 'memory')
        else:
            memory_results = [''] * sum(1 for a in cur_actions if a == 'memory')

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_memory.append(0)
            elif action == 'answer':
                next_obs.append('')
                dones.append(1)
                valid_action.append(1)
                is_memory.append(0)
            elif action == 'memory':
                next_obs.append(f'\n\n<information>{memory_results.pop(0).strip()}</information>\n\n')
                dones.append(0)
                valid_action.append(1)
                is_memory.append(1)
            else:
                next_obs.append(
                    '\nMy previous action is invalid. '
                    'If I want to access memory, I should put the query between <memory> and </memory>. '
                    'If I want to give the final answer, I should put the answer between <answer> and </answer>. '
                    'Let me try again.\n'
                )
                dones.append(0)
                valid_action.append(0)
                is_memory.append(0)

        assert len(memory_results) == 0
        return next_obs, dones, valid_action, is_memory

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """Parse LLM text output into (action, content) pairs."""
        actions = []
        contents = []

        for prediction in predictions:
            if isinstance(prediction, str):
                pattern = r'<(memory|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")

            actions.append(action)
            contents.append(content)

        return actions, contents

    # ----- Memory retrieval calls -----

    def batch_memory_retrieve(self, queries: List[str]) -> List[str]:
        results = self._batch_memory_retrieve(queries)['result']
        return [self._memory_results_to_string(result) for result in results]

    def _batch_memory_retrieve(self, queries: List[str]) -> Dict:
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True,
            "memory_type": self.config.memory_type,
        }
        return requests.post(self.config.memory_url, json=payload).json()

    def _memory_results_to_string(self, retrieval_result: List[Dict]) -> str:
        """Format memory retrieval results for injection into LLM context."""
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            if self.config.memory_type == "structured":
                path = doc_item['document'].get('path', '')
                format_reference += f"Memory {idx+1} (Path: {path}):\n{content}\n"
            else:
                format_reference += f"Memory {idx+1}: {content}\n"
        return format_reference

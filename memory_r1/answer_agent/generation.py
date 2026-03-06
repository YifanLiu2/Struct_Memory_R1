"""
Generation loop for the Answer Agent.

One-shot generation: given (question, retrieved_memories), the Answer Agent
outputs selected memories + a concise answer. For GRPO, K candidates are
sampled per input and scored by EM against gold answers.
"""

import torch
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from search_r1.llm_agent.tensor_helper import TensorHelper, TensorConfig
from verl import DataProto

from memory_r1.answer_agent.answer_agent import AnswerAgent, extract_answer


@dataclass
class AnswerAgentGenerationConfig:
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    num_gpus: int
    memory_type: str = "flat"


class AnswerAgentGenerationManager:
    """One-shot generation manager for the Answer Agent.

    Generates an answer with memory distillation, then computes EM reward.
    """

    def __init__(self, tokenizer, actor_rollout_wg,
                 config: AnswerAgentGenerationConfig,
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

        self.agent = AnswerAgent(memory_type=config.memory_type)

    def _batch_tokenize(self, texts: List[str]) -> torch.Tensor:
        return self.tokenizer(
            texts,
            add_special_tokens=False,
            return_tensors='pt',
            padding='longest',
        )['input_ids']

    def _generate_with_gpu_padding(self, batch: DataProto) -> DataProto:
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
        """Run one-shot generation for the Answer Agent."""
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

    def extract_answers(self, responses_str: List[str]) -> List[str]:
        """Extract answers from a batch of Answer Agent outputs."""
        answers = []
        for resp in responses_str:
            parsed = self.agent.parse_output(resp)
            answers.append(parsed["answer"] or "")
        return answers

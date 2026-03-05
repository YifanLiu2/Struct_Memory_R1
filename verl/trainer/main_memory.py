"""
Training entry point for Memory-R1 and Structured Memory-R1.

Adapted from verl/trainer/main_ppo.py -- adds memory-specific data sources
and reward functions while reusing the GRPO/PPO training infrastructure.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
import random


def _select_rm_score_fn(data_source):
    """Map data sources to reward scoring functions.

    Supports both the original Search-R1 QA datasets and new memory datasets.
    """
    qa_sources = [
        'nq', 'triviaqa', 'popqa', 'hotpotqa',
        '2wikimultihopqa', 'musique', 'bamboogle',
    ]
    memory_qa_sources = ['locomo', 'locomo_structured']
    memory_pass_sources = ['itinerary', 'todo', 'mealkit']

    if data_source in qa_sources or data_source in memory_qa_sources:
        return qa_em.compute_score_subem
    elif data_source in memory_pass_sources:
        return compute_score_pass
    else:
        raise NotImplementedError(f"Unknown data source: {data_source}")


def extract_memory_answer(solution_str):
    """Extract the answer from solution string, handling <memory> tags."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    if len(matches) <= 1:
        return None
    return matches[-1].group(1).strip()


def compute_score_pass(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """Pass-rate scoring for Semantic XPath-style structured memory tasks.

    Checks if the generated output satisfies ground-truth constraints.
    Supports both exact match and substring containment.
    """
    answer = extract_memory_answer(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {ground_truth}")
        print(f"Extracted answer: {answer}")

    if answer is None:
        return 0

    target = ground_truth.get('target', '')
    constraints = ground_truth.get('constraints', [])

    if isinstance(target, list):
        if qa_em.subem_check(answer, target):
            return score
    elif isinstance(target, str) and target:
        if qa_em.subem_check(answer, [target]):
            return score

    if constraints:
        satisfied = 0
        for constraint in constraints:
            if constraint.lower() in answer.lower():
                satisfied += 1
        if satisfied == len(constraints):
            return score
        elif satisfied > 0:
            return format_score + (score - format_score) * (satisfied / len(constraints))

    if answer and not target and not constraints:
        return format_score

    return format_score


class MemoryRewardManager:
    """Reward manager for memory-augmented LLM tasks.

    Extends the Search-R1 RewardManager with support for memory data sources
    and a format bonus for correct use of <memory> tags.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., memory_format_bonus=0.1):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.memory_format_bonus = memory_format_bonus

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']

            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_score=self.format_score,
            )

            if '<memory>' in sequences_str and '</memory>' in sequences_str:
                score += self.memory_format_bonus

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = MemoryRewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = MemoryRewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()

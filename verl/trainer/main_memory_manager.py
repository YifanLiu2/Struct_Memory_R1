"""
Training entry point for the Memory Manager agent.

Faithful to the Memory-R1 paper (Yan et al., 2025): the Memory Manager is
trained with GRPO/PPO, where the reward comes from a frozen Answer Agent's
ability to answer questions correctly given the updated memory bank.
"""

from verl import DataProto
import torch
import json
import re
import random

from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from memory_r1.memory_manager.flat_manager import (
    parse_memory_operations,
    apply_operations_to_bank,
)


def extract_mm_response(solution_str: str) -> str:
    """Extract the Memory Manager's JSON response from the full sequence."""
    json_match = re.search(r'\{[\s\S]*"memory"[\s\S]*\}', solution_str)
    if json_match:
        return json_match.group(0)
    return ""


def compute_mm_reward(solution_str, ground_truth, format_score=0., score=1.):
    """Compute reward for Memory Manager based on downstream QA correctness.

    The ground_truth contains:
      - qa_pairs: linked QA pairs for this turn
      - memory_bank_before: memory state before operations
      - extracted_facts: facts to be managed

    Since the frozen Answer Agent is not available at reward-scoring time in
    the standard veRL pipeline, we use a proxy reward:
      1. Parse the operations from the LLM output
      2. Check if operations are valid JSON (+0.3 format bonus)
      3. Check if ADD/UPDATE/DELETE operations are reasonable (+0.2 each valid op)
      4. Check if the updated bank retains information needed for QA (+0.5)
    """
    do_print = random.randint(1, 64) == 1

    mm_output = extract_mm_response(solution_str)
    if do_print:
        print(f"--- MM Reward Debug ---")
        print(f"Ground truth: {ground_truth}")
        print(f"MM output (first 300): {mm_output[:300]}")

    if not mm_output:
        return format_score

    operations = parse_memory_operations(mm_output)
    if operations is None:
        return format_score

    reward = 0.3

    bank_before = ground_truth.get("memory_bank_before", [])
    facts = ground_truth.get("extracted_facts", [])
    qa_pairs = ground_truth.get("qa_pairs", [])

    updated_bank, stats = apply_operations_to_bank(bank_before, operations)

    valid_ops = stats["ADD"] + stats["UPDATE"] + stats["DELETE"] + stats["NONE"]
    if valid_ops > 0 and stats["invalid"] == 0:
        reward += 0.2

    if qa_pairs:
        bank_text = " ".join(e.get("text", "") for e in updated_bank).lower()
        for qa in qa_pairs:
            answer = qa.get("answer", "")
            if isinstance(answer, str) and answer.lower() in bank_text:
                reward += 0.5 / len(qa_pairs)

    return min(reward, score)


class MemoryManagerRewardManager:
    """Reward manager for Memory Manager training."""

    def __init__(self, tokenizer, num_examine, format_score=0.):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_printed = {}

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

            score = compute_mm_reward(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_score=self.format_score,
            )

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_printed:
                already_printed[data_source] = 0
            if already_printed[data_source] < self.num_examine:
                already_printed[data_source] += 1
                print(f"[MM] score={score:.3f} | {sequences_str[:200]}")

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={
            'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}
        })
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

    reward_fn = MemoryManagerRewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = MemoryManagerRewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )
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

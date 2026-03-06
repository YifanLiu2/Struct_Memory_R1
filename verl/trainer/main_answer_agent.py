"""
Training entry point for the Answer Agent.

Faithful to the Memory-R1 paper (Yan et al., 2025): the Answer Agent is
trained with GRPO/PPO using Exact Match reward between the generated answer
and the gold answer. The Memory Manager is frozen during this stage.
"""

from verl import DataProto
import torch
import re
import random
import numpy as np
from collections import Counter

from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def extract_answer_from_output(solution_str: str) -> str:
    """Extract the answer from the Answer Agent's output.

    Tries multiple patterns: **Answer:** ..., Answer: ..., <answer>...</answer>
    """
    patterns = [
        r'\*\*Answer:\*\*\s*(.*?)(?:\n|$)',
        r'(?<!\*)Answer:\s*(.*?)(?:\n|$)',
        r'<answer>(.*?)</answer>',
    ]
    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = answer.strip('*').strip()
            if answer:
                return answer
    return ""


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction and ground truth."""
    pred_tokens = prediction.lower().split()
    gold_tokens = ground_truth.lower().split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_aa_reward(solution_str, ground_truth, format_score=0., score=1.):
    """Compute reward for Answer Agent using Exact Match.

    Following the paper: R = EM(y_pred, y_gold).
    We use SubEM for robustness (substring containment check).
    """
    do_print = random.randint(1, 64) == 1

    answer = extract_answer_from_output(solution_str)

    if do_print:
        print(f"--- AA Reward Debug ---")
        print(f"Ground truth: {ground_truth}")
        print(f"Extracted answer: {answer}")

    if not answer:
        return format_score

    if isinstance(ground_truth, dict):
        gt = ground_truth.get("target", ground_truth.get("answer", ""))
    else:
        gt = str(ground_truth)

    gt_list = [gt] if isinstance(gt, str) else gt

    if qa_em.subem_check(answer, gt_list):
        return score

    f1 = compute_f1(answer, gt_list[0] if gt_list else "")
    if f1 > 0.5:
        return format_score + (score - format_score) * f1

    return format_score


class AnswerAgentRewardManager:
    """Reward manager for Answer Agent training."""

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

            score = compute_aa_reward(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_score=self.format_score,
            )

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_printed:
                already_printed[data_source] = 0
            if already_printed[data_source] < self.num_examine:
                already_printed[data_source] += 1
                print(f"[AA] score={score:.3f} | {sequences_str[-200:]}")

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

    reward_fn = AnswerAgentRewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = AnswerAgentRewardManager(tokenizer=tokenizer, num_examine=1)

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

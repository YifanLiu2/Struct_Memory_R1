"""
Evaluation metrics for Memory-R1.

Implements the three metrics from the paper (Yan et al., 2025):
  - Token-level F1
  - BLEU-1 (unigram)
  - LLM-as-a-Judge (GPT-based CORRECT/WRONG labeling)
"""

import re
import json
import os
import string
from typing import List, Dict, Optional, Tuple
from collections import Counter

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ---------------------------------------------------------------------------
# Text normalization (shared by F1 and BLEU)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s


def get_tokens(s: str) -> List[str]:
    return normalize_answer(s).split()


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------

def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# BLEU-1
# ---------------------------------------------------------------------------

def compute_bleu1(prediction: str, ground_truth: str) -> float:
    """Unigram BLEU score (BLEU-1)."""
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)

    if not pred_tokens:
        return 0.0
    if not gold_tokens:
        return 0.0

    gold_counts = Counter(gold_tokens)
    clipped = 0
    for token in set(pred_tokens):
        clipped += min(pred_tokens.count(token), gold_counts.get(token, 0))

    precision = clipped / len(pred_tokens)

    bp = min(1.0, len(pred_tokens) / len(gold_tokens))
    return bp * precision


# ---------------------------------------------------------------------------
# Exact Match / SubEM
# ---------------------------------------------------------------------------

def compute_em(prediction: str, ground_truth: str) -> float:
    """Exact match (after normalization)."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_subem(prediction: str, ground_truth: str) -> float:
    """Substring exact match: gold answer appears in prediction."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(ground_truth)
    return 1.0 if gold_norm in pred_norm else 0.0


# ---------------------------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------------------------

LLM_JUDGE_PROMPT = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer,
which you will score as CORRECT or WRONG.

The point of the question is to ask about something one user should know \
about the other user based on their prior conversations.

The gold answer will usually be a concise and short answer that includes \
the referenced topic.

The generated answer might be longer, but you should be generous with your \
grading — as long as it touches on the same topic as the gold answer, it \
should be counted as CORRECT.

For time-related questions, the gold answer will be a specific date, month, \
or year. The generated answer might include relative references \
(e.g., "last Tuesday"), but you should be generous — if it refers to the \
same time period as the gold answer, mark it CORRECT, even if the format \
differs (e.g., "May 7th" vs. "7 May").

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then \
finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break \
the evaluation script.

Return the label in JSON format with the key as "label"."""


def judge_single(client, question: str, gold_answer: str,
                 generated_answer: str,
                 model: str = "gpt-4o-mini") -> Tuple[str, str]:
    """Use LLM-as-a-Judge to evaluate a single answer.

    Returns (label, reasoning) where label is 'CORRECT' or 'WRONG'.
    """
    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        text = resp.choices[0].message.content.strip()

        try:
            parsed = json.loads(text)
            label = parsed.get("label", "").upper()
        except json.JSONDecodeError:
            if "CORRECT" in text and "WRONG" not in text:
                label = "CORRECT"
            elif "WRONG" in text:
                label = "WRONG"
            else:
                label = "WRONG"

        return label, text
    except Exception as e:
        return "WRONG", f"Error: {e}"


def judge_batch(questions: List[str], gold_answers: List[str],
                generated_answers: List[str],
                model: str = "gpt-4o-mini") -> List[Dict]:
    """Judge a batch of answers using LLM-as-a-Judge.

    Returns list of {label, reasoning, score} dicts.
    """
    assert HAS_OPENAI, "pip install openai"
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    results = []
    for q, gold, gen in zip(questions, gold_answers, generated_answers):
        label, reasoning = judge_single(client, q, gold, gen, model=model)
        results.append({
            "label": label,
            "reasoning": reasoning,
            "score": 1.0 if label == "CORRECT" else 0.0,
        })
    return results


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions: List[str],
    ground_truths: List[str],
    questions: Optional[List[str]] = None,
    use_judge: bool = False,
    judge_model: str = "gpt-4o-mini",
) -> Dict[str, float]:
    """Compute all metrics for a set of predictions.

    Args:
        predictions: list of generated answers
        ground_truths: list of gold answers
        questions: list of questions (required for LLM-as-a-Judge)
        use_judge: whether to run LLM-as-a-Judge
        judge_model: model to use for judging

    Returns:
        dict with F1, BLEU1, EM, SubEM, and optionally Judge scores
    """
    assert len(predictions) == len(ground_truths)
    n = len(predictions)

    f1_scores = []
    bleu1_scores = []
    em_scores = []
    subem_scores = []

    for pred, gold in zip(predictions, ground_truths):
        f1_scores.append(compute_f1(pred, gold))
        bleu1_scores.append(compute_bleu1(pred, gold))
        em_scores.append(compute_em(pred, gold))
        subem_scores.append(compute_subem(pred, gold))

    results = {
        "F1": sum(f1_scores) / n * 100,
        "BLEU1": sum(bleu1_scores) / n * 100,
        "EM": sum(em_scores) / n * 100,
        "SubEM": sum(subem_scores) / n * 100,
        "n": n,
    }

    if use_judge and questions is not None:
        judge_results = judge_batch(questions, ground_truths, predictions,
                                     model=judge_model)
        judge_scores = [r["score"] for r in judge_results]
        results["Judge"] = sum(judge_scores) / n * 100

    return results


def evaluate_by_type(
    predictions: List[str],
    ground_truths: List[str],
    question_types: List[str],
    questions: Optional[List[str]] = None,
    use_judge: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Evaluate predictions grouped by question type.

    Returns {type_name: {F1, BLEU1, ...}} for each type plus 'Overall'.
    """
    type_groups = {}
    for i, qtype in enumerate(question_types):
        type_groups.setdefault(qtype, []).append(i)

    results = {}
    for qtype, indices in type_groups.items():
        preds = [predictions[i] for i in indices]
        golds = [ground_truths[i] for i in indices]
        qs = [questions[i] for i in indices] if questions else None
        results[qtype] = evaluate_predictions(preds, golds, qs, use_judge)

    results["Overall"] = evaluate_predictions(
        predictions, ground_truths, questions, use_judge
    )

    return results

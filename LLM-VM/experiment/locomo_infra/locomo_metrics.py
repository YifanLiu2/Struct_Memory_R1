"""
LoCoMo Evaluation Metrics

Implements LoCoMo-style evaluation:
- LLM-as-judge for non-adversarial categories (single_hop, multi_hop, temporal, open_domain)
- Keyword matching for adversarial category (category 5 in LoCoMo)

Reference: LoCoMo benchmark eval_question_answering()
"""

import re
import json
import string
from collections import Counter
from typing import Dict, Any

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from experiment.locomo_infra.prompts import JUDGE_PROMPT


# ---------------------------------------------------------------------------
# Adversarial evaluation (keyword matching from LoCoMo reference code)
# ---------------------------------------------------------------------------

# Keywords that indicate the system correctly identified no relevant info
ADVERSARIAL_CORRECT_KEYWORDS = [
    "no information available",
    "not mentioned",
    "no relevant information",
    "insufficient information",
    "no information found",
    "not discussed",
    "no data available",
    "cannot be determined",
    "not enough information",
]


def eval_adversarial(output: str) -> Dict[str, Any]:
    """Evaluate adversarial question using keyword matching.
    
    For adversarial questions, the correct answer is that the info doesn't exist.
    If the system output contains keywords indicating 'no info', it's correct.
    If the system fabricates an answer, it's wrong.
    
    Returns:
        Dict with verdict (CORRECT/WRONG), score (1.0/0.0), and reasoning.
    """
    output_lower = output.lower().strip()
    
    for keyword in ADVERSARIAL_CORRECT_KEYWORDS:
        if keyword in output_lower:
            return {
                "verdict": "CORRECT",
                "score": 1.0,
                "reasoning": f"Adversarial: correctly indicated no information (matched: '{keyword}')",
            }
    
    return {
        "verdict": "WRONG",
        "score": 0.0,
        "reasoning": f"Adversarial: system fabricated an answer instead of indicating no information. Answer: '{output[:100]}'",
    }


# ---------------------------------------------------------------------------
# LLM-as-Judge evaluation (for non-adversarial categories)
# ---------------------------------------------------------------------------

def judge_answer(
    question: str,
    ground_truth: str,
    system_answer: str,
    client: Any,
) -> Dict[str, Any]:
    """Use LLM to judge if system answer matches ground truth (LoCoMo paper style)."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        system_answer=system_answer,
    )
    
    try:
        text = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        
        # Try to extract JSON with "label" key
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            judgment = json.loads(json_match.group())
            label = judgment.get("label", judgment.get("verdict", "")).upper()
            reasoning = text[:json_match.start()].strip()
            return {
                "verdict": label,
                "score": 1.0 if label == "CORRECT" else 0.0,
                "reasoning": reasoning,
            }
        
        # Fallback: look for CORRECT or WRONG in raw text
        text_upper = text.upper()
        if "CORRECT" in text_upper and "WRONG" not in text_upper:
            return {"verdict": "CORRECT", "score": 1.0, "reasoning": text.strip()}
        elif "WRONG" in text_upper and "CORRECT" not in text_upper:
            return {"verdict": "WRONG", "score": 0.0, "reasoning": text.strip()}
        else:
            return {"verdict": "ERROR", "score": 0.0, "reasoning": f"Could not parse judge response: {text}"}
    except Exception as e:
        return {"verdict": "ERROR", "score": 0.0, "reasoning": str(e)}


# ---------------------------------------------------------------------------
# Top-level evaluation dispatcher
# ---------------------------------------------------------------------------

def evaluate_answer(
    category: str,
    system_answer: str,
    ground_truth: str,
    question: str,
    client: Any = None,
) -> Dict[str, Any]:
    """Evaluate a system answer based on question category.
    
    - adversarial → keyword matching (no LLM needed)
    - all others  → LLM-as-judge
    
    Args:
        category: Question category string (single_hop, multi_hop, temporal, open_domain, adversarial)
        system_answer: The system's generated answer
        ground_truth: The gold/expected answer
        question: The original question text
        client: OpenAI client (required for non-adversarial)
    
    Returns:
        Dict with verdict, score, and reasoning.
    """
    if category == "adversarial":
        res = eval_adversarial(system_answer)
    else:
        # All non-adversarial categories use LLM judge
        if client is None:
            raise ValueError("LLM client required for non-adversarial evaluation")
        res = judge_answer(question, ground_truth, system_answer, client)
    
    # Remove Prompt-enforced prefixes before computing lexical metrics
    clean_answer = system_answer.replace("**Answer:**", "").replace("Answer:", "").strip()
    
    # Calculate F1 and BLEU-1 for all responses based on cleaned answer
    res["f1"] = f1_score(clean_answer, ground_truth)
    res["bleu_1"] = bleu_1_score(clean_answer, ground_truth)
    
    return res

# ---------------------------------------------------------------------------
# Lexical Metric Utils: F1 and BLEU-1
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)

def bleu_1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = [normalize_answer(ground_truth).split()]
    if not pred_tokens or not ref_tokens[0]:
        return 0.0
    smoothie = SmoothingFunction().method1
    score = sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    return round(score, 4)

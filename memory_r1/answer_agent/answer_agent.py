"""
Answer Agent for Memory-R1 and Struct Memory-R1.

Implements the Answer Agent with Memory Distillation: given a question and
retrieved memories, the agent selects relevant memories and generates a
concise answer. Faithful to the Memory-R1 paper (Yan et al., 2025).
"""

import re
from typing import List, Dict, Optional, Tuple, Any


def extract_answer(llm_output: str) -> Optional[str]:
    """Extract the answer from the Answer Agent's output.

    Looks for text after 'Answer:' or '**Answer:**' markers.
    """
    patterns = [
        r'\*\*Answer:\*\*\s*(.*)',
        r'Answer:\s*(.*)',
        r'<answer>(.*?)</answer>',
    ]
    for pattern in patterns:
        match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = answer.split('\n')[0].strip()
            answer = answer.strip('*').strip()
            return answer
    return None


def extract_selected_memories(llm_output: str) -> List[str]:
    """Extract the memories selected by the distillation step.

    Looks for listed memories before the Answer: marker.
    """
    answer_split = re.split(r'(?:\*\*)?Answer:(?:\*\*)?', llm_output, maxsplit=1)
    if len(answer_split) < 2:
        return []

    pre_answer = answer_split[0]
    memories = []
    for line in pre_answer.split('\n'):
        line = line.strip()
        if line.startswith('-') or line.startswith('•') or line.startswith('*'):
            memories.append(line.lstrip('-•* ').strip())
        elif re.match(r'^\d+[\.\)]\s', line):
            memories.append(re.sub(r'^\d+[\.\)]\s', '', line).strip())
    return memories


def format_flat_memory_context(memories: List[Dict],
                                 group_by_speaker: bool = True) -> str:
    """Format flat memories for the Answer Agent prompt.

    Args:
        memories: list of {id, text, speaker, timestamp} dicts
        group_by_speaker: if True, group memories by speaker
    """
    if not memories:
        return "No memories available."

    if group_by_speaker:
        by_speaker = {}
        for m in memories:
            spk = m.get("speaker", "unknown")
            by_speaker.setdefault(spk, []).append(m)

        parts = []
        for spk, mems in by_speaker.items():
            parts.append(f"Memories for user {spk}:")
            for m in mems:
                ts = m.get("timestamp", "")
                if ts:
                    parts.append(f"- {ts}: {m['text']}")
                else:
                    parts.append(f"- {m['text']}")
            parts.append("")
        return "\n".join(parts)
    else:
        parts = []
        for i, m in enumerate(memories):
            ts = m.get("timestamp", "")
            if ts:
                parts.append(f"Memory {i+1} ({ts}): {m['text']}")
            else:
                parts.append(f"Memory {i+1}: {m['text']}")
        return "\n".join(parts)


def format_tree_memory_context(memories_with_paths: List[Dict]) -> str:
    """Format tree-structured memories for the Answer Agent prompt.

    Args:
        memories_with_paths: list of dicts with 'path', 'text', 'timestamp' keys
    """
    if not memories_with_paths:
        return "No memories available."

    parts = ["Retrieved memories from tree-structured memory:"]
    for i, m in enumerate(memories_with_paths):
        path = m.get("path", "")
        text = m.get("text", m.get("contents", ""))
        ts = m.get("timestamp", "")
        ts_str = f" ({ts})" if ts else ""
        parts.append(f"Memory {i+1} [Path: {path}]{ts_str}:")
        parts.append(f"  {text}")
    return "\n".join(parts)


class AnswerAgent:
    """Answer Agent with Memory Distillation.

    Given a question and retrieved memories, selects the most relevant
    memories and produces a concise answer.
    """

    def __init__(self, memory_type: str = "flat"):
        self.memory_type = memory_type

    def format_input(self, question: str, memories: List[Dict]) -> str:
        """Format the complete prompt for the Answer Agent."""
        if self.memory_type == "structured":
            context = format_tree_memory_context(memories)
        else:
            context = format_flat_memory_context(memories)

        return f"{context}\n\nQuestion: {question}\n"

    def parse_output(self, llm_output: str) -> Dict[str, Any]:
        """Parse the Answer Agent's output into structured components."""
        answer = extract_answer(llm_output)
        selected = extract_selected_memories(llm_output)
        return {
            "answer": answer,
            "selected_memories": selected,
            "raw_output": llm_output,
        }

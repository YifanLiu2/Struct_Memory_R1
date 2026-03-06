"""
Prompt templates for the Answer Agent.

Flat prompts follow the Memory-R1 paper (Yan et al., 2025) Figure 11.
Tree prompts extend these with hierarchical path context for Struct Memory-R1.
"""

# ---------------------------------------------------------------------------
# Answer Agent prompt (faithful to paper Figure 11)
# ---------------------------------------------------------------------------

ANSWER_AGENT_SYSTEM = """\
You are an intelligent memory assistant tasked with retrieving \
accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation.
These memories contain timestamped information that may be relevant \
to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago"), \
calculate the actual date based on the memory timestamp.
6. Always convert relative time references to specific dates, months, or years.
7. Focus only on the content of the memories. Do not confuse character names
8. The answer should be less than 5-6 words.
9. IMPORTANT: Select memories you found that are useful for answering the questions, \
and output it before you answer questions.
10. IMPORTANT: Output the final answer after **Answer:**

# APPROACH (Think step by step):
1. Examine all relevant memories
2. Examine the timestamps carefully
3. Look for explicit mentions that answer the question
4. Convert relative references if needed
5. Formulate a concise answer
6. Double-check the answer correctness
7. Ensure the final answer is specific
8. First output the memories that you found are important before you answer questions
"""

ANSWER_AGENT_USER_FLAT = """\
{memory_context}

Question: {question}
"""

# ---------------------------------------------------------------------------
# Tree-formatted Answer Agent prompt (Struct Memory-R1)
# ---------------------------------------------------------------------------

ANSWER_AGENT_SYSTEM_TREE = """\
You are an intelligent memory assistant tasked with retrieving \
accurate information from a tree-structured conversation memory.

# CONTEXT:
You have access to memories organized in a hierarchical tree. Each memory \
entry has a path showing its position in the tree (e.g., \
Dialogue / Session 3 / Topic: Travel / Memory Entry). The tree structure \
encodes temporal and thematic relationships between memories.

# INSTRUCTIONS:
1. Carefully analyze all provided memories, paying attention to their \
tree paths for contextual understanding
2. Pay special attention to the timestamps and session ordering
3. If the question asks about a specific event or fact, look for direct evidence
4. If the memories contain contradictory information, prioritize the most recent memory
5. Use the tree structure to understand relationships between memories \
(e.g., memories under the same session/topic are related)
6. The answer should be less than 5-6 words.
7. IMPORTANT: Select memories you found useful and output them before answering.
8. IMPORTANT: Output the final answer after **Answer:**

# APPROACH (Think step by step):
1. Examine all relevant memories and their tree positions
2. Use the hierarchical context to disambiguate similar memories
3. Look for explicit mentions that answer the question
4. Formulate a concise answer
5. First output the relevant memories, then provide the answer
"""

ANSWER_AGENT_USER_TREE = """\
{memory_context}

Question: {question}
"""


# ---------------------------------------------------------------------------
# LLM-as-a-Judge prompt (from paper Figure 12)
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
the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace

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

Return the label in JSON format with the key as "label".
"""


def format_flat_memories_for_answer(memories_by_speaker: dict) -> str:
    """Format retrieved flat memories grouped by speaker.

    Args:
        memories_by_speaker: {speaker_name: [(text, timestamp), ...]}
    """
    parts = []
    for speaker, mems in memories_by_speaker.items():
        parts.append(f"Memories for user {speaker}:")
        for text, ts in mems:
            if ts:
                parts.append(f"- {ts}: {text}")
            else:
                parts.append(f"- {text}")
        parts.append("")
    return "\n".join(parts)


def format_tree_memories_for_answer(memories_with_paths: list) -> str:
    """Format retrieved tree memories with their hierarchical paths.

    Args:
        memories_with_paths: [(path, text, timestamp), ...]
    """
    parts = ["Retrieved memories from the tree-structured memory bank:"]
    for path, text, ts in memories_with_paths:
        ts_str = f" ({ts})" if ts else ""
        parts.append(f"- [{path}]{ts_str}: {text}")
    return "\n".join(parts)

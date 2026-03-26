"""
LoCoMo Evaluation Prompts 
"""

# # Answer Agent prompt
# ANSWER_PROMPT = """You are an Answer Agent. Given a question and retrieved memory observations, provide a concise, direct answer.

# Question: {question}

# Retrieved Observations:
# {context}

# Rules:
# - Answer the question directly and concisely based ONLY on the provided observations.
# - If the observations contain the answer, state it clearly in 1-2 sentences.
# - If the observations do not contain enough information to answer, say "No relevant information found"
# - Do NOT add speculation beyond what the observations state.
# - For temporal questions, use specific dates/times from the observation metadata when available.

# Answer:"""

ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation.
These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago"), calculate the actual date based on the memory timestamp
6. Always convert relative time references to specific dates, months, or years
7. Focus only on the content of the memories. Do not confuse character names
8. The answer should be less than 5-6 words
9. IMPORTANT: Output the final answer after **Answer:**

# APPROACH (Think step by step):
1. Examine all relevant memories
2. Examine the timestamps carefully
3. Look for explicit mentions that answer the question
4. Convert relative references if needed
5. Formulate a concise answer
6. Double-check the answer correctness
7. Ensure the final answer is specific
8. Answer the question based ONLY on the provided observations.

Question: {question}

Retrieved Observations:
{context}

Answer:
"""




# LLM-as-Judge prompt (from LoCoMo paper)
JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You will be given the following data:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer,
which you will score as CORRECT or WRONG.
The point of the question is to ask about something one user should know about the other user based on their
prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be longer, but you should be generous with your grading — as long as it touches
on the same topic as the gold answer, it should be counted as CORRECT.
For time-related questions, the gold answer will be a specific date, month, or year. The generated answer
might include relative references (e.g., "last Tuesday"), but you should be generous — if it refers to
the same time period as the gold answer, mark it CORRECT, even if the format differs (e.g., "May 7th" vs.
"7 May").
Now it's time for the real question:
Question: {question}
Gold answer: {ground_truth}
Generated answer: {system_answer}
First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
Return the label in JSON format with the key as "label"."""

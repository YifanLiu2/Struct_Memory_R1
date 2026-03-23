"""
LoCoMo Evaluation Prompts - Separated from locomo_eval.py for clarity.
"""

# Answer Agent prompt (Memory-R1 style)
ANSWER_PROMPT = """You are an Answer Agent. Given a question and retrieved memory observations, provide a concise, direct answer.

Question: {question}

Retrieved Observations:
{context}

Rules:
- Answer the question directly and concisely based ONLY on the provided observations.
- If the observations contain the answer, state it clearly in 1-2 sentences.
- If the observations do not contain enough information to answer, say "No relevant information found"
- Do NOT add speculation beyond what the observations state.
- For temporal questions, use specific dates/times from the observation metadata when available.

Answer:"""

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

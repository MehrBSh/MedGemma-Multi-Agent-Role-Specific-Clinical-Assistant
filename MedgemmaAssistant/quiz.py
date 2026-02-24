# ============================================================
# Generate MCQs from a topic; run quiz with feedback.
# ============================================================

import re
import rag_engine


def generate_quiz(
    topic: str,
    embedder,
    coll,
    processor,
    model,
    num_questions: int = 3,
) -> list[dict]:
    """Returns [{"question": "...", "options": ["A) ...", "B) ...", ...], "correct_index": 0}, ...]."""
    chunks = rag_engine.retrieve(embedder, coll, topic, top_k=5)
    if not chunks:
        return []
    context = "\n\n".join(chunks[:3])
    prompt = f"""Using only the context below, create exactly {num_questions} multiple-choice questions. Each must have 4 options (A, B, C, D) and you must indicate the correct one.
Format each question as:
Q1: [question text]
A) option A
B) option B
C) option C
D) option D
Correct: A

Context:
{context}

Questions:"""
    messages = rag_engine.build_messages_text_only(
        prompt,
        system="You output only the numbered questions in the exact format above. Correct: must be A, B, C, or D.",
    )
    reply = rag_engine.generate(processor, model, messages, max_new_tokens=600)
    return _parse_quiz(reply)


def _parse_quiz(text: str) -> list[dict]:
    questions = []
    # Split by Q1:, Q2:, Q3: or similar
    blocks = re.split(r"\n\s*Q\d+:\s*", text, flags=re.IGNORECASE)[1:]
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) < 5:
            continue
        question_text = lines[0]
        options = []
        correct_index = -1
        for i, line in enumerate(lines[1:6]):
            if re.match(r"^[A-D]\)", line, re.IGNORECASE):
                options.append(line)
            if re.match(r"^Correct:\s*[A-D]", line, re.IGNORECASE):
                letter = line.split(":")[-1].strip().upper()
                if letter in "ABCD" and len(options) >= 4:
                    correct_index = ord(letter) - ord("A")
        if len(options) == 4 and 0 <= correct_index <= 3:
            questions.append({
                "question": question_text,
                "options": options,
                "correct_index": correct_index,
            })
    return questions[:5]


def get_explanation(
    question: str,
    options: list[str],
    correct_index: int,
    user_choice_index: int,
    processor,
    model,
) -> str:
    """Short explanation for quiz feedback."""
    correct_letter = "ABCD"[correct_index]
    prompt = f"""In one or two sentences, briefly explain why the correct answer is {correct_letter} for this question (and why the other options are wrong if the user picked wrong). Be concise and educational.

Question: {question}
Options: {chr(10).join(options)}
Correct: {correct_letter}
User chose: {"ABCD"[user_choice_index]}"""
    messages = rag_engine.build_messages_text_only(prompt)
    return rag_engine.generate(processor, model, messages, max_new_tokens=150)

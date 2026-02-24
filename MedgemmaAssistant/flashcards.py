# ============================================================
# Generate flashcards from topic or last Q&A; save and review.
# ============================================================

import re
import store
import rag_engine


def generate_flashcards_from_topic(
    topic: str,
    embedder,
    coll,
    processor,
    model,
    num_cards: int = 5,
) -> list[dict]:
    """Use RAG to get context on topic, then ask LLM to generate flashcards. Returns [{"front": "...", "back": "..."}, ...]."""
    chunks = rag_engine.retrieve(embedder, coll, topic, top_k=5)
    if not chunks:
        return []
    context = "\n\n".join(chunks[:3])
    prompt = f"""Based only on the context below, create exactly {num_cards} flashcards for studying. Each card should have a clear front (question or term) and back (answer or definition).
Format each card on its own line as: Front: ... Back: ...
Use short, clear phrases. Context:

{context}

Flashcards:"""
    messages = rag_engine.build_messages_text_only(
        prompt,
        system="You output only the requested flashcards in the exact format: Front: ... Back: ... one per line. No other text.",
    )
    reply = rag_engine.generate(processor, model, messages, max_new_tokens=512)
    return _parse_flashcards(reply)


def generate_flashcards_from_qa(question: str, answer: str, processor, model, num_cards: int = 3) -> list[dict]:
    """Generate flashcards from the last Q&A pair."""
    prompt = f"""From this Q&A, create exactly {num_cards} short flashcards (term/definition or question/answer).
Q: {question}
A: {answer}

Format each as: Front: ... Back: ...
Flashcards:"""
    messages = rag_engine.build_messages_text_only(
        prompt,
        system="You output only the requested flashcards in the format: Front: ... Back: ... one per line. No other text.",
    )
    reply = rag_engine.generate(processor, model, messages, max_new_tokens=400)
    return _parse_flashcards(reply)


def _parse_flashcards(text: str) -> list[dict]:
    cards = []
    # Match "Front: ... Back: ..." (allow newlines within)
    pattern = re.compile(r"Front:\s*(.+?)\s*Back:\s*(.+?)(?=Front:|$)", re.DOTALL | re.IGNORECASE)
    for m in pattern.finditer(text):
        front = m.group(1).strip()
        back = m.group(2).strip()
        if front and back and len(front) < 500 and len(back) < 500:
            cards.append({"front": front, "back": back})
    # Fallback: split by "Front:" and then "Back:"
    if not cards and "Front:" in text:
        parts = re.split(r"\s*Front:\s*", text, flags=re.IGNORECASE)[1:]
        for p in parts:
            if "Back:" in p or "back:" in p:
                front, _, back = re.split(r"\s*Back:\s*", p, maxsplit=1, flags=re.IGNORECASE)
                front, back = front.strip(), back.strip()
                if front and back:
                    cards.append({"front": front[:400], "back": back[:400]})
    return cards[:10]


def save_generated_flashcards(cards: list[dict], topic: str | None = None) -> int:
    return store.save_flashcards(cards, topic=topic)


def get_review_deck(limit: int = 20) -> list[dict]:
    return store.get_flashcards_for_review(limit=limit)


def get_all_cards(topic: str | None = None) -> list[dict]:
    return store.get_all_flashcards(topic=topic)

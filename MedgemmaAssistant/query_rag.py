# ============================================================
# CLI: RAG + Image with citations, memory, save, flashcards, quiz, export.
# ============================================================

import os
import sys
from pathlib import Path

import rag_engine
import store
import flashcards
import quiz
import export_notes
from PIL import Image

store.init_db()

CHROMA_DIR = rag_engine.CHROMA_DIR


def main():
    if not os.path.isdir(CHROMA_DIR):
        print("Run build_rag_index.py first to create", CHROMA_DIR)
        return

    print("Loading retriever and ChromaDB...")
    embedder, coll = rag_engine.load_retriever()
    print("Loading MedGemma...")
    processor, model = rag_engine.load_medgemma()

    conversation_history = []

    print("\nUsage:")
    print("  [question]           - Ask (RAG). Answer shows with sources.")
    print("  image: <path>        - Image + optional question.")
    print("  /save                - Save last Q&A to My notes.")
    print("  /flashcards [topic]  - Generate flashcards from topic or last Q&A.")
    print("  /quiz [topic]        - Start a short quiz on topic.")
    print("  /export              - Export My notes to Markdown.")
    print("  Enter nothing to quit.\n")

    last_question, last_answer, last_sources = None, None, []

    while True:
        line = input("Question or image path: ").strip()
        if not line:
            break

        # Commands
        if line.lower() == "/save":
            if last_question and last_answer is not None:
                store.save_note(last_question, last_answer, last_sources)
                print("Saved to My notes.")
            else:
                print("No previous Q&A to save.")
            continue
        if line.lower() == "/export":
            path = export_notes.export_to_markdown()
            print("Exported to", path)
            continue
        if line.lower().startswith("/flashcards"):
            topic = line[10:].strip() if len(line) > 10 else None
            if topic:
                cards = flashcards.generate_flashcards_from_topic(topic, embedder, coll, processor, model)
            elif last_question and last_answer:
                cards = flashcards.generate_flashcards_from_qa(last_question, last_answer, processor, model)
            else:
                print("Provide a topic (e.g. /flashcards corticosteroids) or ask a question first for /flashcards from last Q&A.")
                continue
            if not cards:
                print("No flashcards generated.")
                continue
            n = flashcards.save_generated_flashcards(cards, topic=topic or "last_qa")
            print(f"Saved {n} flashcards. Use the web app to review (/review in future).")
            continue
        if line.lower().startswith("/quiz"):
            topic = line[5:].strip() or (last_question if last_question else "general medicine")
            questions = quiz.generate_quiz(topic, embedder, coll, processor, model, num_questions=3)
            if not questions:
                print("No quiz generated. Try a different topic.")
                continue
            score = 0
            for i, q in enumerate(questions, 1):
                print(f"\n--- Question {i} ---")
                print(q["question"])
                for opt in q["options"]:
                    print(" ", opt)
                choice = input("Your answer (A/B/C/D): ").strip().upper()
                idx = ord(choice) - ord("A") if choice in "ABCD" else -1
                if idx == q["correct_index"]:
                    score += 1
                    print("Correct!")
                else:
                    correct_letter = "ABCD"[q["correct_index"]]
                    print(f"Wrong. Correct is {correct_letter}.")
                expl = quiz.get_explanation(q["question"], q["options"], q["correct_index"], idx if idx >= 0 else 0, processor, model)
                print("Explanation:", expl)
            print(f"\nScore: {score}/{len(questions)}")
            continue

        # Image or text question
        if rag_engine.is_image_path(line):
            path = line.strip('"')
            rest = ""
            if line.lower().startswith("image:"):
                parts = line[6:].strip().split(maxsplit=1)
                path = parts[0].strip('"') if parts else path
                rest = parts[1] if len(parts) > 1 else ""
            question = rest or "Describe the findings in this medical image."
            try:
                image = Image.open(path).convert("RGB")
            except Exception as e:
                print("Could not load image:", e)
                continue
            print("Generating...")
            last_answer, last_sources = rag_engine.ask_image(image, question, embedder, coll, processor, model)
            last_question = question
        else:
            question = line
            print("Generating...")
            last_answer, last_sources = rag_engine.ask_text(question, embedder, coll, processor, model, conversation_history=conversation_history)
            last_question = question

        print("Answer:", last_answer)
        if last_sources:
            print("\nBased on:")
            for i, s in enumerate(last_sources[:3], 1):
                snippet = (s[:180] + "...") if len(s) > 180 else s
                print(f"  [{i}] {snippet}")

        conversation_history.append({"question": last_question, "answer": last_answer})
        if len(conversation_history) > 5:
            conversation_history.pop(0)


if __name__ == "__main__":
    main()

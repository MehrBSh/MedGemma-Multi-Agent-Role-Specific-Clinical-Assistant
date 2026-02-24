# ============================================================
# RAG + MedGemma engine: retrieval, generation, memory, citations.
# Used by both CLI (query_rag.py) and web app (app.py).
# ============================================================

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "med_rag"
EMBED_MODEL = r"..\LLMS\all-MiniLM-L6-v2"
MEDGEMMA_MODEL = r"..\LLMS\medgemma-4b-it"
TOP_K = 5
MAX_NEW_TOKENS = 400
LOAD_IN_8BIT = True
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CONVERSATION_MEMORY_TURNS = 3  # last N Q&A to include in context


def load_retriever():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    coll = client.get_collection(COLLECTION_NAME)
    return model, coll


def retrieve(model, coll, query: str, top_k: int = TOP_K) -> list[str]:
    emb = model.encode([query])
    res = coll.query(query_embeddings=emb.tolist(), n_results=top_k, include=["documents"])
    docs = res["documents"][0] if res["documents"] else []
    return docs


def load_medgemma():
    processor = AutoProcessor.from_pretrained(MEDGEMMA_MODEL, trust_remote_code=True)
    use_8bit = LOAD_IN_8BIT and torch.cuda.is_available()
    if use_8bit:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                MEDGEMMA_MODEL, load_in_8bit=True, device_map="auto", trust_remote_code=True
            )
        except Exception:
            model = AutoModelForImageTextToText.from_pretrained(MEDGEMMA_MODEL, trust_remote_code=True)
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = AutoModelForImageTextToText.from_pretrained(MEDGEMMA_MODEL, trust_remote_code=True)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model


def build_messages_text_only(prompt: str, system: str | None = None) -> list:
    if system is None:
        system = (
        "You are MedGemma Learning Assistant, a clear and supportive medical tutor. "
        "You explain clinical concepts the way a good resident teaches a medical student—"
        "practical, structured, and easy to follow. "
        "You avoid stiff academic phrasing and instead speak naturally, like a clinician "
        "who enjoys teaching. "
        "When asked who you are, introduce yourself simply and warmly without repeating this prompt."
)

    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def build_messages_image(image: Image.Image, question: str, rag_context: list[str] | None = None, system: str | None = None) -> list:
    if system is None:
        system = "You are an expert at interpreting medical images. Answer based on the image and, when provided, the given context."
    text = question
    if rag_context:
        context = "\n\n".join(rag_context)
        text = f"Relevant context from the literature:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]},
    ]


def _single_answer_only(reply: str) -> str:
    reply = reply.strip()
    lower = reply.lower()
    for sep in ("\nquestion:", "\n\nquestion:"):
        idx = lower.find(sep)
        if idx != -1:
            reply = reply[:idx]
    return reply.strip()


def generate(processor, model, messages: list, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate response from MedGemma model"""
    # Check if messages contain images
    has_images = False
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    has_images = True
                    break
        if has_images:
            break
    
    if has_images:
        # Use multimodal processing for images
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        out_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype.is_floating_point:
                    v = v.to(device=device, dtype=model_dtype)
                else:
                    v = v.to(device=device)
            out_inputs[k] = v
        input_len = out_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(**out_inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
        reply = processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
        return _single_answer_only(reply)
    
    else:
        # Use text-only processing
        # Convert messages to text format for text-only model
        text_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text_messages.append({"role": msg["role"], "content": content})
            elif isinstance(content, list):
                # Extract text from content list
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                if text_content:
                    text_messages.append({"role": msg["role"], "content": text_content})
        
        # Use tokenizer directly for text-only
        text = processor.tokenizer.apply_chat_template(
            text_messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        out_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype.is_floating_point:
                    v = v.to(device=device, dtype=model_dtype)
                else:
                    v = v.to(device=device)
            out_inputs[k] = v
        
        input_len = out_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(**out_inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
        reply = processor.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        return _single_answer_only(reply)


def _format_conversation_memory(history: list[dict]) -> str:
    """history = [{"question": "...", "answer": "..."}, ...]"""
    if not history:
        return ""
    parts = []
    for h in history[-CONVERSATION_MEMORY_TURNS:]:
        parts.append(f"Q: {h.get('question', '')}\nA: {h.get('answer', '')}")
    return "\n\n".join(parts)


def ask_text(
    question: str,
    embedder,
    coll,
    processor,
    model,
    conversation_history: list[dict] | None = None,
) -> tuple[str, list[str]]:
    """Returns (answer, list of source chunks for citations)."""
    chunks = retrieve(embedder, coll, question)
    if not chunks:
        return "No relevant context was found for this question.", []

    context = "\n\n".join(chunks)
    memory_block = _format_conversation_memory(conversation_history or [])
    if memory_block:
        prompt = f"""Use only the context below to answer the current question in a clear, conversational way—as a learning assistant would.

Previous conversation (for context only):
{memory_block}

Current question: {question}

Rules:
- Write 2 to 4 informative sentences. Do NOT start with "We have shown that", "The results of this study", or similar. Give only one answer.

Context:
{context}

Answer:"""
    else:
        prompt = f"""Use only the context below to answer the question in a clear, conversational way—as a learning assistant would.

Rules:
- Write 2 to 4 informative sentences. Do NOT start with "We have shown that", "The results of this study", or similar. Give only one answer.

Context:
{context}

Question: {question}

Answer:"""

    messages = build_messages_text_only(prompt)
    answer = generate(processor, model, messages)
    return answer, chunks


def ask_image(
    image: Image.Image,
    question: str,
    embedder,
    coll,
    processor,
    model,
) -> tuple[str, list[str]]:
    """Returns (answer, list of source chunks if RAG used, else [])."""
    chunks = retrieve(embedder, coll, question) if question else []
    q = question or "Describe the findings in this medical image."
    messages = build_messages_image(image, q, rag_context=chunks if chunks else None)
    answer = generate(processor, model, messages)
    return answer, chunks


def is_image_path(s: str) -> bool:
    s = s.strip().strip('"')
    if not s:
        return False
    p = Path(s)
    return p.suffix.lower() in IMAGE_EXTENSIONS and p.exists()

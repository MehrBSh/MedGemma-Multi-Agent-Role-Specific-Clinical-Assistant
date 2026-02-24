import streamlit as st
from PIL import Image
from datetime import datetime

import rag_engine

# ============================================================
# Patient-safe system prompt
# ============================================================

PATIENT_SYSTEM_PROMPT = """
You are the MedGemma Patient & Caregiver Assistant.

Your role:
- Provide clear, simple, supportive explanations.
- Help patients understand symptoms WITHOUT diagnosing.
- Explain medications in general terms (purpose, how they work).
- Simplify medical reports into plain language.
- Describe medical images in general, understandable terms without giving a diagnosis.
- Support chronic disease understanding and self-management education.
- Encourage appropriate medical follow-up.
- NEVER provide diagnosis, treatment plans, or medical instructions.
- NEVER give medication dosing, timing, or personalized advice.
- NEVER contradict a clinician.

Tone:
- Warm, reassuring, calm.
- Use simple language.
- Avoid medical jargon unless explaining it.
- Focus on understanding, not decision-making.

Additional safety rules:
- Do NOT say that a condition is “under control”, “well managed”, “mild”, “severe”, “improving”, or “worsening”.
- Do NOT make statements about prognosis, risk level, or how well a condition is being managed.
- Do NOT imply that test results or images show improvement, stability, or deterioration.
- Do NOT reassure the user about safety, risk, or outcomes.
- If the report or image sounds reassuring, simply say that it describes findings that should be discussed with the treating clinician.
- If the user asks for interpretation, prognosis, or treatment advice, gently redirect them to their healthcare professional.

Always include:
- A brief safety reminder when discussing symptoms, medications, reports, or images.
"""

# ============================================================
# Forbidden phrases + required phrases + postprocessor
# ============================================================

FORBIDDEN_PATIENT_PHRASES = [
    # Interpretation of normality/abnormality
    "abnormal",
    "abnormality",
    "abnormalities",
    "normal for",
    "lower than expected",
    "higher than expected",
    "slightly longer",
    "slightly shorter",
    "longer than normal",
    "shorter than normal",
    "above normal",
    "below normal",

    # Diagnostic or confirmatory language
    "consistent with",
    "suggestive of",
    "indicative of",
    "points to",
    "in line with",
    "fits with",

    # Risk / causal language
    "can cause dangerous",
    "can cause irregular",
    "can cause serious",
    "can lead to",
    "can result in",
    "dangerous heart rhythm",
    "dangerous heart rhythms",
    "life-threatening",
    "risk of sudden",

]

REQUIRED_PATIENT_PHRASES = [
    "The report or image describes the findings and should be discussed with the treating clinician.",
    "These observations are part of the medical information and can be reviewed with the healthcare team.",
    "If you have questions about what these measurements or images mean, your clinician can explain them in more detail.",
]


def sanitize_patient_output(text: str) -> str:
    import re

    sentences = re.split(r'(?<=[.!?])\s+', text)
    safe_sentences = []
    for s in sentences:
        lower = s.lower()
        if any(p in lower for p in FORBIDDEN_PATIENT_PHRASES):
            continue
        safe_sentences.append(s)

    cleaned = " ".join(safe_sentences).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def enforce_required_phrases(text: str) -> str:
    result = text.strip()
    for phrase in REQUIRED_PATIENT_PHRASES:
        if phrase.lower() not in result.lower():
            result += " " + phrase
    return result.strip()


def postprocess_patient_output(text: str) -> str:
    cleaned = sanitize_patient_output(text)
    final = enforce_required_phrases(cleaned)
    return final


# ============================================================
# Model loading (RAG + MedGemma)
# ============================================================

@st.cache_resource
def load_patient_models():
    embedder, coll = rag_engine.load_retriever()
    processor, model = rag_engine.load_medgemma()
    return embedder, coll, processor, model


# ============================================================
# Patient notes (no sharing with student)
# ============================================================

def _init_patient_notes():
    if "patient_notes" not in st.session_state:
        st.session_state.patient_notes = []  # list of {question, answer, created_at}


def _add_patient_note(question: str, answer: str):
    _init_patient_notes()
    st.session_state.patient_notes.append(
        {
            "question": question,
            "answer": answer,
            "created_at": datetime.now().isoformat(),
        }
    )


# ============================================================
# CHAT TAB
# ============================================================

def patient_chat_tab(processor, model):
    if "patient_conversation" not in st.session_state:
        st.session_state.patient_conversation = [
            {
                "question": None,
                "answer": (
                    "👋 Hello! I'm your **MedGemma Patient & Caregiver Assistant**.\n\n"
                    "I can help you understand symptoms, medications, medical reports, "
                    "medical images, and chronic conditions in clear, simple language.\n\n"
                    "How can I help you today?"
                ),
            }
        ]

    _init_patient_notes()

    st.markdown("### 💬 Patient & Caregiver Chat")

    for h in st.session_state.patient_conversation:
        if h["question"]:
            with st.chat_message("user"):
                st.markdown(h["question"])
        with st.chat_message("assistant"):
            st.markdown(h["answer"])

    user_input = st.chat_input("Describe your symptoms, medication, report, or question...")

    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)

    # Use system prompt as system, user_input as user content
    with st.spinner("Thinking..."):
        messages = rag_engine.build_messages_text_only(
            prompt=user_input,
            system=PATIENT_SYSTEM_PROMPT,
        )
        raw_answer = rag_engine.generate(
            processor, model, messages, max_new_tokens=350
        )
        answer = postprocess_patient_output(raw_answer)

    st.session_state.patient_conversation.append(
        {"question": user_input, "answer": answer}
    )
    _add_patient_note(user_input, answer)
    st.rerun()


# ============================================================
# NOTES TAB (patient-only)
# ============================================================

def patient_notes_tab():
    st.markdown("### 📘 My Health Notes")
    _init_patient_notes()

    notes = st.session_state.patient_notes

    if not notes:
        st.info("No health notes yet. Notes are saved automatically from your chat and tools.")
        return

    for n in notes:
        title = n["question"] if len(n["question"]) < 60 else n["question"][:57] + "..."
        with st.expander(f"📝 {title}"):
            st.markdown(f"**Date:** {n['created_at'][:10]}")
            st.markdown(f"**Question:** {n['question']}")
            st.markdown(f"**Explanation:** {n['answer']}")


# ============================================================
# REPORT & IMAGE SIMPLIFIER TAB (RAG + Vision)
# ============================================================

def report_simplifier_tab(embedder, coll, processor, model):
    st.markdown("### 🧾 Report & Image Simplifier")

    uploaded = st.file_uploader(
        "Upload a medical report or medical image (e.g., ECG, X-ray, MRI, lab report)",
        type=["png", "jpg", "jpeg", "pdf"],
    )

    if not uploaded:
        return

    if uploaded.type.startswith("image"):
        image = Image.open(uploaded).convert("RGB")

        # Task prompt: works for both text reports and medical images
        question = (
    "You are simplifying a medical report or medical image for a patient. "
    "ONLY describe what the report or image says was observed or measured. "
    "Do NOT explain what the condition is, do NOT say what it can cause, "
    "and do NOT talk about risks, danger, or outcomes. "
    "If a diagnosis name (like Long QT Syndrome) appears, you may repeat the name "
    "as part of the report, but do NOT explain it or say what it can lead to. "
    "Do NOT say that a condition is under control, well managed, mild, severe, "
    "improving, or worsening. Do NOT make statements about prognosis, risk, or "
    "how well the condition is being managed. Do NOT give reassurance or interpretation. "
    "If the report or image sounds reassuring, simply say that it describes findings "
    "that should be discussed with the treating clinician. "
    "If the user asks for interpretation, prognosis, or treatment advice, gently "
    "redirect them to their healthcare professional."
)


        with st.spinner("Simplifying report or image..."):
            # RAG context based on the question text
            chunks = rag_engine.retrieve(embedder, coll, question)

            messages = rag_engine.build_messages_image(
                image=image,
                question=question,
                rag_context=chunks if chunks else None,
                system=PATIENT_SYSTEM_PROMPT,
            )

            raw_answer = rag_engine.generate(
                processor, model, messages, max_new_tokens=350
            )
            answer = postprocess_patient_output(raw_answer)

        st.markdown("### 🧾 Simplified Explanation")
        st.markdown(answer)
        _add_patient_note(f"[Report/Image: {uploaded.name}]", answer)


# ============================================================
# MEDICATION HELPER TAB
# ============================================================

def medication_helper_tab(processor, model):
    st.markdown("### 💊 Medication Helper")

    med = st.text_input("Enter a medication name")

    if not med:
        return

    prompt = (
        f"Explain the medication '{med}' in simple language. "
        "Describe its general purpose and how it works in the body. "
        "Do NOT give dosing, timing, or personalized advice. "
        "Remind the user to follow their clinician's instructions."
    )

    with st.spinner("Explaining medication..."):
        messages = rag_engine.build_messages_text_only(
            prompt=prompt,
            system=PATIENT_SYSTEM_PROMPT,
        )
        raw_answer = rag_engine.generate(
            processor, model, messages, max_new_tokens=300
        )
        answer = postprocess_patient_output(raw_answer)

    st.markdown("### 💊 Explanation")
    st.markdown(answer)
    _add_patient_note(f"Medication: {med}", answer)


# ============================================================
# CHRONIC CONDITION COMPANION TAB
# ============================================================

def chronic_condition_tab(processor, model):
    st.markdown("### ❤️ Chronic Condition Companion")

    condition = st.text_input("Enter a chronic condition (e.g., diabetes, asthma)")

    if not condition:
        return

    prompt = (
        f"Explain the chronic condition '{condition}' in simple language. "
        "Focus on what it is, common symptoms, and general self-management education. "
        "Do NOT give diagnosis, treatment plans, or specific medical advice."
    )

    with st.spinner("Preparing explanation..."):
        messages = rag_engine.build_messages_text_only(
            prompt=prompt,
            system=PATIENT_SYSTEM_PROMPT,
        )
        raw_answer = rag_engine.generate(
            processor, model, messages, max_new_tokens=350
        )
        answer = postprocess_patient_output(raw_answer)

    st.markdown("### ❤️ Understanding Your Condition")
    st.markdown(answer)
    _add_patient_note(f"Condition: {condition}", answer)


# ============================================================
# DOCTOR VISIT PREP TAB
# ============================================================

def doctor_visit_prep_tab(processor, model):
    st.markdown("### 🧑‍⚕️ Doctor Visit Prep")

    topic = st.text_input("What is your upcoming appointment about?")

    if not topic:
        return

    prompt = (
        f"Help the patient prepare for a doctor visit about '{topic}'. "
        "Provide questions they might ask, information they might bring, and topics they might discuss. "
        "Do NOT give diagnosis or treatment recommendations."
    )

    with st.spinner("Preparing guidance..."):
        messages = rag_engine.build_messages_text_only(
            prompt=prompt,
            system=PATIENT_SYSTEM_PROMPT,
        )
        raw_answer = rag_engine.generate(
            processor, model, messages, max_new_tokens=300
        )
        answer = postprocess_patient_output(raw_answer)

    st.markdown("### 🧑‍⚕️ Preparation Guide")
    st.markdown(answer)
    _add_patient_note(f"Doctor visit: {topic}", answer)


# ============================================================
# EXPORT TAB (patient-only notes)
# ============================================================

def export_tab():
    st.markdown("### 📤 Export My Health Notes")
    _init_patient_notes()

    notes = st.session_state.patient_notes

    if not notes:
        st.info("No notes to export yet.")
        return

    content = "MEDGEMMA PATIENT NOTES EXPORT\n\n"
    for n in notes:
        content += (
            f"Date: {n['created_at']}\n"
            f"Q: {n['question']}\n"
            f"A: {n['answer']}\n\n"
        )

    st.download_button(
        label="📥 Download Notes",
        data=content,
        file_name=f"patient_notes_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
    )

def return_to_main_tab():
    st.query_params["page"] = "main"
    st.rerun()

# ============================================================
# MAIN
# ============================================================

def main():
    st.title("🧑‍⚕️ MedGemma Patient & Caregiver Assistant")
    st.markdown("*Clear, simple explanations for patients and caregivers*")

    embedder, coll, processor, model = load_patient_models()

    tab_chat, tab_notes, tab_report, tab_meds, tab_chronic, tab_visit, tab_export, tab_main = st.tabs(
        [
            "💬 Chat",
            "📘 My Health Notes",
            "🧾 Report & Image Simplifier",
            "💊 Medication Helper",
            "❤️ Chronic Condition Companion",
            "🧑‍⚕️ Doctor Visit Prep",
            "📤 Export",
            "🏠 Return to Main Page",
        ]
    )

    with tab_chat:
        patient_chat_tab(processor, model)

    with tab_notes:
        patient_notes_tab()

    with tab_report:
        report_simplifier_tab(embedder, coll, processor, model)

    with tab_meds:
        medication_helper_tab(processor, model)

    with tab_chronic:
        chronic_condition_tab(processor, model)

    with tab_visit:
        doctor_visit_prep_tab(processor, model)

    with tab_export:
        export_tab()
    
    with tab_main:
        if st.button("Return to Main Page"):
            return_to_main_tab()


    

if __name__ == "__main__":
    main()

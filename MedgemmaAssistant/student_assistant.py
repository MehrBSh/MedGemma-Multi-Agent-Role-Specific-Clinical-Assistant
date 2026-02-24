# ============================================================
# Streamlit web UI: Chat (RAG + image, memory, citations, save),
# Flashcards, Quiz, My notes, Export.
# ============================================================

import os
import re
from datetime import datetime
from pathlib import Path

# Small Language Model for conversation
class SmallLanguageModel:
    """Lightweight orchestrator + conversational model using Qwen2.5-0.5B-Instruct"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = None
        self.model_loaded = False

        # Fallback dictionary responses (unchanged)
        self.conversation_responses = {
            "hi": "👋 Hello! How can I help you with your medical learning today?",
            "hello": "👋 Hi there! What medical topic would you like to explore?",
            "hey": "👋 Hey! Ready to learn something medical today?",
            "good morning": "🌅 Good morning! What medical questions do you have?",
            "good afternoon": "🌞 Good afternoon! How can I assist with your studies?",
            "good evening": "🌆 Good evening! What would you like to learn?",
            "how are you": "😊 I'm doing great and ready to help with medical questions! How about you?",
            "what's up": "👋 Just here and ready to help with medical learning! What's on your mind?",
            "how are you doing": "😊 I'm excellent and excited to assist with medical knowledge!",
            "thanks": "😊 You're welcome! Any other medical questions?",
            "thank you": "😊 You're very welcome! Feel free to ask more medical questions.",
            "awesome": "🎉 Glad I could help! What else would you like to know?",
            "great": "👍 Happy to help! Any other medical topics?",
            "bye": "👋 Goodbye! Come back anytime for medical learning assistance!",
            "goodbye": "👋 Take care! Feel free to return for more medical help.",
            "see you": "👋 See you soon! Good luck with your medical studies!",
            "ok": "👍 Got it! What medical question can I help with?",
            "okay": "👍 Okay! Ready for your medical questions!",
            "cool": "😎 Cool! What medical topic interests you?",
            "nice": "😊 Nice! What would you like to learn about?",
        }

        print("📝 SmallLanguageModel (Qwen2.5‑0.5B) initialized — lazy loading enabled")

    def _load_model_if_needed(self):
        if self.model_loaded:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔧 Loading Qwen2.5‑0.5B‑Instruct from local directory on {self.device}")

            model_name = r"..\LLMS\Qwen_Qwen2.5-0.5B-Instruct"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                device=0 if self.device == "cuda" else -1
            )

            self.model_loaded = True
            print("✅ Qwen2.5‑0.5B‑Instruct loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load Qwen2.5‑0.5B‑Instruct: {e}")
            print("➡️ Falling back to dictionary responses only")
            self.model_loaded = False

    def generate_response(self, user_input: str) -> str:
        """Generate conversational response using Qwen or fallback dictionary."""
        print(f"🤖 [ORCHESTRATOR] Processing: '{user_input}'")

        # Load model lazily
        self._load_model_if_needed()

        # If model failed → fallback dictionary
        if not self.model_loaded:
            return self._get_dictionary_response(user_input)

        try:
            # Qwen prompt style
            prompt = f"You are a friendly medical learning assistant for casual conversation only.\nUser: {user_input}\nAssistant:"

            out = self.pipeline(
                prompt,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            text = out[0]["generated_text"]

            # Remove prompt echo
            if prompt in text:
                text = text.replace(prompt, "").strip()

            # Safety: keep it short
            return text[:250]

        except Exception as e:
            print(f"⚠️ Qwen error: {e}")
            return self._get_dictionary_response(user_input)

    def _get_dictionary_response(self, user_input):
        """Fallback dictionary response system."""
        text = user_input.lower().strip()

        # Direct match
        if text in self.conversation_responses:
            return self.conversation_responses[text]

        # Pattern match
        for key, resp in self.conversation_responses.items():
            if key in text:
                return resp

        # Generic fallback
        return "😊 I'm here for medical learning and friendly conversation!"


# Initialize small language model
small_model = SmallLanguageModel()

def is_conversational_input(user_input):
    """Detect if input is conversational vs medical"""
    input_lower = user_input.lower().strip()
    
    # Strong medical indicators - these ALWAYS go to MedGemma
    strong_medical_words = [
        "symptom", "disease", "medicine", "drug", "treatment", "diagnosis", "blood pressure",
        "diabetes", "heart", "cancer", "pain", "fever", "headache", "medication",
        "therapy", "surgery", "test", "analysis", "x-ray", "mri", "ecg", "cause",
        "prevent", "cure", "side effect", "dosage", "prescription", "vaccine",
        "infection", "virus", "bacteria", "antibiotic", "inflammation", "stroke",
        "asthma", "arthritis", "fracture", "tumor", "diagnosis", "prognosis"
    ]
    
    # Weak medical indicators - might be conversational context
    weak_medical_words = [
        "help", "learn", "study", "question", "understand", "explain", "tell me"
    ]
    
    # Strong conversational indicators - these ALWAYS go to small model
    strong_conversational = [
        "hi", "hello", "hey", "how are", "what's up", "doing", "thanks", "thank", 
        "awesome", "great", "cool", "nice", "bye", "goodbye", "see you",
        "good morning", "good afternoon", "good evening", "ok", "okay", "yeah", "yes",
        "no", "maybe", "sure", "alright", "fine", "good", "bad", "well"
    ]
    
    # Priority 1: Strong medical words -> Use MedGemma
    if any(word in input_lower for word in strong_medical_words):
        return False
    
    # Priority 2: Strong conversational -> Use Small Model
    if any(word in input_lower for word in strong_conversational):
        return True
    
    # Priority 3: Short inputs (< 6 chars) -> Likely conversational
    if len(input_lower) < 6:
        return True
    
    # Priority 4: Check for question patterns
    question_words = ["what", "how", "why", "when", "where", "who", "which"]
    if any(word in input_lower for word in question_words):
        # If it's a question about medical topics, use MedGemma
        if any(word in input_lower for word in weak_medical_words):
            return False
        # Otherwise conversational question
        return True
    
    # Default: Assume medical for longer, complex inputs
    return False

def orchestrator_route(user_text: str, uploaded_files):
    """Decide which agent should answer."""
    text = (user_text or "").lower().strip()

    # 1. Image → Vision agent
    if uploaded_files:
        return "vision"

    # 2. Strong medical indicators → MedGemma
    medical_keywords = [
        "symptom", "diagnosis", "treatment", "disease", "drug", "medicine",
        "side effect", "dose", "dosage", "infection", "fracture", "scan", "x-ray",
        "mri", "ct", "ecg", "lab", "blood", "cancer", "asthma", "arthritis",
        "fever", "pain", "tumor", "prognosis", "therapy", "surgery", "vaccine"
    ]
    if any(w in text for w in medical_keywords):
        return "medical"

    # 3. Otherwise → small model (conversation)
    return "conversation"

import streamlit as st
from PIL import Image

import rag_engine
import store
import flashcards
import quiz
import export_notes

store.init_db()

CHROMA_DIR = rag_engine.CHROMA_DIR


@st.cache_resource
def load_models():
    if not os.path.isdir(CHROMA_DIR):
        raise FileNotFoundError(f"Run build_rag_index.py first. Missing {CHROMA_DIR}")
    embedder, coll = rag_engine.load_retriever()
    processor, model = rag_engine.load_medgemma()
    return embedder, coll, processor, model

def chat_tab(embedder, coll, processor, model):

    if "conversation" not in st.session_state:
        st.session_state.conversation = [
            {
                "question": None,
                "answer": (
                    "👋 Hello! I'm your MedGemma Learning Assistant.\n\n"
                    "Ask medical questions or attach an image for analysis.\n\n"
                    "What would you like to learn today?"
                ),
            }
        ]

    if "last_question" not in st.session_state:
        st.session_state.last_question = None
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

    st.markdown("### 💬 Medical Q&A Chat")

    # Display conversation
    for h in st.session_state.conversation:
        if h["question"]:
            with st.chat_message("user"):
                st.markdown(h["question"])

        with st.chat_message("assistant"):
            st.markdown(h["answer"])

    # ==========================================================
    # ====== ✅ NEW: ChatGPT-style single input box ============
    # ==========================================================

    prompt = st.chat_input(
        "Ask me anything about medicine...",
        accept_file=True,
        file_type=["png", "jpg", "jpeg", "webp"]
    )

    # ==========================================================
    # Process input (Agentic routing)
    # ==========================================================

    if prompt:

        user_text = prompt.text
        uploaded_files = prompt.files

        if not user_text and not uploaded_files:
            st.warning("⚠️ Please enter a question or attach an image.")
            return

        route = orchestrator_route(user_text, uploaded_files)
        print(f"🔍 [ROUTING] user_text='{user_text}' | files={bool(uploaded_files)} | route={route}")

        # --- Conversation agent (small model) ---
        if route == "conversation":
            with st.chat_message("user"):
                st.markdown(user_text)

            answer = small_model.generate_response(user_text)

            st.session_state.conversation.append({
                "question": user_text,
                "answer": answer
            })
            st.session_state.last_question = user_text
            st.session_state.last_answer = answer
            st.session_state.last_sources = []
            st.rerun()

        # --- Vision agent (MedGemma-Vision) ---
        elif route == "vision":
            file = uploaded_files[0]

            with st.chat_message("user"):
                st.markdown(f"📎 {file.name}")

            with st.spinner("Analyzing image..."):
                image = Image.open(file).convert("RGB")
                question = "Please analyze this medical image and describe the findings."

                answer, sources = rag_engine.ask_image(
                    image, question, embedder, coll, processor, model
                )

            st.session_state.conversation.append({
                "question": f"[Image: {file.name}] {question}",
                "answer": answer
            })

            st.session_state.last_question = question
            st.session_state.last_answer = answer
            st.session_state.last_sources = sources or []
            st.rerun()

        # --- Medical text agent (MedGemma-Text) ---
        elif route == "medical":
            with st.chat_message("user"):
                st.markdown(user_text)

            history = [
                {"question": h["question"], "answer": h["answer"]}
                for h in st.session_state.conversation[-8:]
                if h["question"]
            ]

            with st.spinner("Thinking..."):
                answer, sources = rag_engine.ask_text(
                    user_text,
                    embedder,
                    coll,
                    processor,
                    model,
                    conversation_history=history
                )

            st.session_state.conversation.append({
                "question": user_text,
                "answer": answer
            })

            st.session_state.last_question = user_text
            st.session_state.last_answer = answer
            st.session_state.last_sources = sources or []
            st.rerun()

    # ==========================================================
    # Action Buttons
    # ==========================================================

    if st.session_state.last_answer:

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💾 Save to Notes"):
                store.save_note(
                    st.session_state.last_question,
                    st.session_state.last_answer,
                    st.session_state.last_sources
                )
                st.success("Saved!")

        with col2:
            if st.button("💬 Save Conversation"):
                conversation_to_save = [
                    h for h in st.session_state.conversation
                    if h["question"]
                ]
                if conversation_to_save:
                    store.save_conversation(conversation_to_save)
                    st.success("Conversation saved!")

        with col3:
            if st.button("🗑️ Clear Chat"):
                st.session_state.conversation = [
                    {
                        "question": None,
                        "answer": "👋 Hello! I'm your MedGemma Learning Assistant."
                    }
                ]
                st.session_state.last_question = None
                st.session_state.last_answer = None
                st.session_state.last_sources = []
                st.rerun()

def flashcards_tab(embedder, coll, processor, model):
    st.markdown("### 🎴 Flashcards")
    st.caption("Generate and review medical flashcards for effective learning")
    
    # Better layout for flashcard modes
    col1, col2 = st.columns([1, 1])
    with col1:
        sub = st.radio("🎯 Choose Mode", ["🆕 Generate New", "📚 Review Deck"], horizontal=True)
    
    if sub == "🆕 Generate New":
        st.markdown("---")
        st.markdown("#### 📝 Create Flashcards")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            topic = st.text_input("🔍 Enter Topic", placeholder="e.g., corticosteroids and ARDS", help="Be specific for better flashcards")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            from_last = st.checkbox("📋 Use Last Chat Q&A", value=False, help="Generate from your recent conversation")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("🎴 Generate", type="primary", disabled=not (topic or from_last)):
                with st.spinner("🔄 Generating flashcards..."):
                    if from_last and "last_question" in st.session_state and st.session_state.get("last_question"):
                        cards = flashcards.generate_flashcards_from_qa(
                            st.session_state.last_question, st.session_state.get("last_answer", ""), processor, model
                        )
                        topic_label = "last_qa"
                    elif topic:
                        cards = flashcards.generate_flashcards_from_topic(topic, embedder, coll, processor, model)
                        topic_label = topic
                    else:
                        st.warning("⚠️ Please enter a topic or use last Q&A after chatting.")
                        return
                    
                    if not cards:
                        st.warning("⚠️ No flashcards generated. Try a different topic.")
                        return
                    
                    n = flashcards.save_generated_flashcards(cards, topic=topic_label)
                    st.success(f"✅ Generated and saved {n} flashcards!")
                    st.balloons()
        
        with col2:
            if st.button("📊 View All", type="secondary"):
                all_cards = flashcards.get_all_cards()
                if all_cards:
                    st.info(f"📚 You have {len(all_cards)} total flashcards")
                else:
                    st.info("📚 No flashcards yet")
        
        with col3:
            if st.button("🗑️ Clear All", type="secondary"):
                st.warning("⚠️ This will delete all flashcards. Feature coming soon!")
        
        st.caption("💡 Tip: Generated cards appear in the Review deck automatically")
    
    else:  # Review Deck
        st.markdown("---")
        st.markdown("#### 📚 Review Your Flashcards")
        
        cards = flashcards.get_review_deck(limit=30)
        if not cards:
            st.info("📚 No flashcards to review. Generate some first!")
            return
        
        # Initialize session state for flashcard navigation
        if "fc_index" not in st.session_state:
            st.session_state.fc_index = 0
        if "fc_show_back" not in st.session_state:
            st.session_state.fc_show_back = False
        if "fc_studied" not in st.session_state:
            st.session_state.fc_studied = []

        idx = st.session_state.fc_index % len(cards)
        c = cards[idx]
        
        # Progress bar
        progress = (st.session_state.fc_index + 1) / len(cards)
        st.progress(progress)
        st.caption(f"📊 Progress: {st.session_state.fc_index + 1} of {len(cards)} cards")
        
        # Flashcard design - make it look like a real flashcard
        st.markdown("---")
        
        # Create a card-like container
        with st.container():
            # Card styling
            st.markdown("""
            <style>
            .flashcard {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                margin: 1rem 0;
                min-height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                transition: transform 0.3s ease;
            }
            .flashcard:hover {
                transform: translateY(-5px);
            }
            .card-label {
                font-size: 0.9rem;
                opacity: 0.8;
                margin-bottom: 1rem;
                font-weight: 600;
            }
            .card-content {
                font-size: 1.2rem;
                font-weight: 500;
                line-height: 1.6;
            }
            </style>
            """, unsafe_allow_html=True)
            
            if not st.session_state.fc_show_back:
                # Front of card
                st.markdown(f"""
                <div class="flashcard">
                    <div class="card-label">🎴 QUESTION</div>
                    <div class="card-content">{c['front']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Navigation buttons
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("⬅️ Previous", disabled=st.session_state.fc_index == 0):
                        st.session_state.fc_index = max(0, st.session_state.fc_index - 1)
                        st.session_state.fc_show_back = False
                        st.rerun()
                
                with col2:
                    if st.button("🔍 Show Answer", type="primary"):
                        st.session_state.fc_show_back = True
                        st.rerun()
                
                with col3:
                    if st.button("➡️ Next", disabled=False):
                        st.session_state.fc_index = (st.session_state.fc_index + 1) % len(cards)
                        st.session_state.fc_show_back = False
                        st.rerun()
            
            else:
                # Back of card
                st.markdown(f"""
                <div class="flashcard" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <div class="card-label">💡 ANSWER</div>
                    <div class="card-content">{c['back']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Self-assessment buttons
                st.markdown("#### 📊 How well did you know this?")
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                
                with col1:
                    if st.button("😕 Hard", type="secondary"):
                        st.session_state.fc_studied.append({"card": c, "difficulty": "hard"})
                        st.session_state.fc_index = (st.session_state.fc_index + 1) % len(cards)
                        st.session_state.fc_show_back = False
                        st.rerun()
                
                with col2:
                    if st.button("🤔 Medium", type="secondary"):
                        st.session_state.fc_studied.append({"card": c, "difficulty": "medium"})
                        st.session_state.fc_index = (st.session_state.fc_index + 1) % len(cards)
                        st.session_state.fc_show_back = False
                        st.rerun()
                
                with col3:
                    if st.button("😊 Easy", type="secondary"):
                        st.session_state.fc_studied.append({"card": c, "difficulty": "easy"})
                        st.session_state.fc_index = (st.session_state.fc_index + 1) % len(cards)
                        st.session_state.fc_show_back = False
                        st.rerun()
                
                with col4:
                    if st.button("➡️ Skip", type="secondary"):
                        st.session_state.fc_index = (st.session_state.fc_index + 1) % len(cards)
                        st.session_state.fc_show_back = False
                        st.rerun()
        
        # Study session stats
        if st.session_state.fc_studied:
            st.markdown("---")
            st.markdown("#### 📈 Study Session Stats")
            studied_count = len(st.session_state.fc_studied)
            easy_count = sum(1 for s in st.session_state.fc_studied if s["difficulty"] == "easy")
            medium_count = sum(1 for s in st.session_state.fc_studied if s["difficulty"] == "medium")
            hard_count = sum(1 for s in st.session_state.fc_studied if s["difficulty"] == "hard")
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                st.metric("📚 Studied", studied_count)
            with col2:
                st.metric("😊 Easy", easy_count, delta=None)
            with col3:
                st.metric("🤔 Medium", medium_count, delta=None)
            with col4:
                st.metric("😕 Hard", hard_count, delta=None)


def quiz_tab(embedder, coll, processor, model):
    st.markdown("### 📝 Quiz")
    st.caption("Test your knowledge with multiple choice and written answers")
    
    # Quiz mode selection
    col1, col2 = st.columns([1, 1])
    with col1:
        quiz_mode = st.radio("🎯 Quiz Type", ["🔢 Multiple Choice", "✍️ Written Answers"], horizontal=True)
    
    st.markdown("---")
    
    if quiz_mode == "🔢 Multiple Choice":
        st.markdown("#### 📋 Multiple Choice Quiz")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            topic = st.text_input("🔍 Enter Topic", placeholder="e.g., corticosteroids and respiratory disease", value="corticosteroids and respiratory disease")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            num_questions = st.selectbox("📊 Number of Questions", [3, 5, 7, 10], index=0)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚀 Start Quiz", type="primary", disabled=not topic):
                with st.spinner("🔄 Generating quiz questions..."):
                    questions = quiz.generate_quiz(topic, embedder, coll, processor, model, num_questions=num_questions)
                    if not questions:
                        st.warning("⚠️ No questions generated. Try another topic.")
                        return
                    
                    st.session_state.quiz = questions
                    st.session_state.quiz_idx = 0
                    st.session_state.quiz_done = False
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answers = []
                    st.success(f"✅ Quiz generated with {len(questions)} questions!")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Reset Quiz", type="secondary"):
                if "quiz" in st.session_state:
                    del st.session_state.quiz
                    del st.session_state.quiz_idx
                    del st.session_state.quiz_done
                    del st.session_state.quiz_score
                    del st.session_state.quiz_answers
                st.rerun()
        
        # Multiple choice quiz interface
        if "quiz" not in st.session_state or not st.session_state.quiz:
            st.info("📚 Start a quiz to test your knowledge!")
            return
        
        qs = st.session_state.quiz
        idx = st.session_state.quiz_idx
        
        if idx >= len(qs):
            # Quiz completed
            st.markdown("---")
            st.markdown("#### 🎉 Quiz Completed!")
            score = st.session_state.quiz_score
            total = len(qs)
            percentage = (score / total) * 100
            
            # Performance visualization
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.metric("📊 Score", f"{score}/{total}")
            with col2:
                st.metric("📈 Percentage", f"{percentage:.1f}%")
            with col3:
                if percentage >= 80:
                    st.metric("🏆 Grade", "Excellent")
                elif percentage >= 60:
                    st.metric("👍 Grade", "Good")
                else:
                    st.metric("📚 Grade", "Keep Practicing")
            
            # Progress bar
            st.progress(score / total)
            
            # Review answers
            st.markdown("#### 📝 Review Your Answers")
            for i, (q, user_answer) in enumerate(zip(qs, st.session_state.quiz_answers)):
                with st.expander(f"Question {i+1}: {q['question'][:50]}..."):
                    st.markdown(f"**Question:** {q['question']}")
                    st.markdown(f"**Your Answer:** {q['options'][user_answer] if user_answer != -1 else '[Skipped]'}")
                    st.markdown(f"**Correct Answer:** {q['options'][q['correct_index']]}")
                    if user_answer == q['correct_index']:
                        st.success("✅ Correct!")
                    elif user_answer == -1:
                        st.info("⏭️ Skipped")
                    else:
                        st.error("❌ Incorrect")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Try Again", type="primary"):
                    del st.session_state.quiz
                    del st.session_state.quiz_idx
                    del st.session_state.quiz_done
                    del st.session_state.quiz_score
                    del st.session_state.quiz_answers
                    st.rerun()
            with col2:
                if st.button("🎴 Generate Flashcards", type="secondary"):
                    st.info("🔄 Switch to Flashcards tab to generate from this quiz!")
            
            return
        
        # Current question
        q = qs[idx]
        progress = (idx + 1) / len(qs)
        st.progress(progress)
        st.caption(f"📊 Question {idx + 1} of {len(qs)} | Score: {st.session_state.quiz_score}")
        
        # Question display with better styling
        st.markdown("---")
        st.markdown("#### 📖 Question")
        st.markdown(f"**{q['question']}**")
        
        # Answer options
        st.markdown("#### 💭 Choose Your Answer:")
        choice = st.radio("Select one option:", q["options"], key=f"quiz_opt_{idx}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Submit Answer", type="primary"):
                user_idx = q["options"].index(choice)
                correct = user_idx == q["correct_index"]
                
                if correct:
                    st.success("🎉 Correct! Well done!")
                    st.session_state.quiz_score += 1
                else:
                    st.error(f"❌ Not quite right. The correct answer is: {q['options'][q['correct_index']]}")
                
                # Store user's answer
                st.session_state.quiz_answers.append(user_idx)
                
                # Get explanation
                with st.spinner("🤔 Generating explanation..."):
                    expl = quiz.get_explanation(q["question"], q["options"], q["correct_index"], user_idx, processor, model)
                
                st.markdown("#### 💡 Explanation")
                st.markdown(expl)
                
                st.session_state.quiz_idx = idx + 1
                st.rerun()
        
        with col2:
            if st.button("⏭️ Skip Question", type="secondary"):
                st.session_state.quiz_answers.append(-1)  # -1 indicates skipped
                st.session_state.quiz_idx = idx + 1
                st.rerun()
    
    else:  # Written Answers
        st.markdown("#### ✍️ Written Answer Quiz")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            topic = st.text_input("🔍 Enter Topic", placeholder="e.g., cardiovascular anatomy", value="cardiovascular anatomy")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            num_questions = st.selectbox("📊 Number of Questions", [2, 3, 4, 5], index=0)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚀 Start Written Quiz", type="primary", disabled=not topic):
                with st.spinner("🔄 Generating written questions..."):
                    questions = generate_written_quiz(topic, embedder, coll, processor, model, num_questions=num_questions)
                    if not questions:
                        st.warning("⚠️ No questions generated. Try another topic.")
                        return
                    
                    st.session_state.written_quiz = questions
                    st.session_state.written_quiz_idx = 0
                    st.session_state.written_quiz_done = False
                    st.session_state.written_quiz_answers = []
                    st.success(f"✅ Written quiz generated with {len(questions)} questions!")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Reset Quiz", type="secondary"):
                if "written_quiz" in st.session_state:
                    del st.session_state.written_quiz
                    del st.session_state.written_quiz_idx
                    del st.session_state.written_quiz_done
                    del st.session_state.written_quiz_answers
                st.rerun()
        
        # Written quiz interface
        if "written_quiz" not in st.session_state or not st.session_state.written_quiz:
            st.info("📚 Start a written quiz to test your knowledge!")
            return
        
        qs = st.session_state.written_quiz
        idx = st.session_state.written_quiz_idx
        
        if idx >= len(qs):
            # Quiz completed
            st.markdown("---")
            st.markdown("#### 🎉 Written Quiz Completed!")
            
            st.markdown("#### 📝 Your Answers:")
            for i, (q, user_answer) in enumerate(zip(qs, st.session_state.written_quiz_answers)):
                with st.expander(f"Question {i+1}: {q['question'][:50]}..."):
                    st.markdown(f"**Question:** {q['question']}")
                    st.markdown(f"**Your Answer:** {user_answer}")
                    st.markdown(f"**Expected Answer:** {q['expected_answer']}")
                    
                    # Simple evaluation (you could enhance this with AI evaluation)
                    with st.spinner("🤔 Evaluating your answer..."):
                        evaluation = evaluate_written_answer(user_answer, q['expected_answer'], processor, model)
                    st.markdown(f"**Evaluation:** {evaluation}")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Try Again", type="primary"):
                    del st.session_state.written_quiz
                    del st.session_state.written_quiz_idx
                    del st.session_state.written_quiz_done
                    del st.session_state.written_quiz_answers
                    st.rerun()
            with col2:
                if st.button("🎴 Generate Flashcards", type="secondary"):
                    st.info("🔄 Switch to Flashcards tab to generate from this quiz!")
            
            return
        
        # Current question
        q = qs[idx]
        progress = (idx + 1) / len(qs)
        st.progress(progress)
        st.caption(f"📊 Question {idx + 1} of {len(qs)}")
        
        # Question display
        st.markdown("---")
        st.markdown("#### 📖 Question")
        st.markdown(f"**{q['question']}**")
        
        # Answer input
        st.markdown("#### ✍️ Your Answer:")
        user_answer = st.text_area("Write your detailed answer here:", 
                                   placeholder="Provide a comprehensive answer based on your knowledge...",
                                   key=f"written_answer_{idx}",
                                   height=150)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Submit Answer", type="primary", disabled=not user_answer.strip()):
                st.session_state.written_quiz_answers.append(user_answer.strip())
                st.session_state.written_quiz_idx = idx + 1
                st.success("✅ Answer submitted!")
                st.rerun()
        
        with col2:
            if st.button("⏭️ Skip Question", type="secondary"):
                st.session_state.written_quiz_answers.append("[Skipped]")
                st.session_state.written_quiz_idx = idx + 1
                st.rerun()


def generate_written_quiz(topic: str, embedder, coll, processor, model, num_questions: int = 3) -> list[dict]:
    """Generate written answer questions."""
    chunks = rag_engine.retrieve(embedder, coll, topic, top_k=5)
    if not chunks:
        return []
    context = "\n\n".join(chunks[:3])
    
    prompt = f"""Using only the context below, create exactly {num_questions} written answer questions. These should require detailed explanations, not simple one-word answers.

Format each question as:
Q1: [question text]
Expected: [brief expected answer key points]

Q2: [question text]
Expected: [brief expected answer key points]

Context:
{context}

Questions:"""
    
    messages = rag_engine.build_messages_text_only(
        prompt,
        system="You output only the numbered questions in the exact format above. Questions should require thoughtful, detailed responses.",
    )
    reply = rag_engine.generate(processor, model, messages, max_new_tokens=600)
    return _parse_written_quiz(reply)


def _parse_written_quiz(text: str) -> list[dict]:
    """Parse written quiz questions."""
    questions = []
    # Split by Q1:, Q2:, Q3: or similar
    blocks = re.split(r"\n\s*Q\d+:\s*", text, flags=re.IGNORECASE)[1:]
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) >= 2:
            question_text = lines[0]
            expected_answer = ""
            # Look for "Expected:" line
            for line in lines[1:]:
                if line.lower().startswith("expected:"):
                    expected_answer = line.replace("Expected:", "").strip()
                    break
                elif not expected_answer:  # If no Expected: found, use next line
                    expected_answer = line
                    break
            
            if question_text and expected_answer:
                questions.append({
                    "question": question_text,
                    "expected_answer": expected_answer,
                })
    return questions[:5]


def evaluate_written_answer(user_answer: str, expected_answer: str, processor, model) -> str:
    """Evaluate a written answer using AI."""
    prompt = f"""Evaluate this student's answer compared to the expected answer. Be encouraging but constructive.

Expected Answer: {expected_answer}
Student Answer: {user_answer}

Provide a brief evaluation (2-3 sentences) highlighting what they got right and what could be improved."""
    
    messages = rag_engine.build_messages_text_only(prompt)
    return rag_engine.generate(processor, model, messages, max_new_tokens=200)


def conversations_tab():
    st.markdown("### 💬 Saved Conversations")
    st.caption("Review and manage your complete chat history")
    
    conversations = store.get_all_conversations()
    
    if not conversations:
        st.info("📚 No saved conversations yet. Use the Chat tab and click 'Save Conversation' to save your learning sessions.")
        return
    
    # Statistics
    st.markdown("#### 📊 Your Learning Stats")
    total_conversations = len(conversations)
    total_messages = sum(len(conv["conversation"]) for conv in conversations)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("💬 Conversations", total_conversations)
    with col2:
        st.metric("📝 Total Messages", total_messages)
    with col3:
        avg_messages = total_messages // total_conversations if total_conversations > 0 else 0
        st.metric("📈 Avg Messages", avg_messages)
    
    st.markdown("---")
    st.markdown("#### 📚 Your Conversations")
    
    # Search and filter
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("🔍 Search conversations...", placeholder="Search by title or content...")
    with col2:
        sort_by = st.selectbox("📊 Sort by", ["Most Recent", "Oldest", "Title A-Z"])
    
    # Filter conversations based on search
    filtered_conversations = conversations
    if search_term:
        filtered_conversations = [
            conv for conv in conversations 
            if search_term.lower() in conv["title"].lower() or
               any(search_term.lower() in str(item).lower() for item in conv["conversation"])
        ]
    
    # Sort conversations
    if sort_by == "Most Recent":
        filtered_conversations.sort(key=lambda x: x["updated_at"], reverse=True)
    elif sort_by == "Oldest":
        filtered_conversations.sort(key=lambda x: x["updated_at"])
    elif sort_by == "Title A-Z":
        filtered_conversations.sort(key=lambda x: x["title"].lower())
    
    if not filtered_conversations:
        st.warning("🔍 No conversations found matching your search.")
        return
    
    # Display conversations
    for conv in filtered_conversations:
        with st.expander(f"📝 {conv['title']}", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.caption(f"📅 Created: {conv['created_at'][:10]} | Updated: {conv['updated_at'][:10]}")
                st.caption(f"💬 {len(conv['conversation'])} messages")
            
            with col2:
                if st.button("🔄 Load Chat", key=f"load_{conv['id']}"):
                    # Load conversation into current session
                    welcome_msg = {
                        "question": None,
                        "answer": (
                            "👋 Hello! I'm your MedGemma Learning Assistant, here to help you master medical topics.\n\n"
                            "I can help you:\n"
                            "• 🧠 Answer medical questions with evidence-based information\n"
                            "• 🔍 Analyze medical images and explain findings\n"
                            "• 📚 Provide sources for further reading\n"
                            "• 💡 Generate study materials from our conversations\n\n"
                            "Feel free to ask me anything about medicine, upload an image for analysis, "
                            "or let's explore a topic together. What would you like to learn about today?"
                        ),
                    }
                    st.session_state.conversation = [welcome_msg] + conv["conversation"]
                    st.success("✅ Conversation loaded! Switch to Chat tab to continue.")
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{conv['id']}"):
                    if store.delete_conversation(conv['id']):
                        st.success("✅ Conversation deleted!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete conversation.")
            
            # Display conversation preview
            st.markdown("**Conversation Preview:**")
            for i, item in enumerate(conv["conversation"][:3]):  # Show first 3 messages
                if item["question"]:
                    st.markdown(f"**🧑‍⚕️ Q{i+1}:** {item['question']}")
                st.markdown(f"**🤖 A{i+1}:** {item['answer'][:200]}{'...' if len(item['answer']) > 200 else ''}")
            
            if len(conv["conversation"]) > 3:
                st.caption(f"... and {len(conv['conversation']) - 3} more messages")


def notes_tab():
    st.markdown("### 📚 My Notes")
    st.caption("Quick reference for important Q&A pairs")
    
    notes = store.get_all_notes()
    if not notes:
        st.info("📚 No saved notes. Use Chat and click 'Save to Notes' to save important Q&A pairs.")
        return
    
    # Statistics
    st.markdown("#### 📊 Notes Overview")
    total_notes = len(notes)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("📝 Total Notes", total_notes)
    with col2:
        if total_notes > 0:
            latest_date = notes[0]["created_at"][:10]
            st.metric("📅 Latest Note", latest_date)
    
    st.markdown("---")
    
    # Search notes
    search_term = st.text_input("🔍 Search notes...", placeholder="Search by question or answer...")
    
    # Filter notes
    filtered_notes = notes
    if search_term:
        filtered_notes = [
            note for note in notes 
            if search_term.lower() in note["question"].lower() or
               search_term.lower() in note["answer"].lower()
        ]
    
    if not filtered_notes:
        st.warning("🔍 No notes found matching your search.")
        return
    
    # Display notes
    for n in filtered_notes:
        with st.expander(f"❓ {n['question'][:80]}{'...' if len(n['question']) > 80 else ''}", expanded=False):
            st.markdown(f"**📅 Date:** {n['created_at'][:10]}")
            st.markdown(f"**❓ Question:** {n['question']}")
            st.markdown(f"**💡 Answer:** {n['answer']}")
            
            if n.get("sources"):
                st.markdown("**📚 Sources:**")
                for i, s in enumerate(n["sources"][:2], 1):
                    st.caption(f"{i}. {s[:100]}{'...' if len(s) > 100 else ''}")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🎴 Generate Flashcards", key=f"note_fc_{n['id']}"):
                    st.info("🔄 Switch to Flashcards tab to generate from this note!")
            with col2:
                if st.button("🗑️ Delete", key=f"note_del_{n['id']}"):
                    # Add delete functionality if needed
                    st.warning("⚠️ Delete functionality coming soon!")


def export_tab():
    st.markdown("### 📤 Export Your Learning Data")
    st.caption("Download your notes and conversations for offline study")
    
    # Get data
    notes = store.get_all_notes()
    conversations = store.get_all_conversations()
    
    if not notes and not conversations:
        st.info("📚 No data to export. Start chatting and saving your learning sessions!")
        return
    
    # Statistics
    st.markdown("#### 📊 Export Overview")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("📝 Notes Available", len(notes))
    with col2:
        st.metric("💬 Conversations Available", len(conversations))
    
    st.markdown("---")
    
    # Export options
    st.markdown("#### 🎯 Choose What to Export")
    
    export_format = st.selectbox("📄 Format", ["Markdown (.md)", "Plain Text (.txt)", "JSON (.json)"])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        export_notes_data = st.checkbox("📝 Export Notes", value=True)
    with col2:
        export_conversations_data = st.checkbox("💬 Export Conversations", value=True)
    
    if not export_notes_data and not export_conversations_data:
        st.warning("⚠️ Please select at least one type of data to export.")
        return
    
    if st.button("📥 Generate Export", type="primary"):
        with st.spinner("🔄 Preparing your export..."):
            if export_format == "Markdown (.md)":
                content = generate_markdown_export(notes if export_notes_data else [], 
                                                conversations if export_conversations_data else [])
                filename = f"medgemma_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                mime_type = "text/markdown"
            elif export_format == "Plain Text (.txt)":
                content = generate_text_export(notes if export_notes_data else [], 
                                            conversations if export_conversations_data else [])
                filename = f"medgemma_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                mime_type = "text/plain"
            else:  # JSON
                content = generate_json_export(notes if export_notes_data else [], 
                                             conversations if export_conversations_data else [])
                filename = f"medgemma_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                mime_type = "application/json"
            
            st.success(f"✅ Export ready! {len(content)} characters")
            
            # Download button
            st.download_button(
                label=f"📥 Download {filename}",
                data=content,
                file_name=filename,
                mime=mime_type
            )
            
            # Preview
            with st.expander("👁️ Preview Export", expanded=False):
                st.text(content[:1000] + "..." if len(content) > 1000 else content)


def generate_markdown_export(notes: list[dict], conversations: list[dict]) -> str:
    """Generate markdown export."""
    content = f"# MedGemma Learning Assistant Export\n\n"
    content += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    content += f"**Notes:** {len(notes)} | **Conversations:** {len(conversations)}\n\n"
    content += "---\n\n"
    
    if notes:
        content += "## 📝 Notes\n\n"
        for i, note in enumerate(notes, 1):
            content += f"### Note {i}: {note['question'][:50]}{'...' if len(note['question']) > 50 else ''}\n\n"
            content += f"**Date:** {note['created_at'][:10]}\n\n"
            content += f"**Question:** {note['question']}\n\n"
            content += f"**Answer:** {note['answer']}\n\n"
            
            if note.get("sources"):
                content += "**Sources:**\n"
                for j, source in enumerate(note["sources"][:3], 1):
                    content += f"{j}. {source}\n"
                content += "\n"
            
            content += "---\n\n"
    
    if conversations:
        content += "## 💬 Conversations\n\n"
        for i, conv in enumerate(conversations, 1):
            content += f"### Conversation {i}: {conv['title']}\n\n"
            content += f"**Created:** {conv['created_at'][:10]} | **Updated:** {conv['updated_at'][:10]}\n\n"
            content += f"**Messages:** {len(conv['conversation'])}\n\n"
            
            for j, item in enumerate(conv['conversation'], 1):
                if item['question']:
                    content += f"**Q{j}:** {item['question']}\n\n"
                content += f"**A{j}:** {item['answer']}\n\n"
            
            content += "---\n\n"
    
    return content


def generate_text_export(notes: list[dict], conversations: list[dict]) -> str:
    """Generate plain text export."""
    content = f"MEDGEMMA LEARNING ASSISTANT EXPORT\n"
    content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    content += f"Notes: {len(notes)} | Conversations: {len(conversations)}\n"
    content += "=" * 50 + "\n\n"
    
    if notes:
        content += "NOTES\n" + "=" * 30 + "\n\n"
        for i, note in enumerate(notes, 1):
            content += f"Note {i}:\n"
            content += f"Date: {note['created_at'][:10]}\n"
            content += f"Q: {note['question']}\n"
            content += f"A: {note['answer']}\n\n"
    
    if conversations:
        content += "CONVERSATIONS\n" + "=" * 30 + "\n\n"
        for i, conv in enumerate(conversations, 1):
            content += f"Conversation {i}: {conv['title']}\n"
            content += f"Date: {conv['created_at'][:10]}\n\n"
            
            for j, item in enumerate(conv['conversation'], 1):
                if item['question']:
                    content += f"Q{j}: {item['question']}\n"
                content += f"A{j}: {item['answer']}\n\n"
    
    return content


def generate_json_export(notes: list[dict], conversations: list[dict]) -> str:
    """Generate JSON export."""
    import json
    
    export_data = {
        "export_info": {
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "notes_count": len(notes),
            "conversations_count": len(conversations),
            "app": "MedGemma Learning Assistant"
        },
        "notes": notes,
        "conversations": conversations
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def return_to_main_tab():
    st.query_params["page"] = "main"
    st.rerun()





def main():
    st.set_page_config(page_title="MedGemma Learning", layout="wide", initial_sidebar_state="expanded")
    
    # Sidebar with introduction and guide
    with st.sidebar:
        st.title("🩺 MedGemma Learning Assistant")
        st.markdown("---")
        
        with st.expander("📖 Welcome & Guide", expanded=True):
            st.markdown("""
            **Welcome to your personalized medical learning companion!**
            
            ### 🎯 What you can do:
            
            **💬 Chat**
            - Ask medical questions
            - Upload medical images for analysis
            - Get AI-powered responses with sources
            
            **🎴 Flashcards**
            - Generate from topics or your conversations
            - Review with interactive card flipping
            - Spaced repetition learning
            
            **📝 Quiz**
            - Test your knowledge with multiple choice
            - Written answer questions 
            - Detailed explanations
            
            **📚 My Notes**
            - Save important Q&A pairs
            - Review your learning history
            - Export for offline study
            
            ### 🚀 Getting Started:
            1. Start with a chat question
            2. Generate flashcards from topics
            3. Test yourself with quizzes
            4. Save important insights to notes
            """)
        
        st.markdown("---")
        st.caption("Built with MedGemma for medical education")
    
    # Main content area
    st.title("🩺 MedGemma Learning Assistant")
    st.markdown("*Your AI-powered medical learning companion*")

    try:
        embedder, coll, processor, model = load_models()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["💬 Chat", "🎴 Flashcards", "📝 Quiz", "💬 Conversations", "📚 My Notes", "📤 Export", "🏠 Return to Main Page"])
    with tab1:
        chat_tab(embedder, coll, processor, model)
    with tab2:
        flashcards_tab(embedder, coll, processor, model)
    with tab3:
        quiz_tab(embedder, coll, processor, model)
    with tab4:
        conversations_tab()
    with tab5:
        notes_tab()
    with tab6:
        export_tab()
    with tab7:
        if st.button("Return to Main Page"):
            return_to_main_tab()



if __name__ == "__main__":
    main()

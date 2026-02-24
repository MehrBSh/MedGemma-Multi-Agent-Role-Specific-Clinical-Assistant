# 🏥 MedGemma Multi-Agent Role-Specific Clinical Assistant

## 🌟 Project Overview

MedGemma Assistant is a medical AI platform with multiple user interfaces built with Streamlit and powered by local language models. Currently supports student learning, patient health information, and clinical decision support.

## 🚀 Existing Applications

### 1. **Student Assistant** (`student_assistant.py`) ✅
- **📚 Learning Interface**: Medical education platform
- **🗂️ Flashcard System**: AI-generated flashcards from study materials
- **📊 Quiz Engine**: Dynamic question generation with feedback
- **📝 Note Management**: Organized medical notes with tagging
- **📖 Knowledge Base**: RAG-powered medical information retrieval

### 2. **Patient Assistant** (`patient_assistant.py`) ✅
- **💬 Health Chatbot**: Conversational AI for medical questions
- **🔍 Symptom Checker**: Basic AI-powered symptom analysis
- **💊 Medication Info**: Drug information lookup
- **📚 Health Education**: Patient-friendly medical information

### 3. **Doctor Assistant** (`doctor_assistant.py`) ✅
- **📋 Patient Overview**: Comprehensive clinical dashboard
- **🤖 AI Chat / Clinical Assistant**: Context-aware medical Q&A
- **📈 Vitals Trends**: Interactive charts for patient monitoring
- **🚨 Risk Alerts**: Clinical alert system (sepsis, cardiac)
- **🔬 Recent Labs**: Lab results with abnormal value highlighting
- **💾 Database**: Patient data storage with demographics, medications, vitals

### 4. **Application Hub** (`application.py`) ✅
- **🚪 Unified Entry**: Single access point for all user types
- **🎛️ Role Selection**: Choose between Student, Patient, or Doctor
- **🔄 Navigation**: Easy switching between applications

## 🧠 AI Technology Stack

### **Local Language Models** ✅
- **medgemma-4b-it**: Primary medical model
- **Qwen_Qwen2.5-0.5B-Instruct**: General purpose fallback
- **TinyLlama_TinyLlama-1.1B-Chat-v1.0**: Lightweight conversations
- **microsoft_DialoGPT-small**: Basic chat interactions
- **all-MiniLM-L6-v2**: Document embeddings for RAG

### **Supporting Components** ✅
- **rag_engine.py**: RAG implementation with ChromaDB
- **store.py**: Data storage management
- **flashcards.py**: Flashcard generation system
- **quiz.py**: Quiz engine for assessments
- **export_notes.py**: Note export utilities
- **query_rag.py**: RAG query interface
- **build_rag_index.py**: RAG index building
- **download_models.py**: Model download utility

## 📁 Current Project Structure

```
MedgemmaLearningAssistant/
├── 📄 README.md                    # This file
├── 🚪 application.py               # Main application hub
├── 🎓 student_assistant.py          # Student learning platform
├── 🧑‍⚕️ patient_assistant.py          # Patient health interface
├── 👨‍⚕️ doctor_assistant.py          # Clinical decision support
├── 🔧 download_models.py            # Model download utility
├── 🧠 rag_engine.py                # RAG implementation
├── 💾 store.py                     # Data storage management
├── 📝 flashcards.py                # Flashcard generation
├── 📋 quiz.py                      # Quiz engine
├── 📤 export_notes.py              # Note export utilities
├── 🔍 query_rag.py                 # RAG query interface
├── 🏗️ build_rag_index.py           # RAG index building
├── 🗂️ LLMS/                        # all LLMS can be downloaded her
└── 📦 requirements.txt             # Python dependencies
```

## 🛠️ Installation & Setup

### **System Requirements**
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB minimum for models and data


### ** Installation**


### **Model Setup**

**Ensure models are in LLMS directory**:
   ```
   ...\LLMS\
   ├── medgemma-4b-it/
   ├── Qwen_Qwen2.5-0.5B-Instruct/
   ├── TinyLlama_TinyLlama-1.1B-Chat-v1.0/
   ├── microsoft_DialoGPT-small/
   └── all-MiniLM-L6-v2/
   ```

```bash
# Clone repository
git clone https://github.com/MehrBSh/MedGemma-Multi-Agent-Role-Specific-Clinical-Assistant.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Navigate to project directory
cd "....\MedgemmaAssistant"

# Install dependencies
pip install -r requirements.txt

#Download AI models (if needed)
python download_models.py

# Build RAG index**:
python build_rag_index.py


# Run development server
streamlit run application.py --server.port 8501

# or just Launch application hub
streamlit run application.py

# Select your role and start exploring!
```

**Currently Available:**
- 🎓 **Student**: Full learning platform with flashcards and quizzes
- 🧑‍⚕️ **Patient**: Health information and symptom checking
- 👨‍⚕️ **Doctor**: Patient overview and AI clinical chat (partial)



### ** Direct Application Launch**
```bash
# Student Assistant
streamlit run student_assistant.py

# Patient Assistant
streamlit run patient_assistant.py

# Doctor Assistant
streamlit run doctor_assistant.py
```

## 🎯 Current Features

### **Student Assistant** ✅
- Medical learning interface with RAG-powered knowledge base
- AI-generated flashcards from uploaded materials
- Interactive quiz system with immediate feedback
- Note management with tagging and export
- Progress tracking and analytics

### **Patient Assistant** ✅
- Conversational health chatbot
- Basic symptom checking with AI analysis
- Medication information lookup
- Health education content
- User-friendly interface for non-medical users

### **Doctor Assistant** ✅
- Patient overview dashboard with demographics
- Interactive vitals trends visualization
- Recent lab results with abnormal value highlighting
- Risk alert system (sepsis, cardiac events)
- AI-powered clinical chat with patient context
- SQLite database for patient data persistence
- Sample patient data generation for testing

### **Technical Features** ✅
- Local AI model processing (no cloud dependency)
- RAG integration with medical knowledge base
- Multi-user role support through application hub
- SQLite databases for data persistence
- Responsive Streamlit interfaces
- Real-time data visualization with Plotly

## 🔧 Technical Implementation

### **AI Model Integration**
- **Primary**: MedGemma-4B-IT for medical-specific tasks
- **Fallback**: Qwen2.5-0.5B-Instruct when MedGemma unavailable
- **Conversation**: TinyLlama and DialoGPT for casual interactions
- **Embeddings**: all-MiniLM-L6-v2 for document retrieval

### **Database Systems**
- **learning_data.db**: Student progress, flashcards, quiz results
- **doctor_assistant.db**: Patient demographics, vitals, medications, labs, alerts
- **chroma_db**: Vector database for RAG knowledge retrieval

### **Frontend Technologies**
- **Streamlit**: Web application framework
- **Plotly**: Interactive data visualization
- **Pandas**: Data manipulation and analysis
- **SQLite**: Local database storage




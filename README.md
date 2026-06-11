---
title: NUST AI Assistant
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🎓 NUST AI Assistant

RAG-powered chatbot for NUST Pakistan. Documents are pre-loaded by the developer — users just chat.

## 📁 Project Structure

```
nust-chatbot/
├── app.py                  ← Main Streamlit app
├── requirements.txt        ← Dependencies
├── .env.example            ← API keys template
├── documents/              ← PUT YOUR NUST PDFs HERE
│   ├── prospectus.pdf
│   ├── fee_structure.pdf
│   └── handbook.pdf
└── utils/
    ├── pdf_parser.py       ← Reads PDFs from documents/ folder
    ├── vector_store.py     ← FAISS vector database
    └── rag_chain.py        ← Groq LLM + Tavily web search
```

---

## 🖥️ Run Locally — Step by Step

### Step 1 — Get API Keys (Free)

**Groq (Required):**
1. Go to https://console.groq.com
2. Sign up → API Keys → Create API Key
3. Copy the key

**Tavily (Optional):**
1. Go to https://tavily.com
2. Sign up → Copy API key from dashboard

### Step 2 — Setup project

```cmd
cd nust-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Create .env file

Create a file named `.env` in the project folder:
```
GROQ_API_KEY=paste_your_groq_key_here
TAVILY_API_KEY=paste_your_tavily_key_here
```

### Step 4 — Add your NUST documents

Put all your NUST PDF files inside the `documents/` folder:
```
documents/
├── nust_prospectus.pdf
├── fee_structure.pdf
├── student_handbook.pdf
└── scholarship_info.pdf
```

### Step 5 — Run the app

```cmd
streamlit run app.py
```

Open: http://localhost:8501

On first run, app will:
1. Read all PDFs from `documents/` folder
2. Build the vector store automatically
3. Save it as `vector_store/` for faster future loads

---



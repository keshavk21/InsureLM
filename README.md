# 💬 LangChain RAG Chatbot with Gradio & Groq API

This project is an interactive chatbot built using [LangChain](https://www.langchain.com/), [Gradio](https://gradio.app/), and the [Groq API](https://console.groq.com/), enhanced with document retrieval using Chroma vector stores and HuggingFace embeddings.

It performs **Retrieval-Augmented Generation (RAG)** on markdown documents in a `knowledge-base/` directory, enabling intelligent Q&A over your content.

---

## ⚙️ Features

- ✅ Chatbot UI using Gradio  
- ✅ Supports LLMs served via Groq (e.g., LLaMA 3, Mixtral)  
- ✅ Chunked document ingestion with metadata tagging  
- ✅ Vector embeddings via `sentence-transformers`  
- ✅ Persistent Chroma vector store  
- ✅ TSNE embedding visualization (optional)  
- ✅ Conversational memory support  

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

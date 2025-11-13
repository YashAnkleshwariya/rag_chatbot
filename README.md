# 🤖 Gemini RAG Chatbot with User Authentication & Admin Dashboard  
*A full-featured Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, Gemini API, local embeddings, and secure authentication.*

---

## 🚀 Overview

This project is a **production-grade RAG (Retrieval-Augmented Generation) chatbot** that allows users to:

- Upload documents (PDF, DOCX, PPTX, TXT, XLSX)
- Ask questions based on those documents
- Get strictly grounded answers using Gemini 2.0 Flash
- Maintain personal accounts via login/signup
- Track usage (queries, document uploads, sessions)

Admins get a complete Dashboard to:

- View all registered users  
- Monitor user activity (messages, uploads, chunks processed, login count)
- Reset usage  
- Delete user accounts  
- Clear document embeddings  

All embeddings are generated **locally with HuggingFace**, so there are **zero external costs** and **no API quota limits** except Gemini responses.

---

## 🏗 Tech Stack

| Component | Technology |
|----------|------------|
| Frontend | Streamlit |
| Backend | Python 3 |
| LLM Chat | Gemini 2.0 Flash |
| Embeddings | HuggingFace (Local, Free) |
| Vector Store | FAISS |
| Authentication | JSON Database + SHA256 Hashing |
| File Parsing | PyPDF2, python-docx, pptx, pandas |

---

## ✨ Features

### 👤 **User Features**
- Create account (email/password)
- Secure login
- Upload multi-format documents
- Ask questions grounded ONLY in uploaded files
- Full chat history for current session
- Usage automatically tracked:
  - Login count  
  - Queries asked  
  - Documents uploaded  
  - Chunks processed  

---

### 🔐 **Admin Features**
- Admin login
- View all users in a table
- Track usage per user
- Reset usage with one click
- Delete all user accounts
- View full database
- Logout

---

### 📄 **Supported File Formats**
- PDF
- Word (.docx)
- PowerPoint (.pptx)
- Text (.txt)
- Excel (.xlsx)

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

📦 rag-chatbot
 ┣ 📜 app.py
 ┣ 📜 requirements.txt
 ┣ 📜 users_db.json
 ┣ 📜 .env (local only)
 ┣ 📜 README.md

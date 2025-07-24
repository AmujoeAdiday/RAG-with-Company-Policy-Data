🤖 Interactive RAG vs LLM comparison with 3D embedding visualization. Shows how context transforms generic AI into company-specific intelligence. Built with DPR, FAISS, GPT-2, and Streamlit. Perfect demo of why enterprises need RAG, not just LLMs.

# 🤖 RAG vs LLM Demo: Why Context Matters


> **Transform generic AI responses into company-specific intelligence**  
> An interactive demonstration showing why enterprises need RAG, not just LLMs.

<img width="896" height="797" alt="RAG-impact" src="https://github.com/user-attachments/assets/5a3a8465-024c-408e-813c-be6a9af10640" />

## 🎯 What This Demonstrates

**The Problem:** Generic LLMs give generic answers  
**The Solution:** RAG (Retrieval-Augmented Generation) provides context-aware responses  
**The Proof:** Side-by-side comparison + 3D embedding visualization  

### Real Example:
**Question:** *"What should I do if my mobile phone is lost?"*

| Plain LLM | RAG System |
|-----------|------------|
| "Contact your carrier, change passwords, report to police..." | "Immediately report any lost or stolen mobile devices to the IT department or your supervisor." |
| ❌ Generic advice | ✅ Company-specific policy |

## ✨ Key Features

- 🔄 **Side-by-Side Comparison**: See the dramatic difference context makes
- 🎯 **3D Embedding Visualization**: Watch semantic similarity in action  
- 📊 **Real Company Data**: Uses actual corporate policy documents
- ⚡ **Interactive Interface**: Clean Streamlit UI for easy testing
- 🧠 **Enterprise-Ready**: Demonstrates production RAG architecture

## 🏗️ Technical Architecture
### Tech Stack:
<img width="870" height="751" alt="RAG_Technical_Flow" src="https://github.com/user-attachments/assets/91a2974e-395c-4b7a-805e-108ac742dd31" />

- **🔍 Retrieval**: DPR (Dense Passage Retrieval) from Facebook Research
- **⚡ Search**: FAISS for lightning-fast similarity search
- **🧠 Generation**: GPT-2 for natural language responses  
- **📊 Visualization**: t-SNE for 3D embedding space plots
- **🖥️ Interface**: Streamlit for interactive web app
- **📈 Analysis**: NumPy, scikit-learn for data processing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB RAM minimum
- GPU recommended (but not required)

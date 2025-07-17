# -Multilingual-RAG-Chatbot-for-Indian-Government-Schemes

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about various Indian government schemes (like Ayushman Bharat, PM-KISAN, PMAY, etc.) in three languages: English, Tamil, and Hindi.

Built using OpenAI GPT, LangChain, FAISS, and Streamlit, the chatbot allows users to ask questions and get accurate responses by retrieving context directly from a PDF document.

🚀 Features
   Multilingual Support: Ask questions in English, Tamil, or Hindi

   PDF-Based RAG: Uses official government scheme documents as source

   LangChain + FAISS: Vector search and context retrieval for accurate answers

   OpenAI GPT-4: High-quality natural language understanding and response

   Streamlit UI: Simple and responsive web interface

🧱 Tech Stack:

   OpenAI GPT-4 (via LangChain)

   LangChain – document loading, chunking, prompting

   FAISS – vector store for semantic similarity search

   PyPDFLoader – loading multilingual PDFs

   Streamlit – web app frontend

📌 Notes:

Make sure the PDF (indian_govt_schemes.pdf) contains actual Tamil/Hindi/English content (Unicode).

For larger PDFs, FAISS may take a few seconds to index on first run.

All answers are generated using GPT based on retrieved document chunks.

📸 Screenshots



🙌 Acknowledgements:

<img width="1920" height="1080" alt="multi lag" src="https://github.com/user-attachments/assets/a3d078a1-432d-4cbb-8904-1193ec4c81db" />

LangChain

FAISS

Streamlit

OpenAI


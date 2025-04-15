import streamlit as st
import os
import re
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

# Configurações iniciais da página
st.set_page_config(page_title="📘 PDF do Amor com IA", page_icon="💖", layout="wide")
st.title("📘 PDF do Amor")
st.write("Faça upload de um PDF e pergunte qualquer coisa sobre o conteúdo. A IA responde com base no que leu!")

# Pega a chave da API da OpenAI (segura via secrets)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ A chave da OpenAI não está configurada. Vá em Settings > Secrets e adicione 'OPENAI_API_KEY'.")
    st.stop()

# Funções principais
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def split_text(text, max_length=800):
    paragraphs = re.split(r'\n{2,}|\r\n{2,}', text)
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) <= max_length:
            current += p + " "
        else:
            chunks.append(current.strip())
            current = p
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 100]

def get_embeddings(texts):
    client = OpenAI(api_key=openai_api_key)
    res = client.embeddings.create(input=texts, model="text-embedding-ada-002")
    return np.array([r.embedding for r in res.data])

# Upload e processamento do PDF
uploaded_file = st.file_uploader("📄 Envie um arquivo PDF", type="pdf")

if uploaded_file:
    with st.spinner("🧠 Lendo o conteúdo..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(raw_text)
        embeddings = get_embeddings(chunks)
        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embeddings
        st.success(f"✅ Documento processado com {len(chunks)} partes.")

# Campo de pergunta
if "chunks" in st.session_state and "embeddings" in st.session_state:
    question = st.text_input("❓ Pergunte algo sobre o PDF:")
    if question:
        with st.spinner("💬 Gerando resposta..."):
            q_embedding = get_embeddings([question])[0].reshape(1, -1)
            scores = cosine_similarity(q_embedding, st.session_state["embeddings"]).flatten()
            top_idx = scores.argsort()[-3:][::-1]
            context = "\n\n".join([st.session_state["chunks"][i] for i in top_idx])

            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você responde apenas com base no contexto dado."},
                    {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {question}"}
                ],
                temperature=0.0
            )
            answer = response.choices[0].message.content
            st.markdown("### 💡 Resposta:")
            st.success(answer)

            st.markdown("### 🔍 Trechos usados como base:")
            for i, idx in enumerate(top_idx):
                with st.expander(f"Fonte {i+1} — Relevância: {scores[idx]:.2f}"):
                    st.code(st.session_state["chunks"][idx][:1000])

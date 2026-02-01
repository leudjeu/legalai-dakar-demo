import streamlit as st
import tempfile
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# -------------------------------
# CONFIG STREAMLIT
# -------------------------------
st.set_page_config(
    page_title="LegalAI Dakar – Démo OHADA",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ LegalAI Dakar")
st.subheader("IA Juridique – Démo publique OHADA")
st.caption("⚠️ Version démonstration – Documents publics – Aucune consultation juridique")

# -------------------------------
# CSS
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: #e5e7eb;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("ℹ️ À propos")
    st.markdown("""
    **LegalAI Dakar**  
    Démonstration d’un chatbot IA juridique  
    spécialisé en droit OHADA.
    
    🔒 Version cabinet = OFFLINE  
    🌐 Cette version = MARKETING
    """)

    if st.button("🧹 Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

# -------------------------------
# SESSION
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# CHARGEMENT DES PDF (AUTO)
# -------------------------------
PDF_FILES = [
    "data/ohada_auscgie.pdf",
    "data/aupsrve.pdf"
]

documents = []
for pdf in PDF_FILES:
    loader = PyPDFLoader(pdf)
    documents.extend(loader.load())

# -------------------------------
# RAG SETUP (CACHE)
# -------------------------------
@st.cache_resource
def setup_qa(_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    splits = splitter.split_documents(_docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(splits, embeddings)

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

qa_chain = setup_qa(documents)

# -------------------------------
# AFFICHAGE CHAT
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# INPUT UTILISATEUR
# -------------------------------
question = st.chat_input("Posez votre question juridique (OHADA)")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyse juridique en cours..."):
            try:
                result = qa_chain.invoke(question)
                answer = result["result"]

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            except Exception as e:
                st.error("Erreur IA. Vérifiez la clé OpenAI.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Erreur technique temporaire."
                })

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("© 2024 LegalAI Dakar – Démo Marketing – Confidentialité Garantie")

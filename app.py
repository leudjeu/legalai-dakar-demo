# app.py
import streamlit as st
import tempfile
from pathlib import Path

# ---- IMPORTS LANGCHAIN / OLLAMA ----
try:
    from langchain_community.llms import Ollama
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA
except ModuleNotFoundError:
    st.error("⚠️ Veuillez installer les dépendances dans votre requirements.txt :\n"
             "langchain, langchain_community, chromadb, streamlit_text_splitter")

# ---- CONFIGURATION PAGE ----
st.set_page_config(
    page_title="LegalAI Dakar",
    page_icon="⚖️",
    layout="wide"
)

# ---- STYLE ----
st.markdown("""
<style>
.stButton>button { width: 100%; height: 3em; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ LegalAI Dakar")
st.subheader("L'IA Souveraine spécialisée en Droit Sénégalais et OHADA")

# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Paramètres")
    model_name = st.selectbox("Modèle local", ["llama3", "mistral"], index=0)
    st.info("Cette instance tourne localement sur votre machine pour garantir le secret professionnel.")
    
    uploaded_file = st.file_uploader("Charger un document juridique (PDF)", type="pdf")
    
    if st.button("Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

# ---- INITIALISATION SESSION ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- FAQ ----
st.markdown("### 💡 Questions fréquentes")
faq_cols = st.columns(3)
faqs = [
    "Quelles sont les conditions de création d'une SARL ?",
    "Quelle est la durée d'un bail commercial au Sénégal ?",
    "Quelles sont les compétences de la CCJA ?",
    "Quelles sont les mentions obligatoires d'un contrat de travail ?",
    "Comment transformer une SARL en SA selon l'OHADA ?",
    "Quels sont les délais de prescription en droit commercial ?"
]
selected_query = None
for i, question in enumerate(faqs):
    with faq_cols[i % 3]:
        if st.button(question, key=f"faq_{i}"):
            selected_query = question

# ---- LOGIQUE PDF + RAG ----
qa_chain = None
if uploaded_file is not None:
    # Création d'un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    @st.cache_resource
    def setup_qa(file_path, model_name):
        # Chargement PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # Split des textes
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        # Embeddings
        embeddings = OllamaEmbeddings(model=model_name)
        vectorstore = Chroma.from_documents(splits, embeddings)
        # LLM Ollama
        llm = Ollama(model=model_name)
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

    qa_chain = setup_qa(tmp_file_path, model_name)
    st.success("📄 Document analysé. Posez vos questions ci-dessous ou utilisez les FAQ.")

# ---- CHAT INTERFACE ----
user_input = st.chat_input("Votre message...")
query = selected_query if selected_query else user_input

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyse juridique en cours..."):
            if qa_chain:
                try:
                    result = qa_chain.invoke(query)
                    answer = result["result"]
                except Exception as e:
                    answer = f"Erreur avec Ollama : {e}"
            else:
                answer = "⚠️ Aucun PDF chargé, je peux seulement répondre aux FAQ."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("---")
st.caption("© 2024 LegalAI Dakar - Confidentialité Garantie - Solution Air-Gapped")

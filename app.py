import streamlit as st
import tempfile
import os
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- Page config ---
st.set_page_config(
    page_title="LegalAI Dakar Demo",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ LegalAI Dakar")
st.subheader("Mini chat IA OHADA - Démo publique")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    model_name = st.selectbox("Modèle Local", ["llama3", "mistral"])
    st.info("Cette instance tourne localement pour garantir le secret professionnel.")
    uploaded_file = st.file_uploader("Charger un document juridique (PDF)", type="pdf")

    if st.button("Effacer l'historique"):
        st.session_state.messages = []
        st.experimental_rerun()

# --- Initialisation session chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- FAQ ---
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

# --- Logique IA ---
def setup_qa(file_path, model):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=model)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    llm = Ollama(model=model)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

qa_chain = None

# --- Analyse PDF mini OHADA pour démo ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    qa_chain = setup_qa(tmp_path, model_name)
    st.success("Document analysé. Posez vos questions ci-dessous ou utilisez les FAQ.")
else:
    # Démo avec PDF par défaut dans /pdfs/
    default_pdf = os.path.join("pdfs", "acte_uniforme_ohada.pdf")
    if os.path.exists(default_pdf):
        qa_chain = setup_qa(default_pdf, model_name)
        st.info("Mode démo avec PDF OHADA pré-chargé.")
    else:
        st.warning("📄 Aucun PDF trouvé pour la démo. Chargez un fichier ou ajoutez un PDF dans /pdfs/.")

# --- Chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Posez votre question...")
query = selected_query if selected_query else user_input

if query and qa_chain:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyse en cours..."):
            try:
                response = qa_chain.invoke(query)
                answer = response["result"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Erreur Ollama: {str(e)}")
                st.info("Vérifiez que le serveur Ollama tourne localement.")

st.markdown("---")
st.caption("© 2026 LegalAI Dakar - Démo publique")

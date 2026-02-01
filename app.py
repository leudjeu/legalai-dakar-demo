import streamlit as st
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --------------------------------------------------
# CONFIGURATION PAGE
# --------------------------------------------------
st.set_page_config(
    page_title="LegalAI Dakar",
    page_icon="⚖️",
    layout="wide"
)

# --------------------------------------------------
# STYLE CSS (cabinet d'avocats)
# --------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.stButton>button {
    width: 100%;
    border-radius: 6px;
    height: 3em;
    background-color: #ffffff;
    color: #1a1a1a;
    border: 1px solid #ced4da;
}
.stButton>button:hover {
    border-color: #0d6efd;
    color: #0d6efd;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITRES
# --------------------------------------------------
st.title("⚖️ LegalAI Dakar")
st.subheader("IA juridique spécialisée en droit sénégalais et OHADA")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Démo publique sécurisée – GPT + documents OHADA")
    uploaded_file = st.file_uploader(
        "📄 Charger un document juridique (PDF)",
        type="pdf"
    )

    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------
# SESSION CHAT
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# FAQ
# --------------------------------------------------
st.markdown("### 💡 Questions fréquentes")
faq_cols = st.columns(3)

faqs = [
    "Quelles sont les conditions de création d'une SARL selon l'OHADA ?",
    "Quelle est la durée d'un bail commercial au Sénégal ?",
    "Quelles sont les compétences de la CCJA ?",
    "Quelles sont les mentions obligatoires d'un contrat de travail ?",
    "Comment transformer une SARL en SA selon l'OHADA ?",
    "Quels sont les délais de prescription en droit commercial ?"
]

selected_query = None
for i, q in enumerate(faqs):
    with faq_cols[i % 3]:
        if st.button(q, key=f"faq_{i}"):
            selected_query = q

# --------------------------------------------------
# FONCTION RAG (CACHE)
# --------------------------------------------------
@st.cache_resource
def setup_qa(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    splits = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

# --------------------------------------------------
# LOGIQUE PRINCIPALE
# --------------------------------------------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name

    qa_chain = setup_qa(pdf_path)
    st.success("📘 Document analysé. Posez votre question.")

    # Affichage historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Posez votre question juridique…")
    query = selected_query if selected_query else user_input

    if query:
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Analyse juridique en cours…"):
                try:
                    response = qa_chain.invoke(query)
                    answer = response["result"]
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    st.error("Erreur lors de l'analyse.")
                    st.exception(e)

else:
    st.warning("⬅️ Chargez un document PDF OHADA pour démarrer.")
    st.info("Cette démo est conçue pour 1–2 documents juridiques maximum.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("© 2024 LegalAI Dakar — Démo publique | Produit final : IA souveraine")

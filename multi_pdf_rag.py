import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import tempfile
import re

load_dotenv()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_id" not in st.session_state:
    st.session_state.chat_id = 0

st.set_page_config(
    page_title="Multi-PDF RAG with Sources", page_icon="📚", layout="wide"
)
st.title("📚 Multi-PDF RAG with Sources")
st.caption("Ask questions across PDFs and verify the source filename and page.")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Set OPENAI_API_KEY in your .env file first.")
    st.stop()
files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
chunk_size = st.sidebar.slider("Chunk size", 400, 1600, 800, 100)
overlap = st.sidebar.slider("Chunk overlap", 0, 300, 100, 50)
top_k = st.sidebar.slider("Retrieved chunks", 3, 12, 5)
if st.button("Process PDFs", disabled=not files):
    docs = []
    with st.spinner("Reading, chunking and embedding PDFs..."):
        with tempfile.TemporaryDirectory() as tmp:
            for i, f in enumerate(files):
                path = os.path.join(tmp, f"{i}.pdf")
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                for page in PyPDFLoader(path).load():
                    page.metadata["source"] = f.name
                    docs.append(page)
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        ).split_documents(docs)
        if not chunks:
            st.error("No extractable text found. Scanned PDFs require OCR.")
            st.stop()
        
        st.session_state.db = FAISS.from_documents(
             chunks, OpenAIEmbeddings(model="text-embedding-3-small")
        )

        db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "faiss_index"
        )

        st.session_state.db.save_local(db_path)
        st.success("FAISS vector database saved successfully!")

        st.session_state.count = len(chunks)
        st.session_state.names = [f.name for f in files]

        # Start a fresh conversation for newly processed PDFs
        st.session_state.messages = []
        st.session_state.chat_id += 1

    st.success(f"Indexed {len(chunks)} chunks from {len(files)} PDF(s).")

def detect_database(question, uploaded_files):
    q = question.lower()

    # Detect cross-database comparison questions
    comparison_words = [
        "compare",
        "comparison",
        "across",
        "all databases",
        "differences",
        "disagree",
        "versus",
        " vs ",
        "between",
    ]

    if any(word in q for word in comparison_words):
        return None

    # Database keywords
    database_keywords = {
        "mariadb": ["mariadb", "mysql"],
        "postgresql": ["postgresql", "postgres", "pgvector"],
        "sql_server": ["sql server", "mssql", "tempdb"],
    }

    matched_databases = []

    for database, keywords in database_keywords.items():
        if any(keyword in q for keyword in keywords):
            matched_databases.append(database)

    # Search all documents if multiple databases are mentioned
    if len(matched_databases) != 1:
        return None

    database = matched_databases[0]

    # Match actual uploaded filename
    for filename in uploaded_files:
        name = filename.lower()

        if database == "mariadb" and (
            "mariadb" in name or "mysql" in name
        ):
            return filename

        if database == "postgresql" and (
            "postgresql" in name or "postgres" in name
        ):
            return filename

        if database == "sql_server" and (
            "sql_server" in name or "sql server" in name
        ):
            return filename

    return None

if "db" in st.session_state:

    st.info(
        f"Indexed: {', '.join(st.session_state.names)} "
        f"({st.session_state.count} chunks)"
    )

    # Start a new chat
    if st.button("➕ New Chat", use_container_width=False):
        st.session_state.messages = []
        st.session_state.chat_id += 1
        st.rerun()

    if not st.session_state.messages:
        st.info("👋 Start a new conversation by asking a question below.")

    # Display previous conversations
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant":

                for source in message.get("sources", []):
                    with st.expander(
                        f'[S{source["id"]}] '
                        f'{source["file"]} — Page {source["page"]}'
                    ):
                        st.write(source["content"])

    # ChatGPT-style question input
    question = st.chat_input(
        "💬 Ask a question about your PDFs...",
        key=f"chat_input_{st.session_state.chat_id}"
    )

    if question:

        # Save and display user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Generate assistant response
        with st.chat_message("assistant"):

            with st.spinner("Searching PDF documents..."):

                database_file = detect_database(
                    question,
                    st.session_state.names
                )

                if database_file:
                    results = (
                        st.session_state.db.similarity_search(
                            question,
                            k=top_k,
                            filter={"source": database_file}
                        )
                    )
                else:
                    results = (
                        st.session_state.db.similarity_search(
                            question,
                            k=top_k
                        )
                    )

                if not results:
                    answer = (
                        "No matching information was found "
                        "in the uploaded documents."
                    )
                    sources = []

                else:

                    # Build context from retrieved PDFs
                    context = "\n\n".join(
                        f'[S{i}] FILE: {d.metadata["source"]} '
                        f'| PAGE: {d.metadata.get("page", 0) + 1}\n'
                        f'{d.page_content}'
                        for i, d in enumerate(results, 1)
                    )

                    prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                """
You are a professional database RAG assistant.

Answer questions using ONLY the provided
PDF excerpts.

Rules:
1. Give clear and accurate answers.
2. Cite sources using [S1], [S2], etc.
3. If information is missing, say so.
4. If documents disagree, explain the
   differences and cite both sources.
5. Do not invent information.
6. Use concise, readable formatting.

EXCERPTS:
{context}
                                """
                            ),
                            ("human", "{question}")
                        ]
                    )

                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0
                    )

                    answer = (
                        (prompt | llm)
                        .invoke(
                            {
                                "context": context,
                                "question": question
                            }
                        )
                        .content
                    )

                    # Identify source IDs cited in the generated answer
                    cited_ids = {
                        int(source_id)
                        for source_id in re.findall(r"\[S(\d+)\]", answer)
                    }

                    # Display only cited sources
                    sources = []

                    for i, d in enumerate(results, 1):
                        if i in cited_ids:
                            sources.append(
                                {
                                    "id": i,
                                    "file": d.metadata.get("source", "Unknown"),
                                    "page": d.metadata.get("page", 0) + 1,
                                    "content": d.page_content
                                }
                            )
            # Display answer
            st.markdown(answer)

            # Display source references
            if sources:
                st.caption("📚 Retrieved PDF Sources")

                for source in sources:
                    with st.expander(
                        f'[S{source["id"]}] '
                        f'{source["file"]} — Page {source["page"]}'
                    ):
                        st.write(source["content"])

        # Store answer and sources in chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources
            }
        )
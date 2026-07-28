import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Streamlit Page Setup
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="💬", layout="wide")
st.title("💬 Interactive PDF Chatbot")
st.caption("Upload PDFs in the sidebar, process them, and chat with your documents in real time.")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar for PDF Upload & Ingestion
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Reading & indexing PDFs..."):
                all_docs = []
                for file in uploaded_files:
                    # Write file temporarily to disk so PyPDFLoader can parse it
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(file.read())
                        tmp_path = tmp_file.name

                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    
                    # Store original filename in document metadata
                    for doc in docs:
                        doc.metadata["source"] = file.name
                    
                    all_docs.extend(docs)
                    os.remove(tmp_path)  # Clean up temp file

                # Chunking text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(all_docs)

                # Build Vectorstore in memory using FAISS
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                vector_store = FAISS.from_documents(chunks, embeddings)
                
                st.session_state.vector_store = vector_store
                st.success(f"Indexed {len(uploaded_files)} PDF(s) into {len(chunks)} text chunks!")

# Helper function to construct the RAG chain with chat history support
def get_rag_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Step 1: Prompt to contextualize user question based on chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Step 2: Answer synthesis prompt
    system_prompt = (
        "You are an assistant for question-answering tasks.\n"
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer based on the context, "
        "state that clearly.\n"
        "Cite the document name and page number for facts when possible.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Render Chat Messages from History
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat Input UI
user_query = st.chat_input("Ask a question about your uploaded PDFs...")

if user_query:
    if not st.session_state.vector_store:
        st.error("Please upload and click 'Process Documents' in the sidebar first!")
    else:
        # Display user input
        with st.chat_message("user"):
            st.write(user_query)

        # Generate response using RAG chain
        rag_chain = get_rag_chain(st.session_state.vector_store)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                response = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                
                answer = response["answer"]
                st.write(answer)
                
                # Expandable view for cited source chunks
                if "context" in response and response["context"]:
                    with st.expander("🔍 View Referenced Sources"):
                        for doc in response["context"]:
                            source = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", 0) + 1
                            st.markdown(f"- **{source}** (Page {page})")

        # Save to session chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=answer))

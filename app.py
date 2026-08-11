import os

import streamlit as st
from dotenv import load_dotenv

import rag_pipeline as rag

load_dotenv()

st.set_page_config(page_title="Document Assistant", page_icon="📚", layout="wide")

if not rag.has_api_key():
    st.error(
        "GOOGLE_API_KEY not found. Create a `.env` file in the project root "
        "with `GOOGLE_API_KEY=your_key` and restart the app."
    )
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "last_processing_result" not in st.session_state:
    st.session_state.last_processing_result = None

st.title("📚 Document Assistant")
st.write("Upload your documents or add a webpage, then ask questions about the information inside them.")

with st.sidebar:
    st.header("Knowledge Base")
    st.metric("Stored chunks", rag.get_database_count())
    st.divider()

    st.subheader("Current sources")
    if st.session_state.documents:
        for source in st.session_state.documents:
            st.write(f"• {source}")
    else:
        st.caption("No sources have been added in this session.")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Clear source list", use_container_width=True):
        st.session_state.documents = []
        st.session_state.last_processing_result = None
        st.rerun()

st.subheader("Add documents")
uploaded_files = st.file_uploader("Choose PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
url = st.text_input("Webpage URL", placeholder="https://example.com/article")

if st.button("Process documents", type="primary", use_container_width=True):
    if not uploaded_files and not url.strip():
        st.warning("Please upload at least one document or enter a webpage URL.")
    else:
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(rag.DATA_DIR, uploaded_file.name)
            try:
                with open(file_path, "wb") as out:
                    out.write(uploaded_file.getbuffer())
                saved_files.append((uploaded_file.name, file_path))
            except Exception as exc:
                st.error(f"Could not save {uploaded_file.name}: {exc}")

        with st.status("Processing your sources...", expanded=True) as status:
            try:
                result = rag.process_inputs(saved_files, url.strip() or None)
                st.session_state.last_processing_result = result

                if result["documents"] == 0 and result["chunks"] == 0:
                    status.update(label="Nothing was processed", state="error")
                    st.error("No readable content was found.")
                else:
                    for filename, _ in saved_files:
                        if filename not in st.session_state.documents:
                            st.session_state.documents.append(filename)

                    if url.strip() and url.strip() not in st.session_state.documents:
                        st.session_state.documents.append(url.strip())

                    st.write(f"Loaded document pages/sections: {result['documents']}")
                    st.write(f"Created chunks: {result['chunks']}")

                    if result["errors"]:
                        st.warning("Some sources could not be processed.")
                        for error in result["errors"]:
                            st.write(f"• {error}")
                        status.update(label="Processing completed with warnings", state="complete")
                    else:
                        status.update(label="Processing completed", state="complete")

            except Exception as exc:
                status.update(label="Processing failed", state="error")
                st.error(f"Processing error: {exc}")

if st.session_state.last_processing_result and st.session_state.last_processing_result["chunks"] > 0:
    st.success(f"{st.session_state.last_processing_result['chunks']} chunks are now available for questions.")

st.divider()
st.subheader("Ask your documents")

for message in st.session_state.chat_history:
    role = message.get("role")
    if role not in ("user", "assistant"):
        continue

    with st.chat_message(role):
        st.markdown(message.get("content", ""))
        sources = message.get("sources", [])
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.write(f"• {source}")

question = st.chat_input("Ask something about your documents...")

if question and question.strip():
    question = question.strip()
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                answer, sources = rag.ask_question(question, chat_history=st.session_state.chat_history[:-1])
            except Exception as exc:
                answer = "Something went wrong while searching the knowledge base."
                sources = []
                st.error(str(exc))

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.write(f"• {source}")

    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})

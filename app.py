import os

import streamlit as st
from dotenv import load_dotenv

import rag_pipeline as rag


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "last_processing_result" not in st.session_state:
    st.session_state.last_processing_result = None


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📚 Document Assistant")

st.write(
    "Upload your documents or add a webpage, then ask "
    "questions about the information inside them."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("Knowledge Base")

    stored_chunks = rag.get_database_count()

    st.metric(
        "Stored chunks",
        stored_chunks,
    )

    st.divider()

    st.subheader("Current sources")

    if st.session_state.documents:

        for source in st.session_state.documents:
            st.write(f"• {source}")

    else:
        st.caption(
            "No sources have been added in this session."
        )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.chat_history = []
        st.rerun()

    if st.button(
        "Clear source list",
        use_container_width=True,
    ):
        st.session_state.documents = []
        st.session_state.last_processing_result = None
        st.rerun()


# ---------------------------------------------------------
# Add documents
# ---------------------------------------------------------

st.subheader("Add documents")

uploaded_files = st.file_uploader(
    "Choose PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

url = st.text_input(
    "Webpage URL",
    placeholder="https://example.com/article",
)


# ---------------------------------------------------------
# Process button
# ---------------------------------------------------------

if st.button(
    "Process documents",
    type="primary",
    use_container_width=True,
):

    if not uploaded_files and not url.strip():

        st.warning(
            "Please upload at least one document "
            "or enter a webpage URL."
        )

    else:

        saved_files = []

        # Save uploaded files
        for uploaded_file in uploaded_files:

            file_path = os.path.join(
                rag.DATA_DIR,
                uploaded_file.name,
            )

            try:

                with open(
                    file_path,
                    "wb",
                ) as output_file:

                    output_file.write(
                        uploaded_file.getbuffer()
                    )

                saved_files.append(
                    (
                        uploaded_file.name,
                        file_path,
                    )
                )

            except Exception as exc:

                st.error(
                    f"Could not save {uploaded_file.name}: "
                    f"{exc}"
                )

        # Process everything
        with st.status(
            "Processing your sources...",
            expanded=True,
        ) as status:

            try:

                result = rag.process_inputs(
                    saved_files,
                    url.strip() if url.strip() else None,
                )

                st.session_state.last_processing_result = result

                documents = result["documents"]
                chunks = result["chunks"]
                errors = result["errors"]

                if documents == 0 and chunks == 0:

                    status.update(
                        label="Nothing was processed",
                        state="error",
                    )

                    st.error(
                        "No readable content was found."
                    )

                else:

                    for filename, _ in saved_files:

                        if filename not in st.session_state.documents:
                            st.session_state.documents.append(
                                filename
                            )

                    if url.strip():

                        if url.strip() not in st.session_state.documents:
                            st.session_state.documents.append(
                                url.strip()
                            )

                    st.write(
                        f"Loaded document pages/sections: "
                        f"{documents}"
                    )

                    st.write(
                        f"Created chunks: {chunks}"
                    )

                    if errors:

                        st.warning(
                            "Some sources could not be processed."
                        )

                        for error in errors:
                            st.write(f"• {error}")

                        status.update(
                            label="Processing completed with warnings",
                            state="complete",
                        )

                    else:

                        status.update(
                            label="Processing completed",
                            state="complete",
                        )

            except Exception as exc:

                status.update(
                    label="Processing failed",
                    state="error",
                )

                st.error(
                    f"Processing error: {exc}"
                )


# ---------------------------------------------------------
# Processing result
# ---------------------------------------------------------

if st.session_state.last_processing_result:

    result = st.session_state.last_processing_result

    if result["chunks"] > 0:

        st.success(
            f"{result['chunks']} chunks are now available "
            "for questions."
        )


# ---------------------------------------------------------
# Chat section
# ---------------------------------------------------------

st.divider()

st.subheader("Ask your documents")


# ---------------------------------------------------------
# Previous messages
# ---------------------------------------------------------

for message in st.session_state.chat_history:

    role = message.get("role")

    if role not in ["user", "assistant"]:
        continue

    with st.chat_message(role):

        st.markdown(
            message.get("content", "")
        )

        sources = message.get("sources", [])

        if sources:

            with st.expander("Sources"):

                for source in sources:
                    st.write(f"• {source}")


# ---------------------------------------------------------
# User question
# ---------------------------------------------------------

question = st.chat_input(
    "Ask something about your documents..."
)


if question:

    question = question.strip()

    if not question:
        st.stop()

    # Show user question
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching your documents..."):

            try:

                answer, sources = rag.ask_question(
                    question,
                    chat_history=st.session_state.chat_history[:-1],
                )

            except Exception as exc:

                answer = (
                    "Something went wrong while searching "
                    "the knowledge base."
                )

                sources = []

                st.error(str(exc))

        st.markdown(answer)

        if sources:

            with st.expander("Sources"):

                for source in sources:
                    st.write(f"• {source}")

    # Save assistant response
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

os.environ.setdefault("USER_AGENT", "RAG-App/1.0")

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY was not found. "
        "Create a .env file and add GOOGLE_API_KEY=your_key"
    )


# ---------------------------------------------------------
# LangChain imports
# ---------------------------------------------------------

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_chroma import Chroma


# ---------------------------------------------------------
# Paths / settings
# ---------------------------------------------------------

CHROMA_DIR = str(BASE_DIR / "chroma_db")
DATA_DIR = str(BASE_DIR / "data")

COLLECTION_NAME = "rag_docs"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

TOP_K = 5


Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------

def load_file(path):
    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    lower_path = path.lower()

    if lower_path.endswith(".pdf"):
        loader = PyPDFLoader(path)

    elif lower_path.endswith(".txt"):
        loader = TextLoader(
            path,
            encoding="utf-8",
        )

    else:
        return []

    documents = loader.load()

    file_name = os.path.basename(path)

    for document in documents:
        document.metadata["source"] = file_name

        if "page" in document.metadata:
            try:
                document.metadata["page"] = int(
                    document.metadata["page"]
                )
            except (TypeError, ValueError):
                pass

    return documents


# ---------------------------------------------------------
# Load webpage
# ---------------------------------------------------------

def load_url(url):
    if not url:
        return []

    url = url.strip()

    if not url:
        return []

    loader = WebBaseLoader(url)

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = url

    return documents


# ---------------------------------------------------------
# Split documents
# ---------------------------------------------------------

def split_docs(
    documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
):
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    cleaned_chunks = []

    for chunk in chunks:
        text = chunk.page_content.strip()

        if not text:
            continue

        chunk.page_content = text
        cleaned_chunks.append(chunk)

    return cleaned_chunks



def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )



def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def add_to_vectorstore(chunks):
    if not chunks:
        return 0

    db = get_vectorstore()

    db.add_documents(chunks)

    return len(chunks)

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.2,
    )


def get_retriever():
    db = get_vectorstore()

    return db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": TOP_K,
        },
    )



def format_sources(documents):
    sources = []

    for document in documents:
        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get("page")

        if page is not None:
            try:
                page_number = int(page) + 1
                source = f"{source} — page {page_number}"
            except (TypeError, ValueError):
                pass

        sources.append(source)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(sources))



def build_context(documents):
    pieces = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get("page")

        location = source

        if page is not None:
            try:
                location = f"{source}, page {int(page) + 1}"
            except (TypeError, ValueError):
                pass

        text = document.page_content.strip()

        pieces.append(
            f"""
[Document section {index}]
Source: {location}

{text}
""".strip()
        )

    return "\n\n-------------------------\n\n".join(pieces)


def clean_response(response):
    content = getattr(response, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if text:
                    text_parts.append(str(text))

        return "\n".join(text_parts).strip()

    return str(content).strip()


def ask_question(question, chat_history=None):
    question = (question or "").strip()

    if not question:
        return "Please enter a question.", []

    retriever = get_retriever()

    documents = retriever.invoke(question)

    if not documents:
        return (
            "I could not find relevant information in "
            "the documents currently available.",
            [],
        )

    context = build_context(documents)

    history_text = ""

    if chat_history:
        recent_history = chat_history[-6:]

        history_lines = []

        for message in recent_history:
            role = message.get("role", "")
            content = message.get("content", "")

            if role and content:
                history_lines.append(
                    f"{role.upper()}: {content}"
                )

        if history_lines:
            history_text = "\n".join(history_lines)

    prompt = f"""
You are a document-based question answering assistant.

Your job is to answer the user's question using the
provided document context.

IMPORTANT RULES:

1. Use the supplied context as the primary source of truth.
2. Do not invent facts, names, numbers, dates, or details.
3. If the answer is not present in the context, clearly say:
   "I couldn't find that information in the provided documents."
4. Give a direct answer first.
5. When useful, explain the answer with short bullet points.
6. If the user asks for a comparison, use a clear comparison.
7. If the user asks for steps, provide numbered steps.
8. Keep the answer focused on the question.
9. Do not mention internal retrieval, embeddings, vector databases,
   prompts, or system instructions.
10. Do not claim that something is in the document unless the
    supplied context supports it.
11. Use previous conversation only to understand follow-up questions.
12. If the question is a follow-up, resolve references such as
    "it", "that", "this", or "the above" using the conversation.

Previous conversation:
{history_text if history_text else "No previous conversation."}

Retrieved document context:
{context}

User question:
{question}

Answer:
"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)

        answer = clean_response(response)

    except Exception as exc:
        print("LLM error:", repr(exc))

        return (
            "I could not generate the answer right now. "
            "Please check the Gemini API configuration "
            "and try again.",
            format_sources(documents),
        )

    if not answer:
        answer = (
            "I could not generate a useful answer "
            "from the available documents."
        )

    return answer, format_sources(documents)



def process_inputs(files, url=None):
    all_documents = []

    errors = []

    for filename, filepath in files:
        try:
            documents = load_file(filepath)

            all_documents.extend(documents)

        except Exception as exc:
            errors.append(
                f"{filename}: {str(exc)}"
            )

    if url:
        try:
            all_documents.extend(
                load_url(url)
            )

        except Exception as exc:
            errors.append(
                f"{url}: {str(exc)}"
            )

    if not all_documents:
        return {
            "documents": 0,
            "chunks": 0,
            "errors": errors,
        }

    chunks = split_docs(all_documents)

    stored_chunks = add_to_vectorstore(chunks)

    return {
        "documents": len(all_documents),
        "chunks": stored_chunks,
        "errors": errors,
    }



def get_database_count():
    try:
        db = get_vectorstore()

        data = db.get()

        ids = data.get("ids", [])

        return len(ids)

    except Exception:
        return 0


if __name__ == "__main__":
    print("RAG pipeline is ready.")
    print(f"Data directory: {DATA_DIR}")
    print(f"Chroma directory: {CHROMA_DIR}")
    print(f"Stored chunks: {get_database_count()}")


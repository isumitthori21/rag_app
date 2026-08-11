import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

os.environ.setdefault("USER_AGENT", "RAG-App/1.0")

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

CHROMA_DIR = str(BASE_DIR / "chroma_db")
DATA_DIR = str(BASE_DIR / "data")
COLLECTION_NAME = "rag_docs"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)


def has_api_key():
    return bool(os.getenv("GOOGLE_API_KEY"))


def load_file(path):
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    lower_path = path.lower()
    if lower_path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif lower_path.endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
    else:
        return []

    documents = loader.load()
    file_name = os.path.basename(path)
    for document in documents:
        document.metadata["source"] = file_name
        if "page" in document.metadata:
            try:
                document.metadata["page"] = int(document.metadata["page"])
            except (TypeError, ValueError):
                pass

    return documents


def load_url(url):
    url = (url or "").strip()
    if not url:
        return []

    loader = WebBaseLoader(url)
    documents = loader.load()
    for document in documents:
        document.metadata["source"] = url

    return documents


def split_docs(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
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
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


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
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)


def get_retriever():
    db = get_vectorstore()
    return db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})


def format_sources(documents):
    sources = []
    for document in documents:
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page")
        if page is not None:
            try:
                source = f"{source} — page {int(page) + 1}"
            except (TypeError, ValueError):
                pass
        sources.append(source)

    return list(dict.fromkeys(sources))  # dedupe, keep order


def build_context(documents):
    pieces = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page")
        location = source
        if page is not None:
            try:
                location = f"{source}, page {int(page) + 1}"
            except (TypeError, ValueError):
                pass

        text = document.page_content.strip()
        pieces.append(f"[Document section {index}]\nSource: {location}\n\n{text}")

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
            elif isinstance(item, dict) and item.get("text"):
                text_parts.append(str(item["text"]))
        return "\n".join(text_parts).strip()

    return str(content).strip()


PROMPT_TEMPLATE = """You are a document-based question answering assistant.
Answer using the provided context only, don't invent facts, names, numbers or dates.
If the answer isn't in the context, say "I couldn't find that information in the provided documents."
Give a direct answer first, use bullet points or numbered steps if that helps.
Use the previous conversation only to resolve follow-up references like "it" or "that".
Don't mention retrieval, embeddings, or vector databases in your answer.

Previous conversation:
{history}

Retrieved context:
{context}

Question: {question}

Answer:"""


def ask_question(question, chat_history=None):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", []

    retriever = get_retriever()
    documents = retriever.invoke(question)
    if not documents:
        return "I could not find relevant information in the documents currently available.", []

    context = build_context(documents)

    history_text = "No previous conversation."
    if chat_history:
        recent = chat_history[-6:]
        lines = [f"{m.get('role', '').upper()}: {m.get('content', '')}" for m in recent if m.get("role") and m.get("content")]
        if lines:
            history_text = "\n".join(lines)

    prompt = PROMPT_TEMPLATE.format(history=history_text, context=context, question=question)

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        answer = clean_response(response)
    except Exception as exc:
        print("LLM error:", repr(exc))
        return "I could not generate the answer right now. Please check the Gemini API configuration and try again.", format_sources(documents)

    if not answer:
        answer = "I could not generate a useful answer from the available documents."

    return answer, format_sources(documents)


def process_inputs(files, url=None):
    all_documents = []
    errors = []

    for filename, filepath in files:
        try:
            all_documents.extend(load_file(filepath))
        except Exception as exc:
            errors.append(f"{filename}: {exc}")

    if url:
        try:
            all_documents.extend(load_url(url))
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    if not all_documents:
        return {"documents": 0, "chunks": 0, "errors": errors}

    chunks = split_docs(all_documents)
    stored_chunks = add_to_vectorstore(chunks)

    return {"documents": len(all_documents), "chunks": stored_chunks, "errors": errors}


def get_database_count():
    try:
        db = get_vectorstore()
        return len(db.get().get("ids", []))
    except Exception as exc:
        print("count error:", repr(exc))
        return 0


if __name__ == "__main__":
    print("RAG pipeline is ready.")
    print(f"Data directory: {DATA_DIR}")
    print(f"Chroma directory: {CHROMA_DIR}")
    print(f"API key set: {has_api_key()}")
    print(f"Stored chunks: {get_database_count()}")

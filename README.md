# Multi-Doc RAG Chat

A simple RAG application built with Streamlit, LangChain, Gemini and Chroma.

The app lets you upload PDF/TXT files or provide a webpage URL. The content is processed, split into smaller chunks, and stored in a local Chroma database. You can then ask questions about the uploaded content from the chat interface.

## Features

* Upload PDF and TXT files
* Add a webpage using its URL
* Text extraction and chunking
* Gemini embeddings for document search
* ChromaDB for local vector storage
* Ask questions about the uploaded documents
* Shows the source file and page number when available
* Multiple documents can be processed together
* Chat history is maintained during the current session
* Local vector database, so processed data can remain available after restarting the app

## Project Structure

```text
rag_app/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── .env
├── .env.example
│
├── data/
│   └── uploaded files
│
└── chroma_db/
    └── local vector database
```

## Requirements

* Python 3.10+
* Gemini API key
* Internet connection for Gemini API and webpage loading

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```env
GOOGLE_API_KEY=your_api_key_here
```

Do not commit the `.env` file to Git.

## Run the Application

Start Streamlit with:

```bash
python -m streamlit run app.py
```

The application will open in the browser, normally at:

```text
http://localhost:8501
```

## How to Use

1. Open the application.
2. Upload one or more PDF/TXT files.
3. You can also enter a webpage URL.
4. Click **Process documents**.
5. Wait for the documents to finish processing.
6. Enter a question in the chat box.
7. The application retrieves relevant document chunks and sends the retrieved context to Gemini.
8. The answer is displayed along with the available source information.

## Configuration

The main RAG settings are kept in `rag_pipeline.py`.

The chunk size and overlap can be changed depending on the type of documents being used.

For example:

```python
chunk_size=1000
chunk_overlap=150
```

The number of retrieved chunks can also be adjusted in the retriever settings.

## Vector Database

ChromaDB is stored locally in the `chroma_db` directory.

This means the embeddings do not have to be recreated every time the application starts.

If you want to start with a completely empty database, stop the Streamlit application and delete:

```text
chroma_db/
```

The database will be created again when documents are processed.

## Environment Variables

The application uses:

```text
GOOGLE_API_KEY
```

Keep the API key in `.env` rather than putting it directly in the Python files.

## Current Limitations

* Scanned/image-only PDFs may not contain extractable text and may require OCR.
* Web pages that depend heavily on JavaScript may not load correctly with the basic web loader.
* There is currently no user authentication.
* The vector database is local to the machine.
* Uploading the same document multiple times can create duplicate chunks.
* Changing the embedding model may require creating a fresh Chroma database.

## Notes

This project is mainly intended as a small RAG demonstration rather than a production-ready document management system.

The main parts of the application are separated into two files:

* `app.py` handles the Streamlit interface.
* `rag_pipeline.py` handles document loading, splitting, embeddings, retrieval and Gemini responses.

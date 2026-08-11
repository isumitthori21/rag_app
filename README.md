# Multi-Doc RAG Chat

A small Streamlit app that lets you dump some PDFs, TXT files, or a webpage URL into a Chroma vector store and then chat with them using Gemini. Built this as part of a job assignment, sharing it here in case it's useful as a starting point.

## What it does

- Upload multiple PDFs/TXTs, or paste a URL
- Splits everything with `RecursiveCharacterTextSplitter` and stores embeddings in a local ChromaDB folder (persistent, so it survives restarts)
- Uses `gemini-1.5-flash` for answering questions, `embedding-001` for embeddings
- Shows which file/page an answer came from
- Basic chat history + clear button

Nothing fancy — no reranking, no hybrid search, no query rewriting. It's a working baseline, not a production RAG system.

## File structure

```
rag-app/
├── app.py              # streamlit UI
├── rag_pipeline.py      # loading, chunking, embedding, QA chain
├── requirements.txt
├── .env.example
├── data/                 # uploaded files land here (created automatically)
└── chroma_db/            # persistent vector store (created automatically)
```

## Setup (VS Code)

1. Open the project folder in VS Code (`File > Open Folder`).

2. Open a terminal (`` Ctrl+` `` or `Terminal > New Terminal`) and create a virtual env:

   ```bash
   python -m venv venv
   ```

   Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

   VS Code will usually prompt you to select this venv as the interpreter — say yes (or do it manually via `Ctrl+Shift+P` -> "Python: Select Interpreter").

3. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

4. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

5. Create a `.env` file in the root of the project (right-click in VS Code Explorer -> New File -> name it `.env`). Copy the contents of `.env.example` into it and paste your key:

   ```
   GOOGLE_API_KEY=your_actual_key
   ```

6. Run the app from the terminal:

   ```bash
   streamlit run app.py
   ```

   It should open in your browser at `localhost:8501`.

## Using it

1. Upload files and/or paste a URL, then hit "Process documents". Wait for the success message — bigger PDFs take a bit since it's embedding every chunk.
2. Once documents are processed, just type your question in the chat box at the bottom.
3. Answers show the source file (and page number for PDFs) below the response.
4. "Clear chat" wipes conversation history, not the vector store — your documents stay embedded until you delete the `chroma_db` folder manually.

## Known limitations/things I didn't bother with

- No de-duplication if you upload the same file twice — it'll just add duplicate chunks
- Web scraping via `WebBaseLoader` is pretty basic, doesn't handle JS-heavy sites
- Page numbers only show up for PDFs; obviously TXT/web docs don't have pages
- No auth, no multi-user support, everything is local
- If you want to start fresh, just delete the `chroma_db` and `data` folders and restart the app
- Error handling is minimal on purpose — if Gemini's API key is wrong or you hit a rate limit, you'll see the raw error in the terminal instead of a nice message

## Notes

Retrieval is set to `k=4` in `rag_pipeline.py` — bump that up if you want more context per answer, at the cost of slower/more expensive calls. Chunk size is 1000 with 150 overlap, tweak in `split_docs()` if your docs need bigger/smaller chunks.

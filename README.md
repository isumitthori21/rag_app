# Multi-Doc RAG Chat

A small Streamlit app that lets you dump some PDFs, TXT files, or a webpage URL into a Chroma vector store and then chat with them using Gemini. Built this as part of a job assignment, sharing it here in case it's useful as a starting point.

## What it does

- Upload multiple PDFs/TXTs, or paste a URL
- Splits everything with `RecursiveCharacterTextSplitter` and stores embeddings in a local ChromaDB folder (persistent, so it survives restarts)
- Uses `gemini-3.6-flash` for answering questions, `gemini-embedding-2-preview` for embeddings
- Shows which file/page an answer came from, with a sources expander under each answer
- Chat history is passed back into the prompt, so follow-up questions ("what about that one" etc) actually work
- Sidebar shows stored chunk count + list of sources added this session
- Clear chat button (wipes conversation, not the vector store) and a separate clear source-list button


## File structure

```
rag-app/
├── app.py              
├── rag_pipeline.py     
├── requirements.txt
├── .env.example
├── .gitignore
├── data/                 
└── chroma_db/          
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

   It should open in your browser at `localhost:8501`. If the API key isn't set, the app shows a message and stops instead of crashing.

## Using it

1. Upload files and/or paste a URL, then hit "Process documents". Wait for the status box to finish - bigger PDFs take a bit since it's embedding every chunk.
2. Once documents are processed, just type your question in the chat box at the bottom.
3. Answers show a "Sources" expander below with the file name (and page number for PDFs).
4. "Clear conversation" wipes chat history only. "Clear source list" just clears the sidebar list for this session — it does not touch the vector store.
5. Your documents stay embedded across restarts until you delete the `chroma_db` folder manually.

## Known limitations/things 

- No de-duplication if you upload the same file twice - it'll just add duplicate chunks
- Web scraping via `WebBaseLoader` is pretty basic; it doesn't handle JS-heavy sites
- Page numbers only show up for PDFs; obviously TXT/web docs don't have pages
- No auth, no multi-user support, everything is local
- If you change the embedding model or switch to a different one later, delete the `chroma_db` folder first - Chroma will throw a dimension mismatch error if you mix embeddings from different models in the same collection
- If you want to start fresh, delete the `chroma_db` and `data` folders and restart the app
- Error handling is minimal on purpose - if you hit a Gemini rate limit mid-question, you'll see it as a chat message instead of a crash, but that's about as fancy as it gets

## Notes

Retrieval is set to `k=5` in `rag_pipeline.py` (`TOP_K`) - bump that up if you want more context per answer, at the cost of slower/more expensive calls. Chunk size is 900 with 150 overlap (`CHUNK_SIZE` / `CHUNK_OVERLAP`), tweak those constants if your docs need bigger/smaller chunks.

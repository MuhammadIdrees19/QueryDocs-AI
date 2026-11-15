# QueryDocs AI

Ask questions directly from documents.

## Project Structure

- `src/` - application package
- `data/` - local data files
- `utils/` - helper utilities and scripts

## Setup

1. Create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Extract the PDFs and create chunks:

```bash
python utils/extract_pdf_data.py
python utils/chunks.py
```

The script uses LangChain's `PyPDFLoader` and saves one JSON file per PDF in
`data/process data`. Each page record contains the source filename, 1-based page
number, and extracted text.

The script uses LangChain's `RecursiveCharacterTextSplitter`, preferring
paragraph and line boundaries with no overlap. It saves chunk JSON files in
`data/chunks`, including source, page range, document type, and chunk order.

## Create and Store Embeddings

After creating chunks, add `QDRANT_URL` and `QDRANT_API_KEY` to `.env` and run:

```bash
python src/embedding.py
```

The script uses LangChain's Hugging Face embeddings integration with
`BAAI/bge-m3` locally, then stores new or changed vectors in the existing
`AI _ML`, `Finance`, and `HR` Qdrant collections. The collections must use
1024-dimensional cosine vectors. The embedding model downloads on first use;
the reranker model downloads when the router/retrieval loop starts. No Hugging
Face API key is required. Keep Qdrant credentials private and do not commit
`.env`.

On Windows with an NVIDIA GPU, install the CUDA-enabled PyTorch build after the
standard requirements:

```powershell
pip install -r requirements-gpu.txt
```

The API will use CUDA for embeddings and reranking when available, and fall back
to CPU otherwise. GPU acceleration applies only to these local stages; routing
and answer generation still depend on Groq response times.

## Route Questions and Retrieve Chunks

Run the router/retrieval test loop with:

```bash
python -m src.retrieval
```

The router needs `GROQ_API_KEY` in `.env`.
It uses `openai/gpt-oss-120b` through Groq to select one or more collections. The question is embedded once
with local `BAAI/bge-m3`, those collections are searched in parallel, and
`CrossEncoderReranker` reranks the candidates with the local
`BAAI/bge-reranker-base` model before displaying the best chunks with source
pages. On CUDA both local models load in float16. If the router is uncertain, it searches all three collections. This test
loop does not invoke answer generation.

## Generate an Answer

`src/generation.py` provides `build_answer_chain()` and `generate_answer()`.
Pass the user's question and the reranked `Document` results from
`src/retrieval.py`. The answer chain is LangChain's stuff-documents chain: it
formats each chunk with its source and page label, then sends that context to
the model. Its prompt requests complete process steps and requires unfilled
source placeholders to be identified rather than guessed. It uses the same
Groq model and key as routing.

## FastAPI Interface

Start the API from the project root with:

```bash
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/` to use the chat interface, ask follow-up questions,
and view cited source pages. Re-run `python src/embedding.py` after updating the
chunk files.
Visit `http://127.0.0.1:8000/docs` to explore the API or try `POST /chat`
directly. Set `GROQ_API_KEY` in `.env`; do not share it in chat or commit
it. Sessions are held in memory and reset when the server restarts.

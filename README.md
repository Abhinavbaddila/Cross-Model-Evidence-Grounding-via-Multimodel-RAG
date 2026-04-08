# Cross-Model Evidence Grounding via Multimodal RAG

This project is a grounded multimodal retrieval-augmented generation system with a web interface. It accepts a question and an optional uploaded image, retrieves supporting evidence, and returns an answer with proof.

## Features

- Multimodal input with question plus optional image upload
- Grounded answer generation with proof cards
- Detector-first reasoning for unseen uploaded images
- Similar-image retrieval from the indexed corpus
- PDF, CSV, XLSX, and TXT ingestion
- Programmatic confidence scoring
- Explainable output through retrieved evidence

## Tech Stack

- Backend: FastAPI
- Frontend: React + Vite
- Text embeddings: `BAAI/bge-large-en-v1.5`
- Image embeddings: `openai/clip-vit-base-patch32`
- Vector store: ChromaDB
- Lexical retrieval: BM25
- Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Detector: YOLOv8 by default

## Project Structure

```text
mmrag/
  config.py
  corpus.py
  documents.py
  generation.py
  indexing.py
  schemas.py
  service.py
  vision.py
webapp/
  app.py
  frontend/
run_pipeline.py
start_app.ps1
start_backend.ps1
start_frontend.ps1
stop_backend.ps1
requirements.txt
.env.example
```

## Important Note

This repository intentionally excludes:

- dataset files
- generated vector indexes
- uploaded runtime files
- local virtual environments
- model weight files

These are local assets and should be recreated after cloning.

## Setup

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Backend:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_backend.ps1 -ForceRestart
```

Frontend:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_frontend.ps1
```

Open:

```text
http://127.0.0.1:5173
```

## Manual Run

Backend:

```powershell
$env:PYTHONPATH="$PWD\.venv313\Lib\site-packages;$PWD"
.\.venv313\Scripts\python.exe -m uvicorn webapp.app:app --host 127.0.0.1 --port 9000
```

Frontend:

```powershell
cd webapp\frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

## Environment Variables

See [.env.example](./.env.example) for configuration options.

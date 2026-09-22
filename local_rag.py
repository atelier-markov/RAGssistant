# local_rag.py
# 
# A minimal, fully-local RAG pipeline performing
#   - Embeddings: sentence-transformers (GPU, if available)
#   - Vector indexing: FAISS
#   - LLM synthesis: Ollama (local HTTP server)
# 
# Place internal KB documents in ./docs (supports .txt, .md, .pdf). Then run
#     python local_rag.py --docs ./docs --rebuild
# to build the initial FAISS index.
#
# Subsequent runs reuse the saved index:
#     python local_rag.py --docs ./docs
# 
# Make sure Ollama background service is active before running:
#     $ollama serve
#     $curl http://localhost:11434/api/tags
# 
# Use the --model flag to specify the installed Ollama LLM.
# 
# Default model is set to qwen2.5:7b. To download a new model, use
#       $ollama pull **model_name**
# then run, for example,
#       python local_rag.py --docs ./docs --model mistral
# Find the latest models at https://ollama.com/search
#
# Copyright (c) 2026 Rashid Vladimir Williams-Garcia, Atelier Markov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import os
import json
import textwrap
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# 1. Document loading
# ---------------------------------------------------------------------------

def load_text_file(path:Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_pdf(path:Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))

    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_documents(doc_dir:str) -> List[Dict[str,str]]:
    supported = {".txt", ".md", ".pdf"}
    docs = []

    for path in Path(doc_dir).rglob("*"):
        if path.suffix.lower() not in supported or not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".pdf":
                text = load_pdf(path)
            else:
                text = load_text_file(path)
            if text.strip():
                docs.append({"source": str(path), "text": text})
                print(f"   Loaded {path.name} ({len(text)} chars)")
        except Exception as e:
            print(f"   Skipped {path.name}: {e}")

    return docs

# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

#word-based chunking. for production, consider token-based chunks
def chunk_text(text:str, chunk_size:int = 500, overlap:int = 80) -> List[str]:
    words = text.split()

    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+chunk_size])

        if chunk.strip():
            chunks.append(chunk)

        if i+chunk_size >= len(words):
            break

    return chunks

# build chunks of text,metadata tuples from available documents
def build_chunks(docs:List[Dict[str,str]], chunk_size:int, overlap:int):
    all_chunks, all_meta = [], []

    for doc in docs:
        for j,chunk in enumerate(chunk_text(doc['text'], chunk_size, overlap)):
            all_chunks.append(chunk)
            all_meta.append({"source": doc['source'], "chunk_id": j})

    return all_chunks, all_meta
 
# ---------------------------------------------------------------------------
# 3. Embedding + FAISS indexing
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self, model_name:str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)    #uses CUDA automatically
        print(f"Embedding model loaded on {self.model.device}")
        self.index = None
        self.chunks:List[str] = []
        self.meta:List[Dict] = []

    def build(self, chunks:List[str], meta:List[Dict]):
        print(f"Embedding {len(chunks)} chunks...")

        embeddings = self.model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim) #cosine similarity inner product
        self.index.add(embeddings)
        self.chunks = chunks
        self.meta = meta

        print(f"Index built: {self.index.ntotal} vectors, dim={dim}")

    def search(self, query:str, top_k:int = 4) -> List[Dict]:
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        scores, indices = self.index.search(q_emb, top_k)

        results = []

        for score,idx in zip(scores[0], indices[0]):
            if idx==-1:
                continue
            results.append({
                "text": self.chunks[idx],
                "meta": self.meta[idx],
                "score": float(score)
            })

        return results

    def save(self, path:str):
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "chunks.json"), "w") as file:
            json.dump({"chunks": self.chunks, "meta": self.meta}, file)

    def load(self, path:str):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "chunks.json")) as file:
            data = json.load(file)
        self.chunks = data["chunks"]
        self.meta = data["meta"]

# ---------------------------------------------------------------------------
# 4. LLM call via Ollama
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_llm(prompt:str, model:str = "qwen2.5:7b", temperature:float = 0.2) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "[ERROR] Could not reach Ollama. Is the local server running?"
    except Exception as e:
        return f"[ERROR] {e}"

# ---------------------------------------------------------------------------
# 5. Prompt construction
# ---------------------------------------------------------------------------

def build_manifest(store:VectorStore) -> str:
    from collections import Counter
    counts = Counter(m["source"] for m in store.meta)
    lines = [f"- {Path(src).name} ({n} chunks)" for src, n in sorted(counts.items())]
    return "Documents in the knowledge base:\n"+"\n".join(lines)

SYSTEM_PROMPT = """You are a research assistant. Answer the user's question using
ONLY the context provided below. If the context does not contain the answser, say
so honestly -- do not invent information. Cite sources using [1], [2], etc. when
you use them.

{manifest}

Context:
{context}
"""

def build_prompt(question:str, retrieved:List[Dict], manifest:str) -> str:
    context_parts = []

    for i,r in enumerate(retrieved, 1):
        src = Path(r['meta']['source']).name
        context_parts.append(f"[{i}] (from {src}, chunk {r['meta']['chunk_id']}):\n{r['text']}")

    context = "\n\n".join(context_parts)

    return SYSTEM_PROMPT.format(manifest=manifest, context=context) + f"\n\nQuestion: {question}\nAnswer:"

# ---------------------------------------------------------------------------
# 6. Interactive chat loop
# ---------------------------------------------------------------------------

def chat(store:VectorStore, model:str = "qwen2.5:7b", manifest:str = ""):
    print("\n" + "=" * 60)
    print("Local RAG assistant ready. Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"quit", "exit"}:
            break

        retrieved = store.search(question, top_k=4)
        prompt = build_prompt(question, retrieved, manifest)
        answer = ask_llm(prompt, model=model)

        print("\nAssistant:", answer)
        print("\nSources:")

        for i,r in enumerate(retrieved, 1):
            snippet = textwrap.shorten(r['text'], width=100, placeholder="...")
            print(f"  [{i}] {Path(r['meta']['source']).name} "
                  f"(chunk {r['meta']['chunk_id']}, score={r['score']:.3f})")
            print(f"      {snippet}")
        print()

# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local RAG Research Assistant")
    parser.add_argument("--docs", required=True, help="Directory containing your research documents")
    parser.add_argument("--index", default="./rag_index", help="Where to save/load the FAISS index")
    parser.add_argument("--rebuild", action="store_true", help="Force re-indexing")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name")
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=80)
    args = parser.parse_args()

    store = VectorStore(model_name=args.embed_model)
    index_exists = os.path.exists(os.path.join(args.index, "index.faiss"))

    if index_exists and not args.rebuild:
        print(f"Loading existing index from {args.index}")
        store.load(args.index)
        print(f"Loaded {len(store.chunks)} chunks.")
    else:
        print(f"Loading documents from {args.docs}...")
        docs = load_documents(args.docs)
        if not docs:
            print("No documents found. Exiting.")
            return

        chunks, meta = build_chunks(docs, args.chunk_size, args.overlap)

        store.build(chunks, meta)
        store.save(args.index)

        print(f"Index saved to {args.index}")

    manifest = build_manifest(store)
    print("\nKnowledge base:\n" + manifest + "\n")

    chat(store, model=args.model, manifest=manifest)

if __name__=="__main__":
    main()

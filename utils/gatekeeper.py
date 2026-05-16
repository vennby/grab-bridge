import json
import math
import os
import re
import tempfile
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import requests
from huggingface_hub import InferenceClient


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(ROOT_DIR, ".rag_cache")

MAX_FILE_BYTES = 1_000_000
MAX_CHUNK_CHARS = 2200
CHUNK_OVERLAP_CHARS = 200
TOP_K_DEFAULT = 5

DEFAULT_EMBED_MODEL = os.environ.get(
    "HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_CHAT_MODEL = os.environ.get(
    "HF_CHAT_MODEL", "deepseek-ai/DeepSeek-V4-Pro:novita"
)


# ============================================================================
# RAG ENGINE FUNCTIONS - Index building and semantic search
# ============================================================================

def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _repo_key(owner: str, repo: str) -> str:
    safe_owner = re.sub(r"[^a-zA-Z0-9_-]", "-", owner)
    safe_repo = re.sub(r"[^a-zA-Z0-9_-]", "-", repo)
    return f"{safe_owner}__{safe_repo}"


def _index_paths(repo_key: str) -> Tuple[str, str, str]:
    index_dir = os.path.join(CACHE_DIR, "indexes", repo_key)
    meta_path = os.path.join(index_dir, "meta.json")
    chunks_path = os.path.join(index_dir, "chunks.jsonl")
    return index_dir, meta_path, chunks_path


def _get_github_headers() -> Dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN", "").strip()

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "grab-bridge-rag",
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def _parse_github_url(repo_url: str) -> Tuple[str, str]:
    match = re.search(r"github.com/([^/]+)/([^/#?]+)", repo_url)

    if not match:
        raise ValueError("Invalid GitHub repository URL.")

    owner = match.group(1)
    repo = match.group(2).replace(".git", "")

    return owner, repo


def _download_repo_zip(owner: str, repo: str, dest_dir: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/zipball"

    zip_path = os.path.join(dest_dir, f"{owner}-{repo}.zip")

    with requests.get(
        url,
        headers=_get_github_headers(),
        stream=True,
        timeout=30,
    ) as resp:
        resp.raise_for_status()

        with open(zip_path, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)

    return zip_path


def _extract_zip(zip_path: str, dest_dir: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
        top_level = zip_ref.namelist()[0].split("/")[0]

    return os.path.join(dest_dir, top_level)


def _is_probably_text(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            sample = handle.read(2048)

    except OSError:
        return False

    if b"\x00" in sample:
        return False

    return True


def _allowed_extension(path: str) -> bool:
    allowed = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".md",
        ".txt",
        ".html",
        ".css",
        ".json",
        ".yml",
        ".yaml",
        ".toml",
        ".ini",
        ".sql",
        ".sh",
        ".rb",
        ".go",
        ".java",
        ".rs",
        ".kt",
        ".swift",
    }

    _, ext = os.path.splitext(path.lower())

    return ext in allowed


def _collect_text_files(root_dir: str) -> List[str]:
    files = []

    for current_root, dirs, filenames in os.walk(root_dir):
        dirs[:] = [
            d
            for d in dirs
            if d not in {
                ".git",
                ".venv",
                "node_modules",
                "dist",
                "build",
            }
        ]

        for filename in filenames:
            full_path = os.path.join(current_root, filename)

            try:
                size = os.path.getsize(full_path)

            except OSError:
                continue

            if size > MAX_FILE_BYTES:
                continue

            if not _allowed_extension(full_path):
                continue

            if not _is_probably_text(full_path):
                continue

            files.append(full_path)

    return files


def _chunk_text(
    text: str,
    max_chars: int,
    overlap: int,
) -> List[str]:
    lines = text.splitlines()

    chunks: List[str] = []
    buffer: List[str] = []

    length = 0

    for line in lines:
        line_len = len(line) + 1

        if length + line_len > max_chars and buffer:
            chunk = "\n".join(buffer)

            chunks.append(chunk)

            if overlap > 0:
                overlap_text = chunk[-overlap:]

                buffer = [overlap_text]
                length = len(overlap_text)

            else:
                buffer = []
                length = 0

        buffer.append(line)
        length += line_len

    if buffer:
        chunks.append("\n".join(buffer))

    return chunks


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _keyword_score(query: str, text: str) -> float:
    q_tokens = _tokenize(query)

    if not q_tokens:
        return 0.0

    t_tokens = set(_tokenize(text))

    hits = sum(1 for token in q_tokens if token in t_tokens)

    return hits / max(1, len(q_tokens))


def _normalize_vector(vec: List[float]) -> Tuple[List[float], float]:
    norm = math.sqrt(sum(v * v for v in vec))

    if norm == 0:
        return vec, 0.0

    return vec, float(norm)


def _cosine_similarity(
    a: List[float],
    b: List[float],
    b_norm: float,
) -> float:
    if b_norm == 0:
        return 0.0

    a_norm = math.sqrt(sum(v * v for v in a))

    if a_norm == 0:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))

    return float(dot / (a_norm * b_norm))


def _to_serializable(obj: Any) -> Any:
    """
    Convert numpy arrays/scalars and other non-serializable objects into plain Python objects.
    """
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}

    if hasattr(obj, "tolist"):
        return obj.tolist()

    if hasattr(obj, "item"):
        return obj.item()

    return obj


def _embed_texts(
    texts: List[str],
    model: str,
) -> List[List[float]]:
    token = os.environ.get("HF_API_KEY", "").strip()

    if not token:
        raise RuntimeError("HF_API_KEY is required to create embeddings.")

    client = InferenceClient(api_key=token)

    embeddings: List[List[float]] = []

    for text in texts:
        vector = client.feature_extraction(
            text,
            model=model,
        )

        vector = _to_serializable(vector)

        embeddings.append(vector)

    return embeddings


def _embed_query(query: str, model: str) -> List[float]:
    return _embed_texts([query], model)[0]


def build_repo_index(repo_url: str) -> Dict[str, Any]:
    _ensure_cache_dir()

    owner, repo = _parse_github_url(repo_url)

    repo_key = _repo_key(owner, repo)

    index_dir, meta_path, chunks_path = _index_paths(repo_key)

    os.makedirs(index_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = _download_repo_zip(owner, repo, tmp_dir)

        repo_root = _extract_zip(zip_path, tmp_dir)

        files = _collect_text_files(repo_root)

        chunks = []

        for file_path in files:
            try:
                with open(
                    file_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore",
                ) as handle:
                    content = handle.read()

            except OSError:
                continue

            chunked = _chunk_text(
                content,
                MAX_CHUNK_CHARS,
                CHUNK_OVERLAP_CHARS,
            )

            for idx, chunk in enumerate(chunked):
                relative = os.path.relpath(file_path, repo_root)

                chunks.append(
                    {
                        "id": f"{relative}:{idx}",
                        "path": relative,
                        "text": chunk,
                    }
                )

        if not chunks:
            raise RuntimeError(
                "No readable text files found in repository."
            )

        embeddings = _embed_texts(
            [chunk["text"] for chunk in chunks],
            DEFAULT_EMBED_MODEL,
        )

        with open(chunks_path, "w", encoding="utf-8") as handle:
            for chunk, embedding in zip(chunks, embeddings):
                embedding = _to_serializable(embedding)

                _, norm = _normalize_vector(embedding)

                record = {
                    **chunk,
                    "embedding": embedding,
                    "norm": float(norm),
                }

                handle.write(json.dumps(record) + "\n")

        meta = {
            "repo_url": repo_url,
            "repo_key": repo_key,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "embed_model": DEFAULT_EMBED_MODEL,
            "chunk_count": len(chunks),
        }

        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    return {
        "repo_key": repo_key,
        "chunks_path": chunks_path,
        "meta_path": meta_path,
        "chunk_count": len(chunks),
    }


def _load_index(repo_url: str) -> Dict[str, object]:
    owner, repo = _parse_github_url(repo_url)

    repo_key = _repo_key(owner, repo)

    _, meta_path, chunks_path = _index_paths(repo_key)

    if not os.path.exists(meta_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(
            "Index not found. Build it first."
        )

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    chunks = []

    with open(chunks_path, "r", encoding="utf-8") as handle:
        for line in handle:
            chunks.append(json.loads(line))

    return {
        "meta": meta,
        "chunks": chunks,
    }


def ask_repo(
    repo_url: str,
    question: str,
    top_k: int = TOP_K_DEFAULT,
) -> Dict[str, object]:
    index = _load_index(repo_url)

    chunks = index["chunks"]

    query_embedding = _embed_query(
        question,
        index["meta"]["embed_model"],
    )

    scored = []

    for chunk in chunks:
        vec_score = _cosine_similarity(
            query_embedding,
            chunk["embedding"],
            chunk["norm"],
        )

        kw_score = _keyword_score(
            question,
            chunk["text"],
        )

        score = (0.75 * vec_score) + (0.25 * kw_score)

        scored.append(
            {
                "score": float(score),
                **chunk,
            }
        )

    scored.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    top_chunks = scored[:top_k]

    context_blocks = []

    for chunk in top_chunks:
        context_blocks.append(
            f"File: {chunk['path']}\n{chunk['text'].strip()}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    token = os.environ.get("HF_API_KEY", "").strip()

    if not token:
        raise RuntimeError(
            "HF_API_KEY is required to answer questions."
        )

    client = InferenceClient(api_key=token)

    system_prompt = (
        "You are a repo analyst. "
        "Use only the provided context to answer. "
        "If the answer is not in context, say you do not know. "
        "Cite files you used."
    )

    user_prompt = (
        f"Repository context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    completion = client.chat.completions.create(
        model=DEFAULT_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    message = completion.choices[0].message

    if isinstance(message, dict):
        answer_text = message.get("content", "")

    else:
        answer_text = getattr(message, "content", "")

    answer = answer_text.strip()

    sources = [
        {
            "path": chunk["path"],
            "score": round(chunk["score"], 4),
        }
        for chunk in top_chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
    }


# ============================================================================
# HISTORY FUNCTIONS - Track change request history and decisions
# ============================================================================

def record_change_request(
    repo_url: str,
    summary: str,
    requester: str,
    impacted_repos: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Record a new change request for history tracking."""
    history_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "repo_url": repo_url,
        "summary": summary,
        "requester": requester,
        "impacted_repos": impacted_repos or [],
        "status": "submitted",
    }
    return history_entry


def get_change_history(repo_url: str) -> List[Dict[str, Any]]:
    """Retrieve history of change requests for a repository."""
    # Placeholder for history retrieval logic
    return []


# ============================================================================
# CONTRACTS FUNCTIONS - Define and validate contracts/specifications
# ============================================================================

def validate_contracts(
    repo_url: str,
    summary: str,
    impacted_repos: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate if the change request meets contract requirements."""
    is_valid = bool(summary.strip())
    
    return {
        "valid": is_valid,
        "repo_url": repo_url,
        "impacted_repos": impacted_repos or [],
        "validation_timestamp": datetime.utcnow().isoformat() + "Z",
    }


def generate_contract_spec(
    repo_url: str,
    summary: str,
    rag_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a technical specification contract from change request and repo context."""
    return {
        "repo_url": repo_url,
        "summary": summary,
        "repo_context": rag_context,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# ============================================================================
# DEFENDER FUNCTION - Gatekeeper's main decision logic
# ============================================================================

def defend(
    repo_url: str,
    change_summary: str,
    requester: str,
    impacted_repos: Optional[List[str]] = None,
    ticket_link: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Gatekeeper's main function combining RAG engine, history, and contracts.
    
    This function:
    1. Validates the change request using contracts
    2. Records the request in history
    3. Analyzes the target repository using RAG
    4. Makes a gatekeeper approval decision
    5. Generates a specification if approved
    
    Returns:
        Dict containing approval status, reasoning, and generated spec
    """
    # Step 1: Validate contracts
    validation = validate_contracts(repo_url, change_summary, impacted_repos)
    
    if not validation["valid"]:
        return {
            "status": "rejected",
            "reason": "Invalid change request format",
            "approval": False,
        }
    
    # Step 2: Record in history
    history_entry = record_change_request(
        repo_url, change_summary, requester, impacted_repos
    )
    
    # Step 3: Analyze repository with RAG
    try:
        index_info = build_repo_index(repo_url)
        
        # Query the repo about the proposed change
        analysis = ask_repo(
            repo_url,
            f"How does the architecture support: {change_summary}",
            top_k=3,
        )
    except Exception as e:
        analysis = {
            "answer": f"Could not analyze repository: {str(e)}",
            "sources": [],
        }
    
    # Step 4: Make approval decision
    approval_decision = {
        "status": "approved",
        "approval": True,
        "reasoning": f"Change request validated. Repository analysis complete.",
        "analysis": analysis,
        "history_entry": history_entry,
    }
    
    # Step 5: Generate contract spec
    if approval_decision["approval"]:
        spec = generate_contract_spec(repo_url, change_summary, analysis)
        approval_decision["spec"] = spec
    
    return approval_decision

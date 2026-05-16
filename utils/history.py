import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

import requests
from huggingface_hub import InferenceClient


DEFAULT_EMBED_MODEL = os.environ.get(
    "HF_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

DEFAULT_CHAT_MODEL = os.environ.get(
    "HF_CHAT_MODEL",
    "deepseek-ai/DeepSeek-V4-Pro:novita",
)

TOP_K_DEFAULT = 8


# =========================================================
# GitHub helpers
# =========================================================

def _parse_github_url(repo_url: str) -> Tuple[str, str]:
    match = re.search(r"github.com/([^/]+)/([^/#?]+)", repo_url)

    if not match:
        raise ValueError("Invalid GitHub repository URL.")

    owner = match.group(1)
    repo = match.group(2).replace(".git", "")

    return owner, repo


def _get_github_headers() -> Dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN", "").strip()

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "github-activity-rag",
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


# =========================================================
# Serialization helpers
# =========================================================

def _to_serializable(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()

    if hasattr(obj, "item"):
        return obj.item()

    return obj


# =========================================================
# Embeddings
# =========================================================

def _embed_texts(
    texts: List[str],
    model: str,
) -> List[List[float]]:

    token = os.environ.get("HF_API_KEY", "").strip()

    if not token:
        raise RuntimeError("HF_API_KEY is required.")

    client = InferenceClient(api_key=token)

    vectors = []

    for text in texts:
        vec = client.feature_extraction(
            text,
            model=model,
        )

        vec = _to_serializable(vec)

        vectors.append(vec)

    return vectors


def _embed_query(query: str, model: str) -> List[float]:
    return _embed_texts([query], model)[0]


# =========================================================
# Vector math
# =========================================================

def _cosine_similarity(a, b):
    import math

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# =========================================================
# GitHub data collection
# =========================================================

def _fetch_open_issues(
    owner: str,
    repo: str,
    limit: int = 30,
) -> List[Dict]:

    url = f"https://api.github.com/repos/{owner}/{repo}/issues"

    params = {
        "state": "open",
        "per_page": limit,
    }

    resp = requests.get(
        url,
        headers=_get_github_headers(),
        params=params,
        timeout=30,
    )

    resp.raise_for_status()

    items = resp.json()

    results = []

    for item in items:

        # GitHub returns PRs in issues API too
        if "pull_request" in item:
            continue

        results.append(
            {
                "type": "issue",
                "id": item["number"],
                "title": item["title"],
                "body": item.get("body", "") or "",
                "author": item["user"]["login"],
                "created_at": item["created_at"],
                "url": item["html_url"],
                "labels": [
                    x["name"]
                    for x in item.get("labels", [])
                ],
            }
        )

    return results


def _fetch_recent_prs(
    owner: str,
    repo: str,
    limit: int = 30,
) -> List[Dict]:

    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"

    params = {
        "state": "closed",
        "sort": "updated",
        "direction": "desc",
        "per_page": limit,
    }

    resp = requests.get(
        url,
        headers=_get_github_headers(),
        params=params,
        timeout=30,
    )

    resp.raise_for_status()

    items = resp.json()

    results = []

    for item in items:

        merged = item.get("merged_at")

        if not merged:
            continue

        results.append(
            {
                "type": "pull_request",
                "id": item["number"],
                "title": item["title"],
                "body": item.get("body", "") or "",
                "author": item["user"]["login"],
                "merged_at": merged,
                "url": item["html_url"],
            }
        )

    return results


def _fetch_recent_commits(
    owner: str,
    repo: str,
    limit: int = 30,
) -> List[Dict]:

    url = f"https://api.github.com/repos/{owner}/{repo}/commits"

    params = {
        "per_page": limit,
    }

    resp = requests.get(
        url,
        headers=_get_github_headers(),
        params=params,
        timeout=30,
    )

    resp.raise_for_status()

    items = resp.json()

    results = []

    for item in items:
        commit = item["commit"]

        results.append(
            {
                "type": "commit",
                "sha": item["sha"],
                "message": commit["message"],
                "author": (
                    commit.get("author", {}).get("name")
                    or "unknown"
                ),
                "date": (
                    commit.get("author", {}).get("date")
                ),
                "url": item["html_url"],
            }
        )

    return results


# =========================================================
# Build activity index
# =========================================================

def build_repo_activity_index(
    repo_url: str,
) -> Dict[str, object]:

    owner, repo = _parse_github_url(repo_url)

    issues = _fetch_open_issues(owner, repo)
    prs = _fetch_recent_prs(owner, repo)
    commits = _fetch_recent_commits(owner, repo)

    documents = []

    # ---------------------------------------------
    # Issues
    # ---------------------------------------------

    for issue in issues:

        text = f"""
        OPEN ISSUE

        Issue Number: {issue['id']}
        Title: {issue['title']}
        Author: {issue['author']}
        Labels: {', '.join(issue['labels'])}

        Body:
        {issue['body']}
        """

        documents.append(
            {
                "type": "issue",
                "source": issue,
                "text": text.strip(),
            }
        )

    # ---------------------------------------------
    # PRs
    # ---------------------------------------------

    for pr in prs:

        text = f"""
        MERGED PULL REQUEST

        PR Number: {pr['id']}
        Title: {pr['title']}
        Author: {pr['author']}
        Merged At: {pr['merged_at']}

        Description:
        {pr['body']}
        """

        documents.append(
            {
                "type": "pull_request",
                "source": pr,
                "text": text.strip(),
            }
        )

    # ---------------------------------------------
    # Commits
    # ---------------------------------------------

    for commit in commits:

        text = f"""
        COMMIT

        SHA: {commit['sha']}
        Author: {commit['author']}
        Date: {commit['date']}

        Message:
        {commit['message']}
        """

        documents.append(
            {
                "type": "commit",
                "source": commit,
                "text": text.strip(),
            }
        )

    embeddings = _embed_texts(
        [doc["text"] for doc in documents],
        DEFAULT_EMBED_MODEL,
    )

    indexed_docs = []

    for doc, embedding in zip(documents, embeddings):

        indexed_docs.append(
            {
                **doc,
                "embedding": embedding,
            }
        )

    return {
        "repo_url": repo_url,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "document_count": len(indexed_docs),
        "documents": indexed_docs,
    }


# =========================================================
# Ask activity questions
# =========================================================

def ask_repo_activity(
    repo_url: str,
    question: str,
    top_k: int = TOP_K_DEFAULT,
) -> Dict[str, object]:

    index = build_repo_activity_index(repo_url)

    docs = index["documents"]

    query_embedding = _embed_query(
        question,
        DEFAULT_EMBED_MODEL,
    )

    scored = []

    for doc in docs:

        similarity = _cosine_similarity(
            query_embedding,
            doc["embedding"],
        )

        scored.append(
            {
                "score": similarity,
                **doc,
            }
        )

    scored.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    top_docs = scored[:top_k]

    context = "\n\n---\n\n".join(
        doc["text"]
        for doc in top_docs
    )

    token = os.environ.get("HF_API_KEY", "").strip()

    if not token:
        raise RuntimeError(
            "HF_API_KEY is required."
        )

    client = InferenceClient(api_key=token)

    system_prompt = """
    You are a GitHub repository activity analyst.

    Use ONLY the provided context.

    You answer questions about:
    - open issues
    - merged pull requests
    - recent commits
    - development activity
    - contributor behavior
    - roadmap direction
    - bug trends

    If information is unavailable, say so.

    Cite issue numbers, PR numbers, or commit SHAs when possible.
    """

    user_prompt = f"""
    Repository Activity Context:

    {context}

    Question:
    {question}

    Answer:
    """

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
        answer = message.get("content", "").strip()

    else:
        answer = getattr(message, "content", "").strip()

    sources = []

    for doc in top_docs:

        src = doc["source"]

        source_info = {
            "type": doc["type"],
            "score": round(doc["score"], 4),
        }

        if doc["type"] in {"issue", "pull_request"}:
            source_info["id"] = src["id"]
            source_info["title"] = src["title"]

        elif doc["type"] == "commit":
            source_info["sha"] = src["sha"][:8]
            source_info["message"] = src["message"]

        sources.append(source_info)

    return {
        "answer": answer,
        "sources": sources,
    }


# =========================================================
# Example usage
# =========================================================

if __name__ == "__main__":

    repo = "https://github.com/vennby/hsbc-hackathon"

    result = ask_repo_activity(
        repo,
        "What are the major recent bug fixes and what areas are actively being developed?",
    )

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nSOURCES:\n")
    print(json.dumps(result["sources"], indent=2))
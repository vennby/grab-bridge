import os
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from utils import rag_engine

load_dotenv()

app = Flask(__name__)

APP_NAME = "Grab Bridge"
ORG_NAME = "grab"
CACHE_TTL = timedelta(minutes=5)
_cached = {"expires_at": datetime.min, "data": []}


def _github_headers():
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "codegrab-ui",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_repos():
    if datetime.utcnow() < _cached["expires_at"]:
        return _cached["data"]

    repos = []
    page = 1
    while True:
        resp = requests.get(
            f"https://api.github.com/orgs/{ORG_NAME}/repos",
            headers=_github_headers(),
            params={"per_page": 100, "page": page, "sort": "updated"},
            timeout=15,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        repos.extend(batch)
        page += 1

    curated = [
        {
            "name": repo.get("name"),
            "full_name": repo.get("full_name"),
            "html_url": repo.get("html_url"),
            "description": repo.get("description"),
            "stargazers_count": repo.get("stargazers_count"),
            "updated_at": repo.get("updated_at"),
        }
        for repo in repos
    ]

    _cached["data"] = curated
    _cached["expires_at"] = datetime.utcnow() + CACHE_TTL
    return curated


@app.route("/")
def index():
    return render_template("index.html", org_name=ORG_NAME, app_name=APP_NAME)


@app.route("/docs")
def docs():
    return render_template("docs.html", org_name=ORG_NAME, app_name=APP_NAME)


@app.route("/bridge")
def bridge():
    return render_template("bridge.html", org_name=ORG_NAME, app_name=APP_NAME)


@app.route("/api/repos")
def repos():
    try:
        data = _fetch_repos()
        return jsonify({"org": ORG_NAME, "repos": data})
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response else 500
        return jsonify({"error": "GitHub API request failed", "status": status}), status
    except requests.RequestException:
        return jsonify({"error": "GitHub API request failed", "status": 502}), 502


@app.route("/api/rag/index", methods=["POST"])
def rag_index():
    payload = request.get_json(silent=True) or {}
    repo_url = (payload.get("repo_url") or "").strip()
    if not repo_url:
        return jsonify({"error": "repo_url is required"}), 400

    try:
        result = rag_engine.build_repo_index(repo_url)
        return jsonify({"status": "indexed", **result})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except requests.RequestException:
        return jsonify({"error": "GitHub request failed"}), 502
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/rag/ask", methods=["POST"])
def rag_ask():
    payload = request.get_json(silent=True) or {}
    repo_url = (payload.get("repo_url") or "").strip()
    question = (payload.get("question") or "").strip()
    if not repo_url or not question:
        return jsonify({"error": "repo_url and question are required"}), 400

    try:
        result = rag_engine.ask_repo(repo_url, question)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Index not found. Run /api/rag/index first."}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

import os
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from utils import gatekeeper

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
        result = gatekeeper.build_repo_index(repo_url)
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
        result = gatekeeper.ask_repo(repo_url, question)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Index not found. Run /api/rag/index first."}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/gatekeeper/request", methods=["POST"])
def gatekeeper_request():
    """
    Step 1: User submits change request with optional impacted repos.
    This triggers the architect analysis phase.
    """
    payload = request.get_json(silent=True) or {}
    
    repo_url = (payload.get("repo_url") or "").strip()
    change_request = (payload.get("change_request") or "").strip()
    impacted_repos = (payload.get("impacted_repos") or "").strip()
    requester = (payload.get("requester") or "").strip()
    ticket_link = (payload.get("ticket_link") or "").strip()
    
    if not repo_url or not change_request:
        return jsonify({"error": "repo_url and change_request are required"}), 400

    try:
        # Initialize the defender agent
        defender = gatekeeper.RepoDefenderAgent(repo_url)
        
        # Phase 1: Architect analyzes the change
        analysis = defender.analyze_change_request(change_request)

        # Precompute the gatekeeper's defensive questions so Stage 3 can load
        # immediately without another network round-trip.
        defensive_review = defender.generate_defensive_questions(change_request)
        
        # Find affected repositories
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        affected_repos = defender._find_affected_repositories(change_request, repo_name)
        
        return jsonify({
            "status": "request_received",
            "repo_url": repo_url,
            "change_request": change_request,
            "impacted_repos": impacted_repos if impacted_repos else None,
            "requester": requester if requester else None,
            "ticket_link": ticket_link if ticket_link else None,
            "architect_analysis": {
                "architecture": analysis["architecture"],
            },
            "defensive_questions": defensive_review.get("questions", []),
            "affected_repos": affected_repos,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Unexpected error: {str(exc)}"}), 500


@app.route("/api/gatekeeper/respond", methods=["POST"])
def gatekeeper_respond():
    """
    Step 2: Gatekeeper agent responds to developer answers,
    potentially asks follow-up questions, and evaluates safety.
    """
    payload = request.get_json(silent=True) or {}
    
    repo_url = (payload.get("repo_url") or "").strip()
    change_request = (payload.get("change_request") or "").strip()
    developer_answers = payload.get("developer_answers") or {}
    
    if not repo_url or not change_request:
        return jsonify({"error": "repo_url and change_request are required"}), 400

    try:
        defender = gatekeeper.RepoDefenderAgent(repo_url)
        
        # Evaluate change safety based on developer answers
        evaluation = defender.evaluate_change_safety(
            change_request,
            developer_answers
        )
        
        return jsonify({
            "status": "evaluated",
            "repo_url": repo_url,
            "change_request": change_request,
            "developer_answers": developer_answers,
            "gatekeeper_evaluation": evaluation,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Unexpected error: {str(exc)}"}), 500


@app.route("/api/gatekeeper/approve", methods=["POST"])
def gatekeeper_approve():
    """
    Step 3: If approved, generate collaborative technical documentation
    """
    payload = request.get_json(silent=True) or {}
    
    repo_url = (payload.get("repo_url") or "").strip()
    change_request = (payload.get("change_request") or "").strip()
    evaluation = payload.get("evaluation") or {}
    conversation_history = payload.get("conversation_history") or []
    
    if not repo_url or not change_request:
        return jsonify({"error": "repo_url and change_request are required"}), 400
    
    if evaluation.get("decision") != "APPROVE":
        return jsonify({
            "error": "Cannot generate spec for non-approved changes",
            "decision": evaluation.get("decision")
        }), 400

    try:
        defender = gatekeeper.RepoDefenderAgent(repo_url)
        
        # Get context for technical spec generation
        context = defender.analyze_change_request(change_request)
        
        # Build conversation summary for context
        conversation_summary = ""
        if conversation_history:
            conversation_parts = []
            for msg in conversation_history:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                conversation_parts.append(f"**{role}**: {content}")
            conversation_summary = "\n\n".join(conversation_parts)
        
        # Generate technical specification
        system_prompt = """
        You are a senior staff engineer working with the Gatekeeper.
        
        Your job is to create a detailed, implementable technical specification
        for the approved change.
        
        Include:
        - Implementation approach
        - Files to modify/create
        - API changes (if any)
        - Database migrations (if any)
        - Testing strategy
        - Deployment steps
        - Rollback procedure
        - Monitoring/observability additions
        - Risk mitigation steps
        """
        
        user_prompt = f"""
        =====================================
        APPROVED CHANGE REQUEST
        =====================================
        
        {change_request}
        
        =====================================
        REPOSITORY ARCHITECTURE
        =====================================
        
        {context["architecture"]["answer"]}
        
        =====================================
        GATEKEEPER & ARCHITECT REVIEW
        =====================================
        
        {conversation_summary}
        
        Create a detailed technical specification for implementing this change.
        """
        
        spec = defender._chat(system_prompt, user_prompt)
        
        return jsonify({
            "status": "approved",
            "repo_url": repo_url,
            "technical_specification": spec,
            "next_steps": [
                "Review the technical specification",
                "Create a feature branch",
                "Implement according to the spec",
                "Run all tests",
                "Create a pull request",
                "Deploy following the documented procedure"
            ]
        })
    except Exception as exc:
        return jsonify({"error": f"Unexpected error: {str(exc)}"}), 500


@app.route("/api/gatekeeper/questions", methods=["POST"])
def gatekeeper_questions():
    """
    Get all 3 defensive questions at once (parallel loading)
    """
    payload = request.get_json(silent=True) or {}
    
    repo_url = (payload.get("repo_url") or "").strip()
    change_request = (payload.get("change_request") or "").strip()
    
    if not repo_url or not change_request:
        return jsonify({"error": "repo_url and change_request are required"}), 400

    try:
        defender = gatekeeper.RepoDefenderAgent(repo_url)
        result = defender.generate_defensive_questions(change_request)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Unexpected error: {str(exc)}"}), 500


@app.route("/api/gatekeeper/evaluate-answers", methods=["POST"])
def evaluate_answers():
    """
    Evaluate all user answers and generate confidence score
    """
    payload = request.get_json(silent=True) or {}
    
    repo_url = (payload.get("repo_url") or "").strip()
    change_request = (payload.get("change_request") or "").strip()
    user_answers = payload.get("user_answers") or []
    
    if not repo_url or not change_request:
        return jsonify({"error": "repo_url and change_request are required"}), 400

    try:
        defender = gatekeeper.RepoDefenderAgent(repo_url)
        result = defender.evaluate_user_answers(change_request, user_answers)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Unexpected error: {str(exc)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

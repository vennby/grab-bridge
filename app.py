import os
import re
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


def _slugify(value: str, max_length: int = 48) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    if not cleaned:
        return "change-request"
    return cleaned[:max_length].strip("-") or "change-request"


def _github_request(method: str, path: str, **kwargs):
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required to create specification pull requests.")

    headers = kwargs.pop("headers", {})
    merged_headers = {**_github_headers(), **headers}
    response = requests.request(
        method,
        f"https://api.github.com{path}",
        headers=merged_headers,
        timeout=20,
        **kwargs,
    )
    response.raise_for_status()
    if response.content:
        return response.json()
    return {}


def _create_spec_issue(full_repo_name: str, change_request: str, spec_markdown: str):
    slug = _slugify(change_request)
    issue = _github_request(
        "POST",
        f"/repos/{full_repo_name}/issues",
        json={
            "title": f"Gatekeeper technical specification: {change_request[:80]}",
            "body": (
                "This issue was opened automatically by Grab Bridge to publish the "
                "gatekeeper-approved technical specification for the requested change.\n\n"
                f"Requested change slug: `{slug}`\n\n"
                "---\n\n"
                f"{spec_markdown}"
            ),
        },
    )

    return {
        "repo": full_repo_name,
        "issue_url": issue.get("html_url"),
        "issue_number": issue.get("number"),
        "title": issue.get("title"),
    }


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


@app.route("/api/gatekeeper/trial/start", methods=["POST"])
def gatekeeper_trial_start():
    """
    Start the gatekeeper's three-question trial and return the first
    architect draft for user review.
    """
    payload = request.get_json(silent=True) or {}

    repo_url = (payload.get("repo_url") or "").strip()
    change_request = (payload.get("change_request") or "").strip()
    questions = payload.get("questions") or []

    if not repo_url or not change_request:
        return jsonify({"error": "repo_url and change_request are required"}), 400

    try:
        defender = gatekeeper.RepoDefenderAgent(repo_url)
        result = defender.start_gatekeeper_trial(change_request, questions)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Unexpected error: {str(exc)}"}), 500


@app.route("/api/gatekeeper/trial/respond", methods=["POST"])
def gatekeeper_trial_respond():
    """
    Score one architect answer, update confidence, and either return the next
    question draft or the final pass/fail result.
    """
    payload = request.get_json(silent=True) or {}

    repo_url = (payload.get("repo_url") or "").strip()
    change_request = (payload.get("change_request") or "").strip()
    question = payload.get("question") or {}
    edited_answer = (payload.get("edited_answer") or "").strip()
    answer_history = payload.get("answer_history") or []
    questions = payload.get("questions") or []

    if not repo_url or not change_request:
        return jsonify({"error": "repo_url and change_request are required"}), 400

    if not isinstance(question, dict) or not (question.get("question") or "").strip():
        return jsonify({"error": "question is required"}), 400

    if not edited_answer:
        return jsonify({"error": "edited_answer is required"}), 400

    try:
        defender = gatekeeper.RepoDefenderAgent(repo_url)
        result = defender.review_trial_answer(
            change_request,
            question,
            edited_answer,
            answer_history,
            questions,
        )
        return jsonify(result)
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
    affected_repos = payload.get("affected_repos") or []
    
    if not repo_url or not change_request:
        return jsonify({"error": "repo_url and change_request are required"}), 400
    
    if isinstance(affected_repos, str):
        affected_repos = [affected_repos] if affected_repos.strip() else []

    decision = evaluation.get("decision")
    approval_decision = evaluation.get("approval_decision")
    if decision not in {"APPROVE", "PASS"} and approval_decision != "APPROVE":
        return jsonify({
            "error": "Cannot generate spec for non-approved changes",
            "decision": decision
        }), 400

    try:
        defender = gatekeeper.RepoDefenderAgent(repo_url)
        primary_repo = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        guarded_repos = [primary_repo]
        for repo_name in affected_repos:
            if repo_name and repo_name not in guarded_repos:
                guarded_repos.append(repo_name)
        
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

        affected_repos_text = ", ".join(guarded_repos)
        guarded_repos_block = "\n".join(f"- {repo_name}" for repo_name in guarded_repos)
        key_files = context.get("key_files", [])
        key_files_text = "\n".join(f"- {file_path}" for file_path in key_files[:8])
        if not key_files_text:
            key_files_text = "- No specific files were identified from the local index."
        
        # Generate technical specification
        system_prompt = """
        You are a senior staff engineer working with the Gatekeeper.
        
        Your job is to create a detailed, implementable technical specification
        for the approved change.

        The gatekeeper only approves work inside the guarded repositories.
        You must propose fixes and implementation steps that can be performed
        inside those repositories, and they must remain consistent with what
        the architect and gatekeeper already agreed in the review conversation.

        Do not invent unrelated repositories, teams, or systems.
        If the review conversation leaves something uncertain, call it out as an
        assumption or follow-up item instead of pretending it is settled.

        Output a polished Markdown specification with these sections:
        - Summary
        - Repository Scope
        - Agreed Change Plan
        - Repository-by-Repository Work
        - Validation Plan
        - Deployment and Rollback
        - Open Questions
        
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
        AFFECTED REPOSITORIES
        =====================================

        {affected_repos_text}

        =====================================
        GUARDED REPOSITORY SCOPE
        =====================================

        {guarded_repos_block}

        =====================================
        KEY FILES IDENTIFIED FROM LOCAL INDEX
        =====================================

        {key_files_text}

        =====================================
        REPOSITORY ARCHITECTURE
        =====================================
        
        {context["architecture"]["answer"]}
        
        =====================================
        GATEKEEPER & ARCHITECT REVIEW
        =====================================
        
        {conversation_summary}
        
        Create a detailed technical specification for implementing this change.
        If multiple repositories may be affected, call out the repository-specific
        work split and any cross-repository contracts that must stay aligned.
        Every fix you recommend must be something the guarded repositories can
        actually implement. Tie the plan back to the review conversation.
        """

        try:
            spec = defender._chat(system_prompt, user_prompt).strip()
        except Exception:
            spec = ""

        if len(spec) < 120:
            repository_work = []
            for repo_name in guarded_repos:
                if repo_name == primary_repo:
                    repo_files = key_files[:6]
                else:
                    repo_files = []

                file_list = "\n".join(
                    f"- Review and update {file_path}" for file_path in repo_files
                )
                if not file_list:
                    file_list = "- Review the repository surfaces that implement or consume the agreed change contract."

                repository_work.append(
                    f"### {repo_name}\n"
                    f"- Apply the agreed change within this guarded repository.\n"
                    f"- Keep behavior aligned with the architect and gatekeeper review.\n"
                    f"- Validate cross-repository assumptions before merge.\n"
                    f"{file_list}"
                )

            open_questions = "- Confirm any unresolved assumptions from the gatekeeper review before implementation."
            if not conversation_summary:
                open_questions = "- No detailed review transcript was available; confirm implementation details with the gatekeeper before merge."

            spec = (
                "## Summary\n"
                f"Implement the requested change inside the guarded repository scope: {affected_repos_text}.\n\n"
                "## Repository Scope\n"
                f"{guarded_repos_block}\n\n"
                "## Agreed Change Plan\n"
                f"- Requested change: {change_request}\n"
                "- Keep the blast radius limited to the files and interfaces identified during analysis.\n"
                "- Preserve compatibility across any affected repositories and contracts surfaced during review.\n\n"
                "## Repository-by-Repository Work\n"
                f"{'\n\n'.join(repository_work)}\n\n"
                "## Validation Plan\n"
                "- Add or update unit tests for the directly changed code paths.\n"
                "- Add integration or contract checks where repository boundaries are touched.\n"
                "- Re-run the review scenarios that the gatekeeper focused on during the trial.\n\n"
                "## Deployment and Rollback\n"
                "- Roll out the primary repository first unless a shared contract requires coordinated deployment.\n"
                "- Monitor the user-facing or contract-facing behavior identified in the analysis.\n"
                "- Prepare a rollback that reverts the guarded repositories to the last known compatible state.\n\n"
                "## Open Questions\n"
                f"{open_questions}"
            )

        github_issues = []
        github_sync = {
            "status": "skipped",
            "message": "GITHUB_TOKEN is not configured, so specification issues were not created.",
        }

        if os.environ.get("GITHUB_TOKEN", "").strip():
            github_sync = {
                "status": "created",
                "message": "Specification issues were opened for the guarded repositories.",
            }
            issue_errors = []
            for repo_name in guarded_repos:
                try:
                    github_issues.append(
                        _create_spec_issue(
                            f"{ORG_NAME}/{repo_name}",
                            change_request,
                            spec,
                        )
                    )
                except Exception as exc:
                    issue_errors.append(f"{repo_name}: {str(exc)}")

            if issue_errors and github_issues:
                github_sync = {
                    "status": "partial",
                    "message": "Some specification issues were created, but one or more repositories failed.",
                    "errors": issue_errors,
                }
            elif issue_errors:
                github_sync = {
                    "status": "failed",
                    "message": "Specification issues could not be created.",
                    "errors": issue_errors,
                }
        
        return jsonify({
            "status": "approved",
            "repo_url": repo_url,
            "technical_specification": spec,
            "github_sync": github_sync,
            "issues": github_issues,
            "next_steps": [
                "Review the technical specification",
                "Review the generated specification issues on GitHub",
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

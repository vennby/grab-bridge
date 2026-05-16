"""
Gatekeeper Defender Agent

This is a consolidation of three core modules:
- RAG Engine (ask_repo): Repository architecture & codebase analysis
- History (ask_repo_activity): Repository activity, issues, PRs, commits
- Contracts: CI/CD, API contracts, dependencies, service topology

The RepoDefenderAgent orchestrates all three to provide:
- Comprehensive architectural understanding
- Historical context & risk patterns
- Runtime intelligence & contract analysis
- Defensive question generation
- Safety evaluation for proposed changes
"""

import json
import os
import re
from typing import Dict, List, Tuple

from huggingface_hub import InferenceClient

from . import rag_engine
from . import history
from . import contracts

DEFAULT_CHAT_MODEL = os.environ.get(
    "HF_CHAT_MODEL",
    "deepseek-ai/DeepSeek-V4-Pro:novita",
)


# =========================================================
# Repo Defender Agent
# =========================================================

class RepoDefenderAgent:
    """
    AI orchestrator agent that protects a repository
    from unsafe or incomplete changes.

    The agent consolidates:
    - Repository architecture understanding (rag_engine)
    - Repository activity/history (history)
    - API contracts & runtime intelligence (contracts)

    The agent:
    - understands repository architecture
    - understands repo activity/history/issues
    - understands API contracts & deployment
    - asks defensive questions
    - identifies risk areas
    - checks likely breakages
    - blocks ambiguous changes
    """

    def __init__(self, repo_url: str):

        self.repo_url = repo_url

        token = os.environ.get("HF_API_KEY", "").strip()

        if not token:
            raise RuntimeError(
                "HF_API_KEY is required."
            )

        self.client = InferenceClient(api_key=token)

    # =====================================================
    # Internal helpers
    # =====================================================

    def _parse_repo_url(self, repo_url: str) -> Tuple[str, str]:
        """Parse GitHub URL to owner and repo name."""
        match = re.search(r"github.com/([^/]+)/([^/#?]+)", repo_url)
        if not match:
            raise ValueError("Invalid GitHub repository URL.")
        owner = match.group(1)
        repo = match.group(2).replace(".git", "")
        return owner, repo

    def _load_function_index(self) -> List[Dict]:
        """Load the pre-built grab function index."""
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        func_index_path = os.path.join(root_dir, "grab_function_index.jsonl")
        
        if not os.path.exists(func_index_path):
            return []
        
        functions = []
        try:
            with open(func_index_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        functions.append(json.loads(line))
        except Exception:
            return []
        
        return functions

    def _get_repo_functions_from_index(self, repo_name: str) -> List[Dict]:
        """Extract all functions for a specific repo from the index."""
        all_functions = self._load_function_index()
        repo_functions = [f for f in all_functions if f.get("repo") == repo_name]
        return repo_functions

    def _find_affected_repositories(self, change_request: str, source_repo: str) -> List[str]:
        """
        Identify other repositories that would be affected by this change.
        Uses keyword matching and function call patterns from the index.
        """
        try:
            owner, repo = self._parse_repo_url(self.repo_url)
        except ValueError:
            return []
        
        all_functions = self._load_function_index()
        
        # Get unique repos
        unique_repos = set(f.get("repo") for f in all_functions if f.get("repo"))
        unique_repos.discard(repo)  # Remove the source repo
        
        # Tokenize the change request for matching
        change_tokens = set(re.findall(r'\b\w+\b', change_request.lower()))
        
        affected = []
        
        for other_repo in unique_repos:
            repo_funcs = [f for f in all_functions if f.get("repo") == other_repo]
            
            # Check function names and files for matching keywords
            match_score = 0
            for func in repo_funcs:
                sig = (func.get("signature", "") + " " + func.get("file", "")).lower()
                for token in change_tokens:
                    if len(token) > 3 and token in sig:  # Only significant tokens
                        match_score += 1
            
            # If there's reasonable matching, it's affected
            if match_score > 2:
                affected.append(other_repo)
        
        return sorted(affected)[:10]  # Limit to top 10


    def _format_function_index_summary(self, functions: List[Dict], question: str) -> str:
        """Format function index data as a summary document."""
        if not functions:
            return "No functions found in local index."
        
        # Group by file
        by_file = {}
        for func in functions:
            file_path = func.get("file", "unknown")
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append(func)
        
        summary = f"Found {len(functions)} functions across {len(by_file)} files:\n\n"
        
        for file_path in sorted(by_file.keys())[:10]:  # Limit to first 10 files
            funcs = by_file[file_path]
            summary += f"**{file_path}** ({len(funcs)} functions)\n"
            for func in funcs[:5]:  # Show first 5 functions per file
                summary += f"  - {func.get('signature', 'unknown')}\n"
            if len(funcs) > 5:
                summary += f"  ... and {len(funcs) - 5} more\n"
            summary += "\n"
        
        return summary

    # =====================================================
    # Internal LLM helper
    # =====================================================

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        completion = self.client.chat.completions.create(
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
            return message.get("content", "").strip()

        return getattr(message, "content", "").strip()

    # =====================================================
    # Phase 1:
    # Understand proposed change (FAST - using local index)
    # =====================================================

    def analyze_change_request(
        self,
        proposed_change: str,
    ) -> Dict:
        """
        Analyze a proposed change using the local function index.
        This is MUCH faster than fetching from GitHub.
        """

        try:
            owner, repo = self._parse_repo_url(self.repo_url)
        except ValueError:
            return {
                "architecture": {"answer": "Could not parse repo URL"},
                "activity": {"answer": "Could not parse repo URL"},
                "runtime_intelligence": {"answer": "Could not parse repo URL"},
            }

        # Load local function index
        repo_functions = self._get_repo_functions_from_index(repo)
        
        if not repo_functions:
            # Fallback to slow methods if no local index
            try:
                architecture = rag_engine.ask_repo(
                    self.repo_url,
                    f"Explain the architecture related to: {proposed_change}",
                )
            except Exception as e:
                architecture = {"answer": f"Architecture analysis failed: {str(e)}"}
            
            try:
                activity = history.ask_repo_activity(
                    self.repo_url,
                    f"Find recent issues and PRs related to: {proposed_change}",
                )
            except Exception as e:
                activity = {"answer": f"Activity analysis failed: {str(e)}"}
            
            return {
                "architecture": architecture,
                "activity": activity,
                "runtime_intelligence": {"answer": "No local data available"},
            }

        # FAST PATH: Use local function index
        functions_summary = self._format_function_index_summary(
            repo_functions,
            proposed_change
        )

        system_prompt = """
        You are a repository architect analyzing code structure
        based on function definitions and signatures.
        
        Analyze the codebase and answer the user's question about
        architecture and components.
        """

        architecture_prompt = f"""
        Repository: {repo}
        Proposed Change: {proposed_change}

        Available Functions & Files:
        {functions_summary}

        Based on the function signatures and file structure above,
        analyze the architecture. Include:
        - Key services/modules
        - Main components
        - File organization
        - Likely affected functions
        - Probable dependencies
        """

        architecture = {
            "answer": self._chat(system_prompt, architecture_prompt)
        }

        activity = {
            "answer": f"(Using local index for {repo} - {len(repo_functions)} functions analyzed)"
        }

        return {
            "architecture": architecture,
            "activity": activity,
            "runtime_intelligence": {"answer": "Analyzed based on local function index"},
        }

    # =====================================================
    # Conversation Interface
    # =====================================================

    def generate_defensive_questions(
        self,
        change_request: str,
    ) -> Dict:
        """
        Generate THREE defensive review questions in parallel.
        Faster than iterative conversation.
        """
        try:
            owner, repo = self._parse_repo_url(self.repo_url)
        except ValueError:
            return {
                "error": "Error parsing repository URL",
                "questions": []
            }

        repo_functions = self._get_repo_functions_from_index(repo)
        functions_summary = self._format_function_index_summary(repo_functions, change_request)

        # Keep this path deterministic and fast: the UI needs the questions
        # immediately so it can drive the architect/gatekeeper conversation.
        key_files = []
        for line in functions_summary.splitlines():
            line = line.strip()
            if line.startswith("**") and line.endswith("**"):
                key_files.append(line.strip("*"))
            if len(key_files) >= 3:
                break

        primary_targets = ", ".join(key_files) if key_files else repo

        return {
            "risk_level": "medium",
            "critical_systems": key_files[:3],
            "questions": [
                {
                    "id": 1,
                    "category": "Impact & Scope",
                    "question": (
                        f"Which files, services, or workflows in {primary_targets} are directly affected by {change_request}, "
                        "and what is the smallest feasible change that keeps the blast radius limited?"
                    ),
                },
                {
                    "id": 2,
                    "category": "Risk & Dependencies",
                    "question": (
                        f"What downstream services, shared contracts, or runtime assumptions around {repo} could break, "
                        "and what evidence from the source repo shows those dependencies are safe?"
                    ),
                },
                {
                    "id": 3,
                    "category": "Testing & Deployment",
                    "question": (
                        "What tests, rollout steps, monitoring checks, and rollback path will prove this change is safe before it reaches production?"
                    ),
                },
            ],
            "required_tests": [
                "Unit tests covering the changed code paths",
                "Integration or contract tests for affected interfaces",
                "A rollback-ready deployment validation plan",
            ],
            "possible_breakages": [
                "Hidden coupling to shared utilities or service contracts",
                "Regression in request handling, validation, or deployment behavior",
            ],
            "approval_recommendation": "needs_clarification",
        }

    def evaluate_user_answers(
        self,
        change_request: str,
        user_answers: List[Dict],
    ) -> Dict:
        """
        Evaluate user answers to generate a confidence score and decision.

        This path is deterministic and local so the UI can update confidence
        immediately after every answer without waiting on a model round-trip.
        """
        try:
            owner, repo = self._parse_repo_url(self.repo_url)
        except ValueError:
            return {"error": "Invalid repository URL"}

        repo_functions = self._get_repo_functions_from_index(repo)
        functions_summary = self._format_function_index_summary(repo_functions, change_request)

        def _score_answer(answer: Dict) -> tuple[int, str]:
            text = (answer.get("answer") or "").strip()
            risk_level = (answer.get("risk_level") or "").strip().lower()
            question = (answer.get("question") or "").strip().lower()
            tokens = re.findall(r"\b\w+\b", text.lower())

            score = 0
            notes = []

            if len(text) >= 160:
                score += 18
                notes.append("detailed answer")
            elif len(text) >= 80:
                score += 12
                notes.append("reasonably detailed")
            elif len(text) >= 30:
                score += 6
                notes.append("minimal detail")
            else:
                score -= 12
                notes.append("too short")

            if risk_level in {"low", "medium", "high"}:
                score += 4
                notes.append(f"declared risk level: {risk_level}")

            if any(keyword in text.lower() for keyword in ["test", "testing", "rollback", "deploy", "monitor", "monitoring", "validation", "canary", "rollback"]):
                score += 12
                notes.append("covers testing or rollout")

            if any(keyword in text.lower() for keyword in ["file", "service", "endpoint", "contract", "cache", "ttl", "invalidate", "auth", "session", "permission"]):
                score += 8
                notes.append("mentions implementation details")

            if question and len(tokens) < 6:
                score -= 8
                notes.append("answer is too vague")

            return score, "; ".join(notes)

        confidence = 40
        answer_notes = []
        total_score = 0

        for answer in user_answers:
            score, note = _score_answer(answer)
            total_score += score
            category = answer.get("category", "Unknown")
            answer_notes.append(f"{category}: {note or 'no signal'} ({score:+d})")

        # Base the score on how many questions have been answered well.
        confidence += total_score
        answered_count = sum(1 for answer in user_answers if (answer.get("answer") or "").strip())
        confidence += min(answered_count * 3, 9)

        if len(user_answers) >= 3 and all((answer.get("answer") or "").strip() for answer in user_answers):
            confidence += 5

        confidence = max(0, min(100, confidence))

        if confidence >= 75:
            decision = "APPROVE"
        elif confidence >= 50:
            decision = "NEEDS_CHANGES"
        else:
            decision = "REJECT"

        if decision == "APPROVE":
            summary = "The answers are specific enough to justify a safe change with normal review."
        elif decision == "NEEDS_CHANGES":
            summary = "The answers show progress, but the gatekeeper still needs clearer implementation and rollout detail."
        else:
            summary = "The answers are too vague or incomplete to establish safety yet."

        return {
            "confidence": confidence,
            "decision": decision,
            "explanation": (
                f"{summary}\n\n"
                f"Repository context considered: {len(repo_functions)} indexed functions across the local repo index.\n\n"
                f"Scoring notes:\n- " + "\n- ".join(answer_notes)
            ),
            "user_answers": user_answers,
        }

    def finalize_conversation(
        self,
        change_request: str,
        conversation_history: List[Dict],
    ) -> Dict:
        """
        Finalize the review with a safety evaluation.
        """
        # Build conversation summary
        summary_parts = [
            f"Change Request: {change_request}",
            "\nConversation Summary:"
        ]

        for msg in conversation_history:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")[:200]
            summary_parts.append(f"- {role}: {content}...")

        conversation_text = "\n".join(summary_parts)

        system_prompt = """
        You are a final safety evaluator for code changes.
        Based on the conversation, make a final decision on change safety.
        Output VALID JSON only.
        """

        eval_prompt = f"""
        {conversation_text}

        Provide a final evaluation in this JSON format:
        {{
          "decision": "APPROVE|NEEDS_CHANGES|REJECT",
          "confidence": 0-100,
          "summary": "Brief explanation",
          "major_risks": ["risk1", "risk2"],
          "required_actions": ["action1", "action2"]
        }}
        """

        raw_response = self._chat(system_prompt, eval_prompt)

        try:
            evaluation = json.loads(raw_response)
        except Exception:
            evaluation = {
                "decision": "NEEDS_CHANGES",
                "confidence": 50,
                "summary": "Could not parse evaluation",
                "major_risks": ["Evaluation parsing failed"],
                "required_actions": []
            }

        return {"evaluation": evaluation}

    # =====================================================
    # Phase 2:
    # Generate defensive review questions
    # =====================================================

    def generate_defensive_questions(
        self,
        proposed_change: str,
    ) -> Dict:
        """
        Generate defensive questions by analyzing:
        - Architecture implications
        - Historical patterns
        - Runtime & contract implications

        Always returns a structured `questions` list so the UI can render
        the architect/gatekeeper conversation without a missing-data branch.
        """

        context = self.analyze_change_request(
            proposed_change
        )

        architecture_text = context["architecture"]["answer"]
        activity_text = context["activity"]["answer"]
        runtime_text = context["runtime_intelligence"]["answer"]

        system_prompt = """
        You are a senior staff engineer acting as
        a defensive repository guardian.

        Your responsibility is to PREVENT changes
        that could break the microservice.

        You must:
        - identify hidden dependencies
        - identify unclear assumptions
        - identify API break risks
        - identify migration risks
        - identify infra risks
        - identify concurrency risks
        - identify schema risks
        - identify deployment risks
        - identify backward compatibility risks
        - identify scaling risks
        - identify CI/CD risks
        - identify contract violations

        You should aggressively ask clarifying questions
        BEFORE approving a change.

        Output STRICT JSON.

        Format:

        {
          "risk_level": "...",
          "critical_systems": [...],
          "questions": [...],
          "required_tests": [...],
          "possible_breakages": [...],
          "approval_recommendation": "..."
        }
        """

        user_prompt = f"""
        Proposed Change:
        {proposed_change}

        =====================================
        REPOSITORY ARCHITECTURE
        =====================================

        {architecture_text}

        =====================================
        RECENT ACTIVITY / ISSUES / PRs
        =====================================

        {activity_text}

        =====================================
        RUNTIME INTELLIGENCE & CONTRACTS
        =====================================

        {runtime_text}

        Generate a defensive review.
        """

        raw = self._chat(system_prompt, user_prompt)

        parsed: Dict = {}
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            parsed = {}

        def _default_questions() -> List[Dict]:
            return [
                {
                    "id": 1,
                    "category": "Impact & Scope",
                    "question": (
                        f"Which concrete services, files, or workflows in {repo} are affected by this change, "
                        "and what user-visible behavior changes should we expect?"
                    ),
                },
                {
                    "id": 2,
                    "category": "Risk & Dependencies",
                    "question": (
                        "What other services, contracts, or runtime assumptions could break if this change lands, "
                        "and which dependencies need explicit validation?"
                    ),
                },
                {
                    "id": 3,
                    "category": "Testing & Deployment",
                    "question": (
                        "What is the feasible test, rollout, and rollback plan for this change, including any "
                        "guards needed before it is safe to merge or deploy?"
                    ),
                },
            ]

        normalized_questions: List[Dict] = []
        raw_questions = parsed.get("questions") if isinstance(parsed, dict) else None
        if isinstance(raw_questions, list):
            for index, item in enumerate(raw_questions[:3], start=1):
                if isinstance(item, dict):
                    question_text = (
                        item.get("question")
                        or item.get("text")
                        or item.get("prompt")
                        or ""
                    ).strip()
                    category = (item.get("category") or item.get("topic") or f"Question {index}").strip()
                else:
                    question_text = str(item).strip()
                    category = f"Question {index}"

                if question_text:
                    normalized_questions.append(
                        {
                            "id": index,
                            "question": question_text,
                            "category": category,
                        }
                    )

        if len(normalized_questions) < 3:
            normalized_questions = _default_questions()

        return {
            "risk_level": parsed.get("risk_level", "medium"),
            "critical_systems": parsed.get("critical_systems", []),
            "questions": normalized_questions,
            "required_tests": parsed.get("required_tests", []),
            "possible_breakages": parsed.get("possible_breakages", []),
            "approval_recommendation": parsed.get("approval_recommendation", "needs_clarification"),
            "raw_response": raw,
        }

    # =====================================================
    # Phase 3:
    # Decide whether change is safe
    # =====================================================

    def evaluate_change_safety(
        self,
        proposed_change: str,
        developer_answers: Dict,
    ) -> Dict:
        """
        Evaluate change safety using consolidated context
        """

        context = self.analyze_change_request(
            proposed_change
        )

        system_prompt = """
        You are a repository defense agent.

        Your job is to determine whether
        a proposed code change is SAFE.

        Consider:
        - hidden coupling
        - backward compatibility
        - infra impact
        - database impact
        - race conditions
        - event ordering
        - deployment risk
        - rollback complexity
        - observability gaps
        - auth/security implications
        - CI/CD & contract compliance

        Output STRICT JSON.

        {
          "decision": "APPROVE | NEEDS_CHANGES | REJECT",
          "confidence": 0-100,
          "summary": "...",
          "major_risks": [...],
          "missing_information": [...],
          "required_actions": [...],
          "recommended_tests": [...],
          "rollback_plan_required": true/false
        }
        """

        user_prompt = f"""
        =====================================
        PROPOSED CHANGE
        =====================================

        {proposed_change}

        =====================================
        DEVELOPER ANSWERS
        =====================================

        {json.dumps(developer_answers, indent=2)}

        =====================================
        REPOSITORY ARCHITECTURE
        =====================================

        {context['architecture']['answer']}

        =====================================
        REPOSITORY HISTORY / ISSUES
        =====================================

        {context['activity']['answer']}

        =====================================
        RUNTIME INTELLIGENCE
        =====================================

        {context['runtime_intelligence']['answer']}

        Evaluate safety.
        """

        raw = self._chat(
            system_prompt,
            user_prompt,
        )

        try:
            return json.loads(raw)

        except Exception:
            return {
                "raw_response": raw
            }

    # =====================================================
    # Full workflow
    # =====================================================

    def defend_repo_change(
        self,
        proposed_change: str,
    ) -> Dict:
        """
        Run the complete defensive flow
        """

        review = self.generate_defensive_questions(
            proposed_change
        )

        return {
            "repo": self.repo_url,
            "proposed_change": proposed_change,
            "review": review,
        }


# =========================================================
# Example usage
# =========================================================

if __name__ == "__main__":

    repo = "https://github.com/vennby/hsbc-hackathon"

    agent = RepoDefenderAgent(repo)

    result = agent.defend_repo_change(
        """
        I want to add authentication to this repo.
        """
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

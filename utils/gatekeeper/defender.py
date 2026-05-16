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
import hashlib
import os
import re
from typing import Dict, List, Optional, Tuple

from huggingface_hub import InferenceClient

from . import rag_engine
from . import history
from . import contracts

DEFAULT_CHAT_MODEL = os.environ.get(
    "HF_CHAT_MODEL",
    "deepseek-ai/DeepSeek-V4-Pro:novita",
)
FAST_CHAT_MODEL = os.environ.get("HF_FAST_CHAT_MODEL", DEFAULT_CHAT_MODEL)
FAST_CHAT_MAX_TOKENS = int(os.environ.get("HF_FAST_CHAT_MAX_TOKENS", "700"))


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

    _function_index_cache: Optional[List[Dict]] = None
    _analysis_cache_store: Dict[str, Dict] = {}
    _questions_cache_store: Dict[str, Dict] = {}
    _draft_cache_store: Dict[str, Dict] = {}
    _llm_cache_store: Dict[str, str] = {}

    def __init__(self, repo_url: str):

        self.repo_url = repo_url

        token = os.environ.get("HF_API_KEY", "").strip()

        if not token:
            raise RuntimeError(
                "HF_API_KEY is required."
            )

        self.client = InferenceClient(api_key=token)

    def _cache_key(self, prefix: str, *parts: str) -> str:
        digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
        return f"{prefix}:{digest}"

    def _is_low_signal_file(self, file_path: str) -> bool:
        normalized = (file_path or "").replace("\\", "/").lower()
        noisy_markers = [
            "vendor/",
            "_vendor/",
            "/node_modules/",
            "/vendor/",
            "/_vendor/",
            "/dist/",
            "/build/",
            "/coverage/",
            "/.next/",
            "/generated/",
            "/gen/",
            "/tmp/",
            "/fixtures/",
            "/migrations/",
            "/third_party/",
        ]
        noisy_suffixes = [
            ".min.js",
            ".lock",
            ".snap",
            ".svg",
            ".png",
            ".jpg",
            ".jpeg",
        ]
        if any(marker in normalized for marker in noisy_markers):
            return True
        if any(normalized.endswith(suffix) for suffix in noisy_suffixes):
            return True
        return False

    def _cached_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = FAST_CHAT_MAX_TOKENS,
        cache_prefix: str = "chat",
    ) -> str:
        cache_key = self._cache_key(cache_prefix, model or FAST_CHAT_MODEL, system_prompt, user_prompt)
        cached = self.__class__._llm_cache_store.get(cache_key)
        if cached is not None:
            return cached

        completion = self.client.chat.completions.create(
            model=model or FAST_CHAT_MODEL,
            max_tokens=max_tokens,
            temperature=0.2,
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
            content = message.get("content", "").strip()
        else:
            content = getattr(message, "content", "").strip()

        self.__class__._llm_cache_store[cache_key] = content
        return content

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
        if self.__class__._function_index_cache is not None:
            return self.__class__._function_index_cache

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

        self.__class__._function_index_cache = functions
        
        return functions

    def _select_relevant_functions(
        self,
        functions: List[Dict],
        change_request: str,
        limit: int = 8,
    ) -> List[Dict]:
        tokens = [
            token
            for token in re.findall(r"\b\w+\b", change_request.lower())
            if len(token) > 3
        ]

        primary_pool = [
            func for func in functions if not self._is_low_signal_file(func.get("file", ""))
        ]
        candidate_pool = primary_pool or functions

        ranked = []
        for func in candidate_pool:
            file_path = (func.get("file") or "").strip()
            searchable = (
                f"{func.get('signature', '')} {file_path} {func.get('name', '')}"
            ).lower()
            score = sum(1 for token in tokens if token in searchable)
            if not self._is_low_signal_file(file_path):
                score += 2
            else:
                score -= 4

            if any(hint in file_path.lower() for hint in ["src/", "app/", "lib/", "api/", "service", "controller", "model", "component"]):
                score += 2
            ranked.append((score, func))

        ranked.sort(
            key=lambda item: (
                item[0],
                len(item[1].get("signature", "")),
            ),
            reverse=True,
        )

        selected = [func for score, func in ranked if score > 0][:limit]
        if selected:
            return selected
        return candidate_pool[:limit]

    def _extract_key_files(self, functions: List[Dict], limit: int = 5) -> List[str]:
        key_files: List[str] = []
        for func in functions:
            file_path = (func.get("file") or "").strip()
            if file_path and file_path not in key_files:
                key_files.append(file_path)
            if len(key_files) >= limit:
                break
        return key_files

    def _build_local_analysis(
        self,
        repo: str,
        proposed_change: str,
        repo_functions: List[Dict],
    ) -> Dict:
        relevant_functions = self._select_relevant_functions(repo_functions, proposed_change)
        key_files = self._extract_key_files(relevant_functions)
        affected_repos = self._find_affected_repositories(proposed_change, repo)

        file_scope = ", ".join(key_files) if key_files else "no strong file matches found"
        function_lines = []
        for func in relevant_functions[:6]:
            signature = func.get("signature", "unknown")
            file_path = func.get("file", "unknown")
            function_lines.append(f"- {signature} in {file_path}")

        blast_radius = affected_repos if affected_repos else [repo]
        impact_rows = [
            "| Area | Likely surface | Why it matters |",
            "| --- | --- | --- |",
            f"| Primary implementation | {file_scope} | These files best match the requested change and are the most likely first edit surfaces. |",
            f"| Repository boundaries | {', '.join(blast_radius[:4])} | These repos define the compatibility boundary if behavior or contracts move. |",
            "| Validation focus | Unit, integration, and contract checks | The change should prove behavior locally first, then across affected boundaries. |",
        ]

        architecture_lines = [
            "## Architectural Summary",
            f"The local index suggests that {repo} will most likely implement this request through {file_scope}. Those files are the strongest local signals for the first edit slice and should anchor both the implementation plan and the guarded review.",
            f"The smallest plausible blast radius stays inside {repo} first. If the change introduces new behavior rather than a refactor, treat {repo} as the system of initial ownership and the other repositories as downstream compatibility boundaries.",
            "",
            "## Likely Implementation Surfaces",
            *function_lines,
            "",
            "## Impact Matrix",
            *impact_rows,
            "",
            "## Design Notes",
            "- Keep the first code change close to the relevant entry points before widening into adjacent modules.",
            "- Preserve existing repository contracts unless the review explicitly authorizes a coordinated boundary change.",
            "- Prefer repository-local fixes over speculative changes in unrelated services.",
        ]

        activity_lines = [
            "## Coordination View",
            f"Using the local index for {repo}; {len(repo_functions)} functions are available for repository context. That is enough to reconstruct the likely implementation path without waiting on broad remote repository analysis.",
            f"Potentially affected repositories: {', '.join(affected_repos) if affected_repos else 'none identified beyond the primary repository'}. These should be treated as explicit review partners only when the change modifies a shared workflow, contract, or integration boundary.",
            "",
            "## Review Expectations",
            "- The architect should name the first files to change, not just the general subsystem.",
            "- Any cross-repository effect should be justified with a concrete contract, interface, or workflow dependency.",
            "- The final specification should separate primary implementation work from downstream validation work.",
        ]

        runtime_lines = [
            "## Runtime and Delivery Considerations",
            f"For the proposed change \"{proposed_change}\", validation should focus on the files and signatures listed above, with particular attention to user-visible behavior, contract compatibility, migration safety, and rollback readiness.",
            "",
            "| Concern | Required evidence |",
            "| --- | --- |",
            "| Correctness | Unit or component tests around the directly changed code paths |",
            "| Boundary safety | Contract or integration checks where another repository may consume the behavior |",
            "| Release safety | Rollout and rollback notes tied to the affected files or workflows |",
            "| Operations | Monitoring or verification steps for the user-facing behavior that changes |",
            "",
            "Use repository-specific tests in the primary repo first, then add contract or integration checks for any affected repositories before rollout.",
        ]

        return {
            "architecture": {"answer": "\n".join(architecture_lines)},
            "activity": {"answer": "\n".join(activity_lines)},
            "runtime_intelligence": {"answer": "\n".join(runtime_lines)},
            "key_files": key_files,
            "relevant_functions": relevant_functions,
            "affected_repos": affected_repos,
        }

    def _build_local_evidence_summary(self, analysis: Dict) -> str:
        key_files = analysis.get("key_files", [])
        relevant_functions = analysis.get("relevant_functions", [])
        affected_repos = analysis.get("affected_repos", [])
        architecture = ((analysis.get("architecture") or {}).get("answer") or "").strip()
        activity = ((analysis.get("activity") or {}).get("answer") or "").strip()
        runtime_intelligence = ((analysis.get("runtime_intelligence") or {}).get("answer") or "").strip()

        lines = []
        if key_files:
            lines.append("Key files:")
            for file_path in key_files[:4]:
                lines.append(f"- {file_path}")

        if relevant_functions:
            lines.append("Relevant functions:")
            for func in relevant_functions[:6]:
                lines.append(
                    f"- {func.get('signature', 'unknown')} in {func.get('file', 'unknown')}"
                )

        if affected_repos:
            lines.append(
                "Potentially affected repositories: " + ", ".join(affected_repos[:5])
            )

        if architecture:
            lines.append("Architecture summary:")
            lines.append(architecture)

        if activity:
            lines.append("Repository activity summary:")
            lines.append(activity)

        if runtime_intelligence:
            lines.append("Runtime and validation summary:")
            lines.append(runtime_intelligence)

        return "\n".join(lines)

    def _build_local_architect_draft(
        self,
        repo: str,
        question: Dict,
        analysis: Dict,
        prior_answers: List[Dict],
    ) -> Dict:
        key_files = analysis.get("key_files", [])
        affected_repos = analysis.get("affected_repos", [])
        relevant_functions = analysis.get("relevant_functions", [])
        category = question.get("category", "Impact & Scope")

        lead = ""
        if category == "Impact & Scope":
            lead = (
                f"The change should stay centered on {', '.join(key_files[:3]) if key_files else repo}. "
                "That keeps the blast radius limited to the code paths most directly implied by the request."
            )
        elif category == "Risk & Dependencies":
            lead = (
                f"The main dependency risk is at the shared boundaries around {', '.join(key_files[:2]) if key_files else repo}. "
                f"Cross-repository checks are needed for {', '.join(affected_repos[:3]) if affected_repos else 'shared contracts and downstream consumers'}."
            )
        else:
            lead = (
                "The safest path is to pair the implementation with targeted tests, a staged rollout, and a rollback path tied to the affected files."
            )

        support_lines = []
        for func in relevant_functions[:3]:
            support_lines.append(
                f"- Inspect {func.get('signature', 'unknown')} in {func.get('file', 'unknown')}"
            )

        prior_note = ""
        if prior_answers:
            prior_note = (
                "\n\nThis answer also stays aligned with the earlier trial responses and avoids broadening scope beyond what has already been justified."
            )

        answer = (
            f"{lead}\n\n"
            f"I would anchor the implementation and review on these repository references:\n"
            f"{'\n'.join(support_lines) if support_lines else '- Use the primary repo files surfaced by the local index'}\n\n"
            f"If the change touches shared behavior, validate the contracts used by "
            f"{', '.join(affected_repos) if affected_repos else 'the primary repository only'}, and keep the rollout gated by targeted tests and a clear rollback plan."
            f"{prior_note}"
        )

        return {
            "answer": answer,
            "key_references": key_files[:3] or [repo],
            "assumptions": [
                "Draft generated from locally indexed repository metadata.",
                "User can refine file-level details before submission to the gatekeeper.",
            ],
        }

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
        return self._cached_chat(
            system_prompt,
            user_prompt,
            model=DEFAULT_CHAT_MODEL,
            max_tokens=1200,
            cache_prefix="default-chat",
        )

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

        cache_key = self._cache_key("analysis", self.repo_url, proposed_change)
        cached = self.__class__._analysis_cache_store.get(cache_key)
        if cached is not None:
            return cached

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

        analysis = self._build_local_analysis(repo, proposed_change, repo_functions)
        evidence_summary = self._build_local_evidence_summary(analysis)

        system_prompt = """
        You are a repository architect.
        Use only the provided local-index evidence to analyze the proposed change.
        Ignore vendor, generated, or third-party code unless the evidence strongly suggests it is central.
        Output STRICT JSON with keys architecture, activity, and runtime_intelligence.
        Write a rich but efficient analysis suitable for senior engineers: concrete files, blast radius, repository boundaries, validation, rollout, and open risks.
        Prefer short sections, bullets, and markdown tables over vague prose.
        """

        user_prompt = f"""
        Repository: {repo}
        Proposed change: {proposed_change}

        Local evidence:
        {evidence_summary}

        Return JSON:
        {{
                    "architecture": "3-5 compact sections including likely modules/files, repository boundaries, blast radius, and at least one markdown table",
                    "activity": "2-4 compact sections covering coordination, dependency, cross-repository concerns, and review expectations",
                    "runtime_intelligence": "2-4 compact sections covering tests, rollout, rollback, contract, monitoring, and open operational risks"
        }}
        """

        try:
            raw = self._cached_chat(
                system_prompt,
                user_prompt,
                cache_prefix="analysis-llm",
                max_tokens=1000,
            )
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                if (parsed.get("architecture") or "").strip():
                    analysis["architecture"] = {"answer": parsed["architecture"].strip()}
                if (parsed.get("activity") or "").strip():
                    analysis["activity"] = {"answer": parsed["activity"].strip()}
                if (parsed.get("runtime_intelligence") or "").strip():
                    analysis["runtime_intelligence"] = {"answer": parsed["runtime_intelligence"].strip()}
        except Exception:
            pass

        self.__class__._analysis_cache_store[cache_key] = analysis
        return analysis

    # =====================================================
    # Gatekeeper Trial Helpers
    # =====================================================

    def _score_trial_answer(self, answer: Dict) -> Tuple[int, str]:
        text = (answer.get("answer") or "").strip()
        question = (answer.get("question") or "").strip().lower()
        tokens = re.findall(r"\b\w+\b", text.lower())

        score = 0
        notes = []

        if len(text) >= 420:
            score += 20
            notes.append("deeply detailed answer")
        elif len(text) >= 300:
            score += 11
            notes.append("detailed answer")
        elif len(text) >= 200:
            score += 3
            notes.append("minimally sufficient detail")
        elif len(text) >= 120:
            score -= 7
            notes.append("still too brief")
        else:
            score -= 24
            notes.append("too short")

        implementation_hits = sum(
            1
            for keyword in [
                "file",
                "module",
                "function",
                "service",
                "endpoint",
                "contract",
                "schema",
                "cache",
                "auth",
                "permission",
                "queue",
                "job",
                "repository",
                "interface",
                "workflow",
            ]
            if keyword in text.lower()
        )
        if implementation_hits >= 5:
            score += 14
            notes.append("references concrete implementation details")
        elif implementation_hits >= 3:
            score += 6
            notes.append("some implementation detail present")
        elif implementation_hits >= 1:
            score -= 2
            notes.append("implementation detail is thin")
        else:
            score -= 10
            notes.append("missing concrete implementation detail")

        validation_hits = sum(
            1
            for keyword in [
                "test",
                "testing",
                "rollout",
                "deploy",
                "rollback",
                "monitor",
                "validation",
                "contract test",
                "canary",
                "staging",
                "observability",
            ]
            if keyword in text.lower()
        )
        if validation_hits >= 3:
            score += 12
            notes.append("covers validation and rollout")
        elif validation_hits >= 1:
            score += 2
            notes.append("mentions validation or rollout")
        else:
            score -= 8
            notes.append("missing validation and rollout detail")

        reasoning_hits = sum(
            1
            for keyword in [
                "because",
                "so that",
                "therefore",
                "to avoid",
                "to reduce",
                "which means",
                "this keeps",
                "this avoids",
            ]
            if keyword in text.lower()
        )
        if reasoning_hits >= 3:
            score += 12
            notes.append("clearly explains reasoning")
        elif reasoning_hits >= 1:
            score += 3
            notes.append("explains reasoning")
        else:
            score -= 10
            notes.append("reasoning is implicit")

        repo_specific_hits = sum(
            1
            for keyword in [
                "repo",
                "repository",
                "interface",
                "contract",
                "consumer",
                "producer",
                "downstream",
                "upstream",
                "boundary",
                "file",
                "module",
            ]
            if keyword in text.lower()
        )
        if repo_specific_hits >= 4:
            score += 8
            notes.append("repo-specific scope is explicit")
        elif repo_specific_hits >= 2:
            score += 2
            notes.append("some repo-specific scope is present")
        else:
            score -= 8
            notes.append("repo-specific scope is weak")

        if question and len(tokens) < 24:
            score -= 14
            notes.append("too vague for the question")

        if question and not any(token in text.lower() for token in re.findall(r"\b\w+\b", question) if len(token) > 5):
            score -= 10
            notes.append("does not clearly answer the asked question")

        if any(marker in text for marker in ["\n- ", "\n1.", ":\n"]):
            score += 5
            notes.append("structured answer")
        else:
            score -= 3
            notes.append("structure is weak")

        return score, "; ".join(notes)

    def _format_trial_feedback(
        self,
        confidence: int,
        confidence_delta: int,
        answer_count: int,
        answer_notes: List[str],
        repo_function_count: int,
    ) -> str:
        if confidence_delta > 0:
            movement = f"Confidence increased by {confidence_delta} points."
        elif confidence_delta < 0:
            movement = f"Confidence dropped by {abs(confidence_delta)} points."
        else:
            movement = "Confidence stayed flat on this turn."

        if answer_count >= 3:
            verdict = (
                "The architect passes the trial."
                if confidence >= 70
                else "The architect fails the trial."
            )
        else:
            verdict = "The trial continues with the next question."

        return (
            f"{movement} {verdict}\n\n"
            f"Repository context considered: {repo_function_count} indexed functions from the reference repository.\n\n"
            f"Turn notes:\n- " + "\n- ".join(answer_notes)
        )

    def draft_architect_response(
        self,
        change_request: str,
        question: Dict,
        prior_answers: List[Dict],
    ) -> Dict:
        """
        Draft the architect's next answer using repository context so the
        user can edit it before it is sent back to the gatekeeper.
        """
        try:
            owner, repo = self._parse_repo_url(self.repo_url)
        except ValueError:
            return {
                "answer": "I could not parse the repository URL well enough to draft an answer.",
                "key_references": [],
                "assumptions": ["Repository URL parsing failed."],
            }

        repo_functions = self._get_repo_functions_from_index(repo)
        context = self.analyze_change_request(change_request)

        if repo_functions:
            fallback = self._build_local_architect_draft(
                repo,
                question,
                context,
                prior_answers,
            )
            evidence_summary = self._build_local_evidence_summary(context)
            prior_summary = "No prior answers yet."
            if prior_answers:
                prior_summary = "\n".join(
                    f"Q{answer.get('id', '?')} ({answer.get('category', 'Unknown')}): {str(answer.get('answer', ''))[:180]}"
                    for answer in prior_answers[-2:]
                )

            draft_cache_key = self._cache_key(
                "draft",
                self.repo_url,
                change_request,
                json.dumps(question, sort_keys=True),
                prior_summary,
            )
            cached_draft = self.__class__._draft_cache_store.get(draft_cache_key)
            if cached_draft is not None:
                return cached_draft

            system_prompt = """
            You are the architect agent in a gatekeeper trial.
            Use only the provided local-index evidence.
            Draft a concrete answer to the current question.
            Mention specific files, contracts, tests, or rollout details when supported.
            Output STRICT JSON with answer, key_references, and assumptions.
            """

            user_prompt = f"""
            Repository: {repo}
            Proposed change: {change_request}
            Current question category: {question.get('category', 'Unknown')}
            Current question: {question.get('question', '')}

            Prior answers:
            {prior_summary}

            Local evidence:
            {evidence_summary}

            Fallback guidance:
            {fallback['answer']}

            Return JSON:
            {{
              "answer": "...",
              "key_references": ["..."],
              "assumptions": ["..."]
            }}
            """

            try:
                raw = self._cached_chat(
                    system_prompt,
                    user_prompt,
                    cache_prefix="draft-llm",
                    max_tokens=450,
                )
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and (parsed.get("answer") or "").strip():
                    result = {
                        "answer": parsed.get("answer", "").strip(),
                        "key_references": parsed.get("key_references", []) or fallback["key_references"],
                        "assumptions": parsed.get("assumptions", []) or fallback["assumptions"],
                    }
                    self.__class__._draft_cache_store[draft_cache_key] = result
                    return result
            except Exception:
                pass

            self.__class__._draft_cache_store[draft_cache_key] = fallback
            return fallback

        functions_summary = self._format_function_index_summary(repo_functions, change_request)

        prior_summary = "No prior answers yet."
        if prior_answers:
            parts = []
            for answer in prior_answers[-2:]:
                parts.append(
                    f"Q{answer.get('id', '?')} ({answer.get('category', 'Unknown')}): {str(answer.get('answer', ''))[:220]}"
                )
            prior_summary = "\n".join(parts)

        system_prompt = """
        You are the architect agent in a gatekeeper trial.

        Answer the gatekeeper's current question as well as you can using the
        repository context that is provided. Write a draft that is concrete,
        technically useful, and safe to edit by a human before submission.

        Output STRICT JSON only.

        {
          "answer": "draft answer",
          "key_references": ["file or subsystem", "file or subsystem"],
          "assumptions": ["assumption 1", "assumption 2"]
        }
        """

        user_prompt = f"""
        Repository: {repo}
        Proposed change: {change_request}

        Current question category: {question.get('category', 'Unknown')}
        Current question: {question.get('question', '')}

        Prior answers:
        {prior_summary}

        Reference repository context:
        - Architecture: {context['architecture']['answer']}
        - Activity: {context['activity']['answer']}
        - Runtime intelligence: {context['runtime_intelligence']['answer']}

        Local function index summary:
        {functions_summary}

        Draft the architect's answer.
        """

        raw = self._chat(system_prompt, user_prompt)

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and (parsed.get("answer") or "").strip():
                return {
                    "answer": parsed.get("answer", "").strip(),
                    "key_references": parsed.get("key_references", []),
                    "assumptions": parsed.get("assumptions", []),
                }
        except Exception:
            pass

        return {
            "answer": raw.strip() or (
                "The available repository context suggests the change should stay limited to the files and "
                "contracts most directly tied to this request, with explicit testing and rollback notes."
            ),
            "key_references": [],
            "assumptions": ["Architect draft fell back to plain text."],
        }

    def start_gatekeeper_trial(
        self,
        change_request: str,
        questions: List[Dict] | None = None,
    ) -> Dict:
        review = self.generate_defensive_questions(change_request)
        ordered_questions = questions or review.get("questions", [])
        current_question = ordered_questions[0] if ordered_questions else None
        architect_draft = (
            self.draft_architect_response(change_request, current_question, [])
            if current_question
            else {"answer": "", "key_references": [], "assumptions": []}
        )

        return {
            "questions": ordered_questions,
            "current_question": current_question,
            "architect_draft": architect_draft,
            "confidence": 50,
            "decision": "PENDING",
            "turn_index": 0,
            "required_tests": review.get("required_tests", []),
            "possible_breakages": review.get("possible_breakages", []),
        }

    def evaluate_user_answers(
        self,
        change_request: str,
        user_answers: List[Dict],
    ) -> Dict:
        """
        Score the architect's submitted answers turn by turn and convert the
        final three-turn outcome into a pass/fail gatekeeper decision.
        """
        try:
            owner, repo = self._parse_repo_url(self.repo_url)
        except ValueError:
            return {"error": "Invalid repository URL"}

        repo_functions = self._get_repo_functions_from_index(repo)
        answer_notes = []
        total_score = 0

        for answer in user_answers:
            score, note = self._score_trial_answer(answer)
            total_score += score
            category = answer.get("category", "Unknown")
            answer_notes.append(f"{category}: {note or 'no signal'} ({score:+d})")

        answered_count = sum(
            1 for answer in user_answers if (answer.get("answer") or "").strip()
        )
        confidence = max(0, min(100, 22 + total_score))
        completed_trial = answered_count >= 3

        if completed_trial:
            decision = "PASS" if confidence >= 86 else "FAIL"
            approval_decision = "APPROVE" if decision == "PASS" else "REJECT"
        else:
            decision = "PENDING"
            approval_decision = "NEEDS_CHANGES"

        explanation = self._format_trial_feedback(
            confidence,
            total_score,
            answered_count,
            answer_notes or ["No answers scored yet."],
            len(repo_functions),
        )

        return {
            "confidence": confidence,
            "decision": decision,
            "approval_decision": approval_decision,
            "completed_trial": completed_trial,
            "turns_completed": answered_count,
            "explanation": explanation,
            "user_answers": user_answers,
        }

    def review_trial_answer(
        self,
        change_request: str,
        question: Dict,
        edited_answer: str,
        answer_history: List[Dict],
        questions: List[Dict] | None = None,
    ) -> Dict:
        previous_answers = [
            answer
            for answer in answer_history
            if isinstance(answer, dict) and (answer.get("answer") or "").strip()
        ]
        previous_state = self.evaluate_user_answers(change_request, previous_answers)

        current_answer = {
            "id": question.get("id", len(previous_answers) + 1),
            "category": question.get("category", "Unknown"),
            "question": question.get("question", ""),
            "answer": edited_answer,
        }

        all_answers = previous_answers + [current_answer]
        current_state = self.evaluate_user_answers(change_request, all_answers)
        confidence_delta = current_state["confidence"] - previous_state.get("confidence", 50)

        ordered_questions = questions or self.generate_defensive_questions(change_request).get("questions", [])
        next_question = None
        architect_draft = None

        if not current_state["completed_trial"] and len(ordered_questions) > len(all_answers):
            next_question = ordered_questions[len(all_answers)]
            architect_draft = self.draft_architect_response(
                change_request,
                next_question,
                all_answers,
            )

        return {
            **current_state,
            "confidence_delta": confidence_delta,
            "gatekeeper_response": self._format_trial_feedback(
                current_state["confidence"],
                confidence_delta,
                len(all_answers),
                [current_state["user_answers"][-1].get("category", "Unknown") + ": " + self._score_trial_answer(current_state["user_answers"][-1])[1]],
                len(self._get_repo_functions_from_index(self._parse_repo_url(self.repo_url)[1])),
            ),
            "current_question": question,
            "next_question": next_question,
            "architect_draft": architect_draft,
            "user_answers": all_answers,
        }

    def finalize_conversation(
        self,
        change_request: str,
        conversation_history: List[Dict],
    ) -> Dict:
        """
        Preserve a final evaluation entry point for older callers.
        """
        answers = [
            message
            for message in conversation_history
            if isinstance(message, dict) and message.get("role") == "architect"
        ]
        normalized_answers = [
            {
                "id": index,
                "category": message.get("category", f"Question {index}"),
                "question": message.get("question", ""),
                "answer": message.get("content", ""),
            }
            for index, message in enumerate(answers, start=1)
        ]
        return {"evaluation": self.evaluate_user_answers(change_request, normalized_answers)}

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

        cache_key = self._cache_key("questions", self.repo_url, proposed_change)
        cached = self.__class__._questions_cache_store.get(cache_key)
        if cached is not None:
            return cached

        try:
            owner, repo = self._parse_repo_url(self.repo_url)
        except ValueError:
            return {
                "risk_level": "medium",
                "critical_systems": [],
                "questions": [],
                "required_tests": [],
                "possible_breakages": [],
                "approval_recommendation": "needs_clarification",
                "raw_response": "",
            }

        context = self.analyze_change_request(proposed_change)
        key_files = context.get("key_files", [])
        affected_repos = context.get("affected_repos", [])
        repo_scope = ", ".join(key_files[:3]) if key_files else repo
        affected_scope = ", ".join(affected_repos[:3]) if affected_repos else repo

        fallback = {
            "risk_level": "medium" if affected_repos else "low",
            "critical_systems": key_files[:3],
            "questions": [
                {
                    "id": 1,
                    "category": "Impact & Scope",
                    "question": (
                        f"Which concrete files, workflows, or repository-owned interfaces in {repo_scope} must change first, "
                        "and what exact behavior should the primary repository expose after this change lands?"
                    ),
                },
                {
                    "id": 2,
                    "category": "Risk & Dependencies",
                    "question": (
                        f"Which contracts, data flows, or repository boundaries involving {affected_scope} could break, "
                        "and what evidence will prove those integrations remain compatible?"
                    ),
                },
                {
                    "id": 3,
                    "category": "Testing & Deployment",
                    "question": (
                        "What repository-specific tests, rollout sequencing, monitoring checks, and rollback steps are required before this is safe to merge and deploy?"
                    ),
                },
            ],
            "required_tests": [
                "Unit tests for the directly affected files or functions",
                "Integration or contract checks across any shared boundaries",
                "A rollout and rollback checklist for the affected repository scope",
            ],
            "possible_breakages": [
                "Hidden coupling in shared utilities, contracts, or runtime assumptions",
                "Regression in user-visible behavior across the primary or affected repositories",
            ],
            "approval_recommendation": "needs_clarification",
            "raw_response": "local-index-generated",
        }

        evidence_summary = self._build_local_evidence_summary(context)
        system_prompt = """
        You are a defensive gatekeeper reviewer.
        Use only the supplied local-index evidence.
        Produce exactly three high-value review questions that probe scope, dependency risk, and rollout safety.
        Make the questions repository-aware and specific to the primary repository plus any affected repositories named in the evidence.
        Each question should demand concrete files, interfaces, tests, or rollout actions.
        Ignore vendor or generated code unless it is clearly central.
        Output STRICT JSON.
        """

        user_prompt = f"""
        Repository: {repo}
        Proposed change: {proposed_change}

        Primary repository scope: {repo_scope}
        Potentially affected repositories: {affected_scope}

        Local evidence:
        {evidence_summary}

        Return JSON:
        {{
          "risk_level": "low|medium|high",
          "critical_systems": ["..."],
          "questions": [
            {{"category": "Impact & Scope", "question": "..."}},
            {{"category": "Risk & Dependencies", "question": "..."}},
            {{"category": "Testing & Deployment", "question": "..."}}
          ],
          "required_tests": ["..."],
          "possible_breakages": ["..."],
          "approval_recommendation": "needs_clarification|approve_with_checks"
        }}
        """

        result = fallback
        try:
            raw = self._cached_chat(
                system_prompt,
                user_prompt,
                cache_prefix="questions-llm",
                max_tokens=650,
            )
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                questions = []
                for index, item in enumerate((parsed.get("questions") or [])[:3], start=1):
                    if not isinstance(item, dict):
                        continue
                    question_text = (item.get("question") or item.get("text") or "").strip()
                    category = (item.get("category") or f"Question {index}").strip()
                    if question_text:
                        questions.append(
                            {
                                "id": index,
                                "category": category,
                                "question": question_text,
                            }
                        )
                if len(questions) == 3:
                    result = {
                        "risk_level": parsed.get("risk_level", fallback["risk_level"]),
                        "critical_systems": parsed.get("critical_systems", fallback["critical_systems"]),
                        "questions": questions,
                        "required_tests": parsed.get("required_tests", fallback["required_tests"]),
                        "possible_breakages": parsed.get("possible_breakages", fallback["possible_breakages"]),
                        "approval_recommendation": parsed.get("approval_recommendation", fallback["approval_recommendation"]),
                        "raw_response": raw,
                    }
        except Exception:
            result = fallback

        self.__class__._questions_cache_store[cache_key] = result
        return result

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

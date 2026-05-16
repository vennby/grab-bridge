import json
import os
from typing import Dict, List

from huggingface_hub import InferenceClient

from rag_engine import ask_repo
from history import ask_repo_activity

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

    The agent:
    - understands repository architecture
    - understands repo activity/history/issues
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
    # Understand proposed change
    # =====================================================

    def analyze_change_request(
        self,
        proposed_change: str,
    ) -> Dict:

        architecture = ask_repo(
            self.repo_url,
            f"""
            Explain the architecture and components related to:

            {proposed_change}

            Include:
            - services involved
            - APIs involved
            - dependencies
            - databases
            - event systems
            - queues
            - auth systems
            - external integrations
            - tests
            - likely affected files
            """,
        )

        activity = ask_repo_activity(
            self.repo_url,
            f"""
            Find recent issues, merged PRs, commits,
            regressions, incidents, migrations,
            refactors, or risky areas related to:

            {proposed_change}
            """,
        )

        return {
            "architecture": architecture,
            "activity": activity,
        }

    # =====================================================
    # Phase 2:
    # Generate defensive review questions
    # =====================================================

    def generate_defensive_questions(
        self,
        proposed_change: str,
    ) -> Dict:

        context = self.analyze_change_request(
            proposed_change
        )

        architecture_text = context["architecture"]["answer"]

        activity_text = context["activity"]["answer"]

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

        Generate a defensive review.
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
    # Phase 3:
    # Decide whether change is safe
    # =====================================================

    def evaluate_change_safety(
        self,
        proposed_change: str,
        developer_answers: Dict,
    ) -> Dict:

        architecture = ask_repo(
            self.repo_url,
            f"""
            Analyze technical impact of:

            {proposed_change}
            """,
        )

        activity = ask_repo_activity(
            self.repo_url,
            f"""
            Analyze historical risks related to:

            {proposed_change}
            """,
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

        {architecture['answer']}

        =====================================
        REPOSITORY HISTORY / ISSUES
        =====================================

        {activity['answer']}

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
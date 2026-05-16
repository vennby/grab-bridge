"""
Gatekeeper Module

Complete gatekeeper ecosystem for repository change governance.

Exports:
- rag_engine: Repository architecture & codebase analysis
- history: Repository activity, issues, PRs, commits
- contracts: CI/CD, API contracts, dependencies, topology
- RepoDefenderAgent: Consolidated orchestrator agent
"""

from .rag_engine import ask_repo, build_repo_index
from .history import ask_repo_activity, build_repo_activity_index
from .contracts import (
    ask_runtime_intelligence,
    build_runtime_intelligence,
    extract_api_contracts,
    analyze_ci_cd_configs,
    build_dependency_graph,
    infer_service_topology,
)
from .defender import RepoDefenderAgent

__all__ = [
    # RAG Engine
    "ask_repo",
    "build_repo_index",
    # History
    "ask_repo_activity",
    "build_repo_activity_index",
    # Contracts
    "ask_runtime_intelligence",
    "build_runtime_intelligence",
    "extract_api_contracts",
    "analyze_ci_cd_configs",
    "build_dependency_graph",
    "infer_service_topology",
    # Defender
    "RepoDefenderAgent",
]

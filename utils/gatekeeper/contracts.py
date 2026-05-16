"""
repo_runtime_intelligence.py

Advanced repository intelligence layer for:

- CI/CD analysis
- API contract extraction
- Dependency graph construction
- Infra understanding
- Service topology inference
- Runtime architecture reasoning

This extends your:
- ask_repo()
- ask_repo_activity()
- RepoDefenderAgent()

into true architecture-aware governance.

=========================================================
REQUIRES
=========================================================

pip install:
    requests
    networkx
    pyyaml
    huggingface_hub

OPTIONAL:
    pip install python-hcl2

ENV:
    HF_API_KEY
    GITHUB_TOKEN

=========================================================
"""

import json
import os
import re
import tempfile
import zipfile
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import networkx as nx
import requests
import yaml
from huggingface_hub import InferenceClient


# =========================================================
# Config
# =========================================================

DEFAULT_CHAT_MODEL = os.environ.get(
    "HF_CHAT_MODEL",
    "deepseek-ai/DeepSeek-V4-Pro:novita",
)

ROOT_DIR = os.path.abspath(
    os.path.dirname(__file__)
)

CACHE_DIR = os.path.join(
    ROOT_DIR,
    ".runtime_intelligence",
)

os.makedirs(CACHE_DIR, exist_ok=True)


# =========================================================
# Github helpers
# =========================================================

def _get_github_headers():

    token = os.environ.get(
        "GITHUB_TOKEN",
        "",
    ).strip()

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "repo-runtime-intelligence",
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def _parse_github_url(repo_url):

    match = re.search(
        r"github.com/([^/]+)/([^/#?]+)",
        repo_url,
    )

    if not match:
        raise ValueError(
            "Invalid GitHub repository URL."
        )

    owner = match.group(1)

    repo = match.group(2).replace(
        ".git",
        "",
    )

    return owner, repo


# =========================================================
# Repo download
# =========================================================

def download_repo(repo_url):

    owner, repo = _parse_github_url(repo_url)

    tmp_dir = tempfile.mkdtemp()

    zip_url = (
        f"https://api.github.com/repos/"
        f"{owner}/{repo}/zipball"
    )

    zip_path = os.path.join(
        tmp_dir,
        "repo.zip",
    )

    with requests.get(
        zip_url,
        headers=_get_github_headers(),
        stream=True,
        timeout=30,
    ) as resp:

        resp.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp_dir)

        top = z.namelist()[0].split("/")[0]

    return os.path.join(tmp_dir, top)


# =========================================================
# File utilities
# =========================================================

def iter_files(root):

    for current_root, dirs, files in os.walk(root):

        dirs[:] = [
            d for d in dirs
            if d not in {
                ".git",
                "node_modules",
                ".venv",
                "dist",
                "build",
            }
        ]

        for file in files:

            yield os.path.join(
                current_root,
                file,
            )


def read_text(path):

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            return f.read()

    except:
        return ""


# =========================================================
# CI/CD CONFIG ANALYZER
# =========================================================

def analyze_ci_cd_configs(
    repo_root,
) -> Dict:

    results = {
        "github_actions": [],
        "dockerfiles": [],
        "kubernetes": [],
        "terraform": [],
        "helm": [],
        "env_vars": set(),
        "deployment_risks": [],
    }

    for path in iter_files(repo_root):

        relative = os.path.relpath(
            path,
            repo_root,
        )

        lower = relative.lower()

        # -----------------------------------------
        # Github Actions
        # -----------------------------------------

        if ".github/workflows" in lower:

            content = read_text(path)

            try:
                workflow = yaml.safe_load(content)

            except:
                workflow = {}

            results["github_actions"].append({
                "file": relative,
                "workflow": workflow.get("name"),
                "triggers": workflow.get("on"),
            })

        # -----------------------------------------
        # Dockerfiles
        # -----------------------------------------

        if "dockerfile" in lower:

            content = read_text(path)

            base_images = re.findall(
                r"FROM\s+([^\s]+)",
                content,
                re.IGNORECASE,
            )

            exposed_ports = re.findall(
                r"EXPOSE\s+([0-9]+)",
                content,
                re.IGNORECASE,
            )

            results["dockerfiles"].append({
                "file": relative,
                "base_images": base_images,
                "ports": exposed_ports,
            })

        # -----------------------------------------
        # Kubernetes
        # -----------------------------------------

        if lower.endswith((
            ".yaml",
            ".yml",
        )):

            content = read_text(path)

            if any(
                x in content
                for x in [
                    "apiVersion:",
                    "kind:",
                    "Deployment",
                    "Service",
                ]
            ):

                try:
                    docs = list(
                        yaml.safe_load_all(content)
                    )

                except:
                    docs = []

                for doc in docs:

                    if not isinstance(doc, dict):
                        continue

                    results["kubernetes"].append({
                        "file": relative,
                        "kind": doc.get("kind"),
                        "name": (
                            doc.get("metadata", {})
                            .get("name")
                        ),
                    })

        # -----------------------------------------
        # Terraform
        # -----------------------------------------

        if lower.endswith(".tf"):

            results["terraform"].append({
                "file": relative,
            })

        # -----------------------------------------
        # Helm
        # -----------------------------------------

        if "chart.yaml" in lower:

            results["helm"].append({
                "file": relative,
            })

        # -----------------------------------------
        # ENV vars
        # -----------------------------------------

        content = read_text(path)

        envs = re.findall(
            r'os\.environ\["([^"]+)"\]',
            content,
        )

        envs += re.findall(
            r'process\.env\.([A-Z0-9_]+)',
            content,
        )

        for e in envs:
            results["env_vars"].add(e)

    results["env_vars"] = list(
        results["env_vars"]
    )

    # ---------------------------------------------
    # Risk heuristics
    # ---------------------------------------------

    if not results["github_actions"]:
        results["deployment_risks"].append(
            "No CI/CD workflows detected."
        )

    if not results["kubernetes"]:
        results["deployment_risks"].append(
            "No deployment manifests detected."
        )

    return results


# =========================================================
# API CONTRACT EXTRACTION
# =========================================================

def extract_api_contracts(
    repo_root,
) -> Dict:

    contracts = {
        "rest_endpoints": [],
        "graphql": [],
        "grpc": [],
        "events": [],
    }

    route_patterns = [
        r'@app\.(get|post|put|delete|patch)\("([^"]+)"',
        r'router\.(get|post|put|delete|patch)\("([^"]+)"',
        r'@router\.(get|post|put|delete|patch)\("([^"]+)"',
    ]

    for path in iter_files(repo_root):

        lower = path.lower()

        content = read_text(path)

        # -----------------------------------------
        # REST
        # -----------------------------------------

        for pattern in route_patterns:

            matches = re.findall(
                pattern,
                content,
                re.IGNORECASE,
            )

            for method, route in matches:

                contracts["rest_endpoints"].append({
                    "method": method.upper(),
                    "route": route,
                    "file": path,
                })

        # -----------------------------------------
        # GraphQL
        # -----------------------------------------

        if "graphql" in lower:

            queries = re.findall(
                r"type\s+Query\s+\{([^}]+)\}",
                content,
                re.DOTALL,
            )

            contracts["graphql"].append({
                "file": path,
                "queries": queries,
            })

        # -----------------------------------------
        # gRPC
        # -----------------------------------------

        if lower.endswith(".proto"):

            services = re.findall(
                r"service\s+([A-Za-z0-9_]+)",
                content,
            )

            rpc_calls = re.findall(
                r"rpc\s+([A-Za-z0-9_]+)",
                content,
            )

            contracts["grpc"].append({
                "file": path,
                "services": services,
                "rpc_calls": rpc_calls,
            })

        # -----------------------------------------
        # Events
        # -----------------------------------------

        event_matches = re.findall(
            r"(publish|emit|produce)\([\"']([^\"']+)",
            content,
            re.IGNORECASE,
        )

        for _, event_name in event_matches:

            contracts["events"].append({
                "event": event_name,
                "file": path,
            })

    return contracts


# =========================================================
# DEPENDENCY GRAPH
# =========================================================

def build_dependency_graph(
    repo_root,
):

    graph = nx.DiGraph()

    python_imports = re.compile(
        r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)",
        re.MULTILINE,
    )

    js_imports = re.compile(
        r'import .* from [\'"](.+)[\'"]',
    )

    for path in iter_files(repo_root):

        lower = path.lower()

        if not lower.endswith((
            ".py",
            ".js",
            ".ts",
            ".tsx",
        )):
            continue

        content = read_text(path)

        relative = os.path.relpath(
            path,
            repo_root,
        )

        graph.add_node(relative)

        # -----------------------------------------
        # Python imports
        # -----------------------------------------

        for imp in python_imports.findall(
            content
        ):

            graph.add_edge(
                relative,
                imp,
            )

        # -----------------------------------------
        # JS imports
        # -----------------------------------------

        for imp in js_imports.findall(
            content
        ):

            graph.add_edge(
                relative,
                imp,
            )

    return graph


# =========================================================
# SERVICE TOPOLOGY INFERENCE
# =========================================================

def infer_service_topology(
    repo_root,
) -> Dict:

    topology = defaultdict(list)

    service_patterns = [
        "service",
        "api",
        "gateway",
        "worker",
        "consumer",
    ]

    for path in iter_files(repo_root):

        relative = os.path.relpath(
            path,
            repo_root,
        )

        parts = relative.split(os.sep)

        for part in parts:

            lower = part.lower()

            if any(
                pattern in lower
                for pattern in service_patterns
            ):

                topology[part].append(
                    relative
                )

    return dict(topology)


# =========================================================
# MASTER ANALYZER
# =========================================================

def build_runtime_intelligence(
    repo_url,
) -> Dict:

    repo_root = download_repo(
        repo_url
    )

    ci_cd = analyze_ci_cd_configs(
        repo_root
    )

    contracts = extract_api_contracts(
        repo_root
    )

    dependency_graph = build_dependency_graph(
        repo_root
    )

    topology = infer_service_topology(
        repo_root
    )

    return {
        "repo_url": repo_url,
        "ci_cd": ci_cd,
        "contracts": contracts,
        "dependency_graph": {
            "nodes": list(
                dependency_graph.nodes()
            ),
            "edges": list(
                dependency_graph.edges()
            ),
        },
        "service_topology": topology,
    }


# =========================================================
# RUNTIME REASONING
# =========================================================

def ask_runtime_intelligence(
    repo_url,
    question,
) -> Dict:

    intelligence = build_runtime_intelligence(
        repo_url
    )

    token = os.environ.get(
        "HF_API_KEY",
        "",
    ).strip()

    if not token:
        raise RuntimeError(
            "HF_API_KEY required."
        )

    client = InferenceClient(
        api_key=token
    )

    system_prompt = """
    You are an expert distributed systems
    architecture analyst.

    Analyze:
    - CI/CD risks
    - deployment risks
    - API compatibility risks
    - service dependencies
    - infra risks
    - contract breaking risks
    - runtime coupling
    - event ordering issues
    - scaling concerns
    - migration risks

    Use ONLY provided context.
    """

    user_prompt = f"""
    =====================================
    RUNTIME INTELLIGENCE
    =====================================

    {json.dumps(intelligence, indent=2)}

    =====================================
    QUESTION
    =====================================

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
        answer = message.get(
            "content",
            "",
        ).strip()

    else:
        answer = getattr(
            message,
            "content",
            "",
        ).strip()

    return {
        "answer": answer,
        "runtime_intelligence": intelligence,
    }


# =========================================================
# Example
# =========================================================

if __name__ == "__main__":

    repo = (
        "https://github.com/grab/engineering-blog"
    )

    result = ask_runtime_intelligence(
        repo,
        """
        What are the main CI/CD and deployment risks in this repo?
        """,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

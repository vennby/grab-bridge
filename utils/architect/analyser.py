import os
import re
import json
import time
import requests
import logging
from typing import Dict, List, Tuple

from huggingface_hub import InferenceClient
from requests.exceptions import Timeout, ConnectionError, RequestException


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("org-indexer")


# =========================================================
# CONFIG
# =========================================================

GITHUB_API = "https://api.github.com"

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx",
    ".java", ".go", ".rs",
}

PY_FUNC = re.compile(
    r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\):"
)

JS_FUNC = re.compile(
    r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)"
)


# =========================================================
# GITHUB HELPERS
# =========================================================

def _headers():
    token = os.environ.get("GITHUB_TOKEN", "").strip()

    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "org-indexer-progress",
    }

    if token:
        h["Authorization"] = f"Bearer {token}"

    return h


def _parse(repo_url: str):
    m = re.search(r"github.com/([^/]+)/([^/#?]+)", repo_url)
    return m.group(1), m.group(2).replace(".git", "")


# =========================================================
# PROGRESS PRINT HELPERS
# =========================================================

def _line(msg: str):
    print(f"\r{msg}", end="", flush=True)


def _done(msg: str):
    print(f"\r{msg}" + " " * 20)


# =========================================================
# TREE FETCH
# =========================================================

def _fetch_tree(owner: str, repo: str):
    """Fetch repository tree with graceful fallback for different branches."""
    
    _line(f"[TREE] Fetching {repo}...")

    branches = ["main", "master", "develop", "dev"]
    
    for branch in branches:
        try:
            url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
            r = requests.get(url, headers=_headers(), timeout=30)
            
            if r.status_code == 200:
                tree = r.json().get("tree", [])
                _done(f"[TREE] {repo} ({branch}): {len(tree)} items loaded")
                return tree
            elif r.status_code == 404:
                continue
            elif r.status_code == 403:
                logger.warning(f"[{repo}] Access forbidden (rate limited or private): {r.status_code}")
                return []
            else:
                logger.warning(f"[{repo}] Unexpected status code {r.status_code} on branch {branch}")
                continue
                
        except Timeout:
            logger.warning(f"[{repo}] Timeout fetching tree from branch {branch}")
            continue
        except ConnectionError:
            logger.warning(f"[{repo}] Connection error fetching tree from branch {branch}")
            continue
        except RequestException as e:
            logger.warning(f"[{repo}] Request exception on branch {branch}: {str(e)}")
            continue
        except Exception as e:
            logger.warning(f"[{repo}] Unexpected error fetching tree from branch {branch}: {str(e)}")
            continue
    
    logger.error(f"[{repo}] Failed to fetch tree from any branch")
    return []


# =========================================================
# FILE FETCH
# =========================================================

def _fetch_file(owner: str, repo: str, path: str):
    """Fetch file content with graceful fallback for different branches."""
    
    branches = ["main", "master"]
    
    for branch in branches:
        try:
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
            r = requests.get(url, timeout=20)
            
            if r.status_code == 200:
                return r.text
            elif r.status_code == 404:
                continue
            elif r.status_code == 403:
                logger.debug(f"[{repo}] Access forbidden for {path}")
                return ""
            else:
                logger.debug(f"[{repo}] Status {r.status_code} fetching {path}")
                continue
                
        except Timeout:
            logger.debug(f"[{repo}] Timeout fetching {path}")
            continue
        except ConnectionError:
            logger.debug(f"[{repo}] Connection error fetching {path}")
            continue
        except RequestException as e:
            logger.debug(f"[{repo}] Request exception fetching {path}: {str(e)}")
            continue
        except Exception as e:
            logger.debug(f"[{repo}] Unexpected error fetching {path}: {str(e)}")
            continue
    
    return ""


# =========================================================
# FUNCTION EXTRACTION
# =========================================================

def _extract(content: str, path: str):

    funcs = []

    if path.endswith(".py"):
        for m in PY_FUNC.finditer(content):
            funcs.append((m.group(1), m.group(2), "python"))

    elif path.endswith((".js", ".ts", ".tsx")):
        for m in JS_FUNC.finditer(content):
            funcs.append((m.group(1), m.group(2), "js"))

    return funcs


# =========================================================
# MAIN INDEXER WITH LIVE PROGRESS
# =========================================================

def build_org_function_index_tree_api_progress(org: str):

    print(f"\n🚀 Starting org scan: {org}\n")

    all_functions = []
    stats = {
        "total_repos": 0,
        "processed_repos": 0,
        "failed_repos": 0,
        "total_files": 0,
        "processed_files": 0,
        "failed_files": 0,
        "total_functions": 0,
        "failed_extractions": 0,
    }

    # =====================================================
    # FETCH REPOS LIST
    # =====================================================

    try:
        repos_resp = requests.get(
            f"{GITHUB_API}/orgs/{org}/repos?per_page=100",
            headers=_headers(),
            timeout=30,
        )
        repos_resp.raise_for_status()
        repos = repos_resp.json()
        stats["total_repos"] = len(repos)
        print(f"📦 Repositories found: {len(repos)}\n")

    except Timeout:
        logger.error(f"Timeout fetching repositories for org: {org}")
        print(f"\n❌ FAILED to fetch repos for {org} (timeout)\n")
        return stats

    except ConnectionError:
        logger.error(f"Connection error fetching repositories for org: {org}")
        print(f"\n❌ FAILED to fetch repos for {org} (connection error)\n")
        return stats

    except RequestException as e:
        logger.error(f"Request exception fetching repositories: {str(e)}")
        print(f"\n❌ FAILED to fetch repos for {org} (request error)\n")
        return stats

    except Exception as e:
        logger.error(f"Unexpected error fetching repositories: {str(e)}")
        print(f"\n❌ FAILED to fetch repos for {org} (unexpected error)\n")
        return stats

    # =====================================================
    # REPO LOOP
    # =====================================================

    for i, repo in enumerate(repos, 1):

        repo_name = repo["name"]
        owner = org

        try:
            print(f"\n📁 [{i}/{len(repos)}] Repo: {repo_name}")

            # Try to fetch tree
            tree = _fetch_tree(owner, repo_name)

            if not tree:
                logger.warning(f"[{repo_name}] Empty tree returned, skipping")
                stats["failed_repos"] += 1
                _done(f"[{repo_name}] ⚠️  SKIPPED (empty or inaccessible)")
                continue

            files = [
                t["path"]
                for t in tree
                if t["type"] == "blob"
                and os.path.splitext(t["path"])[1] in ALLOWED_EXTENSIONS
            ]

            total_files = len(files)
            stats["total_files"] += total_files

            if total_files == 0:
                logger.info(f"[{repo_name}] No code files found")
                stats["processed_repos"] += 1
                _done(f"[{repo_name}] ✓ DONE (0 code files)")
                continue

            print(f"[{repo_name}] Code files: {total_files}")

            repo_func_count = 0
            repo_failed_files = 0

            # =================================================
            # FILE LOOP
            # =================================================

            for j, path in enumerate(files, 1):

                _line(f"[{repo_name}] Processing {j}/{total_files} files | funcs: {repo_func_count}")

                try:
                    content = _fetch_file(owner, repo_name, path)

                    if not content:
                        repo_failed_files += 1
                        continue

                    # Try to extract functions
                    try:
                        funcs = _extract(content, path)

                        for name, args, lang in funcs:
                            try:
                                all_functions.append({
                                    "org": org,
                                    "repo": repo_name,
                                    "file": path,
                                    "function": name,
                                    "signature": f"{name}({args})",
                                    "language": lang,
                                    "context": content[:200] if content else "",
                                })
                                repo_func_count += 1
                                stats["total_functions"] += 1

                            except Exception as e:
                                logger.debug(f"[{repo_name}/{path}] Error processing function {name}: {str(e)}")
                                stats["failed_extractions"] += 1
                                continue

                    except Exception as e:
                        logger.debug(f"[{repo_name}/{path}] Error extracting functions: {str(e)}")
                        stats["failed_extractions"] += 1
                        continue

                except Timeout:
                    logger.debug(f"[{repo_name}] Timeout fetching {path}")
                    repo_failed_files += 1
                    continue

                except ConnectionError:
                    logger.debug(f"[{repo_name}] Connection error fetching {path}")
                    repo_failed_files += 1
                    continue

                except RequestException as e:
                    logger.debug(f"[{repo_name}] Request exception fetching {path}: {str(e)}")
                    repo_failed_files += 1
                    continue

                except Exception as e:
                    logger.debug(f"[{repo_name}] Unexpected error processing {path}: {str(e)}")
                    repo_failed_files += 1
                    continue

            stats["processed_files"] += (total_files - repo_failed_files)
            stats["failed_files"] += repo_failed_files
            stats["processed_repos"] += 1

            _done(f"[{repo_name}] ✓ DONE | functions: {repo_func_count} | failed files: {repo_failed_files}")

            time.sleep(0.2)

        except Exception as e:
            logger.error(f"[{repo_name}] Critical error processing repo: {str(e)}")
            stats["failed_repos"] += 1
            _done(f"[{repo_name}] ❌ FAILED (critical error)")
            continue

    # =====================================================
    # SAVE OUTPUT
    # =====================================================

    out = f"{org}_function_index.jsonl"

    try:
        with open(out, "w", encoding="utf-8") as f:
            for item in all_functions:
                f.write(json.dumps(item) + "\n")

        print("\n\n🎉 COMPLETE")
        print(f"Output file: {out}\n")

    except Exception as e:
        logger.error(f"Failed to write output file: {str(e)}")
        print(f"\n❌ Failed to write output file: {str(e)}\n")

    # =====================================================
    # STATISTICS
    # =====================================================

    print("=" * 60)
    print("📊 INDEXING STATISTICS")
    print("=" * 60)
    print(f"Total repositories: {stats['total_repos']}")
    print(f"Successfully processed: {stats['processed_repos']}")
    print(f"Failed/skipped: {stats['failed_repos']}")
    print(f"Total code files found: {stats['total_files']}")
    print(f"Successfully processed files: {stats['processed_files']}")
    print(f"Failed files: {stats['failed_files']}")
    print(f"Total functions extracted: {stats['total_functions']}")
    print(f"Failed extractions: {stats['failed_extractions']}")
    print("=" * 60 + "\n")

    return {
        "org": org,
        "functions": len(all_functions),
        "output": out,
        "stats": stats,
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    build_org_function_index_tree_api_progress("grab")
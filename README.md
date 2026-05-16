# grab-bridge

## Repo RAG utilities

Set environment variables:

- `HF_API_KEY` for embeddings and chat completions.
- `HF_EMBED_MODEL` (optional, default `sentence-transformers/all-MiniLM-L6-v2`).
- `HF_CHAT_MODEL` (optional, default `deepseek-ai/DeepSeek-V4-Pro:novita`).
- `GITHUB_TOKEN` (optional, raises GitHub API limits).

Index a repository:

```bash
curl -X POST http://localhost:5000/api/rag/index \
	-H "Content-Type: application/json" \
	-d '{"repo_url":"https://github.com/vennby/hsbc-hackathon"}'
```

Ask a question:

```bash
curl -X POST http://localhost:5000/api/rag/ask \
	-H "Content-Type: application/json" \
	-d '{"repo_url":"https://github.com/owner/repo","question":"What does the API layer do?"}'
```

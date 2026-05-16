const statusEl = document.getElementById("status");
const gridEl = document.getElementById("repoGrid");
const scrollBtn = document.getElementById("scrollToRepos");

scrollBtn?.addEventListener("click", () => {
    document.getElementById("repos").scrollIntoView({ behavior: "smooth" });
});

function formatDate(iso) {
    if (!iso) return "";
    const date = new Date(iso);
    return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function setStatus(title, body) {
    statusEl.innerHTML = `
    <div class="status-card">
      <div class="status-title">${title}</div>
      <div class="status-body">${body}</div>
    </div>
  `;
}

function renderRepos(repos) {
    gridEl.innerHTML = repos
        .map(
            (repo, index) => `
        <article class="card" style="animation-delay: ${Math.min(index * 0.05, 0.8)}s;">
          <h3><a href="${repo.html_url}" target="_blank" rel="noreferrer">${repo.name}</a></h3>
          <p>${repo.description || "A mysterious cart of code, awaiting brave contributors."}</p>
          <div class="card-meta">
            <span class="spark">★ ${repo.stargazers_count ?? 0}</span>
            <span>Updated ${formatDate(repo.updated_at)}</span>
          </div>
        </article>
      `
        )
        .join("");
}

async function loadRepos() {
    try {
        setStatus("System booting...", "Linking to the GitHub gate.");
        const response = await fetch("/api/repos");
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "API failed");
        }
        setStatus("Gate stabilized.", `Displaying ${payload.repos.length} repositories.`);
        renderRepos(payload.repos);
    } catch (error) {
        setStatus(
            "Gate access denied.",
            "Check your network or set a GITHUB_TOKEN."
        );
    }
}

loadRepos();

# Agentic RAG

[![CI](https://github.com/muralikarlapalem-cmd/agentic-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/muralikarlapalem-cmd/agentic-rag/actions/workflows/ci.yml)

PDF-first RAG with an agentic fallback. Upload a PDF and ask questions. If the PDF
answers it, you get a cited answer from the PDF only. If not, a LangChain tool-calling
agent picks between **Tavily** (web), **Wikipedia** and **arXiv**. The badge on each
answer shows the tools the agent actually called.

```
question ─► FAISS search over PDF chunks
              │
   best cosine ≥ threshold AND LLM says [[ANSWERABLE]]?
        yes ──┴── no
         │         └─► AgentExecutor ─► Tavily | Wikipedia | arXiv
         ▼                                   (badge = tools really called)
   answer from PDF only, with page cites
```

Everything lives in `app.py`. Indexes persist to `/data`, so a restart keeps your PDF
and re-uploading the same file skips re-embedding.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # add OPENAI_API_KEY (TAVILY_API_KEY optional)
streamlit run app.py        # http://localhost:8501
pytest tests -q             # offline, no API keys needed
```

## Deploy on AWS EC2

### What you get
- HTTPS with an automatically renewed certificate (Caddy + Let's Encrypt)
- Password gate with login lockout (5 failures / 15 min) and a per-client question
  limit (default 30/hour) to cap your OpenAI spend
- Secrets read from AWS SSM Parameter Store, never committed or typed onto the box
- API keys stay on the server and are never sent to the browser
- Container runs as non-root with capabilities dropped, memory-capped, logs rotated,
  restarts automatically, and has a health check
- CI (GitHub Actions) runs the tests, builds the image and checks it starts healthy

### 1. Instance and network
- Ubuntu 24.04 or Amazon Linux 2023, **at least 2 GB RAM** (t3.small or larger).
- Security group: **80 and 443** from anywhere, **22 from your own IP only**.
  Do not open 8501.
- Attach an **Elastic IP**, then point a DNS **A record** for your hostname at it.
  (No domain? Skip HTTPS and use the SSH tunnel at the bottom.)

### 2. Secrets in SSM Parameter Store
```bash
aws ssm put-parameter --name /agentic-rag/OPENAI_API_KEY --type SecureString --value 'sk-...'
aws ssm put-parameter --name /agentic-rag/TAVILY_API_KEY --type SecureString --value 'tvly-...'
aws ssm put-parameter --name /agentic-rag/APP_PASSWORD   --type SecureString --value 'a long passphrase'
```
Create an IAM role for the instance using `deploy/iam-policy.json` (fill in your region
and account ID) and attach it under *EC2 → Actions → Security → Modify IAM role*.

### 3. Deploy
```bash
git clone <your-repo-url> && cd <your-repo>
sudo ./deploy/setup-ec2.sh --ssm-prefix /agentic-rag --domain rag.example.com
```
The script installs Docker, pulls the secrets into a root-only `.env`, builds the image,
starts the stack and waits for the health check. It is safe to re-run.

Without SSM: create `.env` from `.env.example` (`chmod 600 .env`) and run
`sudo ./deploy/setup-ec2.sh --domain rag.example.com`.

### 4. Verify
1. Open `https://rag.example.com`, log in with `APP_PASSWORD`.
2. Upload a text PDF, ask something it covers → **PDF** badge, page cites.
3. Ask "What is the latest version of LangChain?" → **Tavily Web** badge.
4. `sudo docker compose restart app`, reload → your PDF is still indexed.

### Operations
| Task | Command |
|---|---|
| Logs | `sudo docker compose logs -f app` |
| Update | `git pull && sudo ./deploy/setup-ec2.sh --ssm-prefix /agentic-rag --domain <host>` |
| Roll back | `git checkout <previous-commit>` then re-run the script |
| Status | `sudo docker compose --profile https ps` (look for `healthy`) |
| Start anything stopped | `sudo docker compose --profile https up -d` |
| Rotate a key | `aws ssm put-parameter --overwrite ...` then re-run the script |

Set a monthly **spend limit in the OpenAI dashboard** as a backstop to the in-app
rate limit. Indexes in the `rag-data` volume can be rebuilt by re-uploading the PDF,
so they are not backed up.

### Stopping and starting the instance
To save money, stop the instance from the EC2 console when you're not using it
(**Instance state → Stop**) and start it again later (**Instance state → Start**).
Nothing else is needed:

- Docker starts at boot and both containers use `restart: unless-stopped`, so the app and
  Caddy come back on their own 1–2 minutes after the instance shows *Running*.
- The Elastic IP keeps the same address; the loaded PDF and the HTTPS certificate are kept
  in Docker volumes.
- Don't run `docker compose stop` or `down` first: containers stopped that way stay stopped.
  If the site isn't up after 5 minutes, run `sudo docker compose --profile https up -d`.

### Private access without a domain
Run `sudo ./deploy/setup-ec2.sh --ssm-prefix /agentic-rag` (no `--domain`). The app listens
on 127.0.0.1 only; reach it with `ssh -L 8501:localhost:8501 <user>@<instance-ip>` and open
http://localhost:8501. Don't expose it over plain HTTP: the password would travel unencrypted.

## Behaviour notes
- **Threshold** (sidebar, default 0.35) is cosine similarity tuned for `text-embedding-3-small`.
  The retrieved-chunks panel shows the score; adjust if answers route the wrong way.
- The PDF answer must carry an explicit `[[ANSWERABLE]]` verdict from the model, otherwise
  the question goes to the agent.
- The agent must call a tool. If it doesn't, the answer is labelled "No Source" with a warning.
- Without a Tavily key, web search is disabled (Wikipedia and arXiv only).
- Scanned (image-only) PDFs are rejected with a message; run OCR first.
- The 5 nearest chunks are retrieved, but only those within 0.15 of the best score are sent
  to the model. The chunks panel shows each chunk's score and whether it was sent.
- Answers render as Markdown (bold, lists, links). Raw HTML from the model or tools is never rendered.
- **Remove PDF** in the sidebar unloads the document and deletes its saved index from the server.
- One active PDF per server (the most recent). Multi-user isolation is out of scope.

# Multi-Agentic RAG System

A multi-agent Retrieval-Augmented Generation system built with FastAPI, Qdrant, PostgreSQL, and LLM providers (OpenAI / Anthropic / Google Gemini). Agents are orchestrated via Kahn's topological sort, enabling parallel execution where dependencies allow and can handle any type of order execution of agents. Dynamic routing of agents according to query need.

### Dynamic routing of agents with TopoSort without using any frameworks like LangChain or LangGraph

| Experience | URL |
|-----------|-----|
| Chat App | https://toposort-rag-1.onrender.com |
| Database Dashboard | https://toposort-rag-1.onrender.com/api/dashboard |
| Swagger Docs | https://toposort-rag-1.onrender.com/docs |


### Frontend:
- http://127.0.0.1:8000/
![Login](images/frontend_login.png)
![Register](images/frontend_register.png)
![Chat](images/frontend-chat.png)


- http://localhost:8080/dashboard.html  
![Database Dashboard](images/dashboard.png)

- http://localhost:8000/shared/{id}
![Share Chat](images/share.png)


### Backend:
- http://localhost:8000/docs#/
![Backend Screenshot](images/backend.png)
![Backend Screenshot](images/backend_2.png)
![Backend Screenshot](images/backend_3.png)

---

## How to Run

### Prerequisites

- **Python 3.11+**
- **Docker** (for PostgreSQL, Qdrant, Redis)
- API keys: **OpenAI** (required), **Anthropic** (optional), **Google Gemini** (optional), **Tavily** (optional, for web search)

### 1. Start Infrastructure (PostgreSQL + Qdrant + Redis)

```bash
docker compose up -d
```

This starts three containers:

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL 16 | `5432` | Users, sessions, conversations, messages, documents, HITL, etc. |
| Qdrant | `6333` / `6334` | Vector storage for document embeddings |
| Redis 7 | `6379` | Celery task broker + result backend + Pub/Sub for SSE status updates |


### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# ── Required ─────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/mrag

# ── Optional — LLM providers ────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
TAVILY_API_KEY=tvly-...

# ── Optional — Security (change in production) ──────────────────
JWT_SECRET=your-secret-key
OAUTH_STATE_SECRET=your-oauth-secret
TOKEN_ENCRYPTION_KEY=your-fernet-key

# ── Optional — Gmail OAuth ──────────────────────────────────────
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...

# ── Optional — GitHub OAuth ─────────────────────────────────────
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...


S3_ENDPOINT=https://s3.us-east-005.backblazeb2.com
S3_REGION=us-east-005
S3_ACCESS_KEY_ID=005241.......
S3_SECRET_ACCESS_KEY=K005oBJ......
S3_BUCKET=linkdrop


# ── Redis / Celery (defaults work with docker-compose) ──────────
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

### 4. Start the Backend (FastAPI)

```bash
python main.py
```

The server starts on **http://localhost:8000**. The frontend is served at the root (`/`), and the API docs are at `/docs`.

### 5. Start the Celery Worker (Document Processing)

In a **separate terminal** (with the venv activated):

```bash
# Windows (solo pool required)
python -m celery -A celery_app worker --loglevel=info --pool=solo
 OR 
python -m celery -A celery_app worker --pool=threads --concurrency=4


# Linux / macOS
celery -A celery_app worker --loglevel=info --concurrency=4
```

The Celery worker processes uploaded documents asynchronously. Without it, file uploads will be queued but never processed.

**Worker configuration** (from `celery_app.py`):
- `acks_late=True` — tasks re-queue if the worker crashes
- `task_soft_time_limit=300` — 5 min soft limit per document
- `max_retries=3` — automatic retry with exponential backoff
- `rate_limit=20/m` — prevents embedding API throttling

### 6. Start Celery Beat (Scheduled Jobs)

In a **separate terminal** (with the venv activated):

```bash
# Required for scheduled/cron jobs — uses RedBeat (Redis-backed dynamic scheduler)
celery -A celery_app beat --scheduler=redbeat.RedBeatScheduler --loglevel=info
```

Without Beat running, scheduled jobs will be created and stored in the database, but they will **not** fire at their cron times. You can still trigger them manually via the API or UI.

### 7. Start the Admin Dashboard (Optional)

In a **separate terminal**:

```bash
cd frontend
python dashboard.py
```

Then open **http://localhost:8080/dashboard.html** in your browser.

The dashboard connects to PostgreSQL via Docker and displays all database tables with sorting, search, filtering, column resizing, CSV export, and row detail panels. It's a read-only admin view — no writes.

### Quick Start Summary

```
Terminal 1:  docker compose up -d
Terminal 2:  python main.py                                    # FastAPI  → :8000
Terminal 3:  python -m celery -A celery_app worker --pool=solo # Celery   → processes uploads + jobs
Terminal 4:  celery -A celery_app beat --scheduler=redbeat.RedBeatScheduler  # Beat → cron triggers
Terminal 5:  cd frontend && python dashboard.py                # Dashboard → :8080  (optional)
```

| URL | What |
|-----|------|
| http://localhost:8000 | Chat frontend (register / login / chat / upload) |
| http://localhost:8000/docs | Swagger API docs |
| http://localhost:8080/dashboard.html | Admin DB dashboard |

---

## API Endpoints

All endpoints (except auth and health) require a JWT token in the `Authorization: Bearer <token>` header.

### Auth — `/api/v1/auth`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/register` | No | Register a new user. Body: `{ username, email, password }`. Returns `{ user_id, display_name, email, token }`. |
| `POST` | `/login` | No | Login with email + password. Body: `{ email, password }`. Returns `{ user_id, display_name, email, token }`. |
| `POST` | `/logout` | JWT | Close all active sessions for the authenticated user. Returns `{ status, sessions_closed }`. |

### Core API — `/api/v1`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Health check. Returns `{ status: "ok" }`. |
| `GET` | `/models` | No | List available LLM models grouped by provider with real-time availability (API key validated in parallel). Returns `{ models: { provider: { available, models: [...] } } }`. |
| `GET` | `/agents` | No | List all registered agents and their capabilities. Returns `{ agents: [...] }`. |
| `POST` | `/query` | JWT | Non-streaming query. Body: `{ query, session_id?, conversation_id?, model? }`. Returns `{ answer, sources, agents_used, session_id, conversation_id }`. The optional `model` field overrides the default LLM (e.g. `gpt-4o`, `claude-sonnet-4-20250514`, `gemini-2.0-flash`). |
| `POST` | `/query/stream` | JWT | Streaming query via SSE. Same body as `/query`. Emits SSE events: `status`, `plan`, `hitl_required`, `token`, `sources`, `done`, `error`. |
| `POST` | `/documents/upload` | JWT | Upload one or more files (PDF, DOCX, Excel, CSV, TXT, MD). Multipart form: `files`. Returns `{ documents: [{ doc_id, filename, status }] }`. Processing runs async via Celery. |
| `GET` | `/documents/status` | JWT | List all documents and their processing status for the authenticated user. Returns `{ documents: [...] }`. |
| `GET` | `/documents/status/stream` | JWT | SSE stream of real-time document processing status updates (Redis Pub/Sub). Events: `doc_status`. |
| `DELETE` | `/documents/{doc_id}` | JWT | Delete a document (S3 file + Qdrant vectors + DB record). Ownership verified. Returns `{ deleted, doc_id }`. |
| `POST` | `/hitl/respond` | JWT | Approve or deny a HITL (Human-in-the-Loop) request. Body: `{ request_id, decision, instructions? }`. Returns `{ request_id, status }`. |
| `GET` | `/conversations` | JWT | List conversations (newest first) with pagination. Query params: `limit` (default 20), `offset` (default 0). Returns `{ conversations: [...], total, limit, offset, has_more }`. |
| `GET` | `/conversations/{conversation_id}/messages` | JWT | Load all messages for a specific conversation. Returns `{ conversation_id, messages: [...] }`. |
| `POST` | `/conversations/{conversation_id}/share` | JWT | Generate a shareable read-only link for a conversation. Idempotent — returns existing token if already shared. Returns `{ share_token, share_url }`. |
| `DELETE` | `/conversations/{conversation_id}/share` | JWT | Revoke the share link for a conversation. Returns `{ status: "unshared" }`. |
| `GET` | `/shared/{share_token}` | No | Public read-only page for a shared conversation. No authentication required. |
| `DELETE` | `/conversations/{conversation_id}` | JWT | Delete a conversation and all its messages, agent executions, and summaries. Ownership verified. Returns `{ deleted, conversation_id }`. |
| `PUT` | `/conversations/{conversation_id}/messages/{message_id}/edit` | JWT | Edit a user message and truncate all messages after it. Deletes subsequent messages, agent executions, HITL requests, and stale summaries. Body: `{ content }`. Returns `{ message_id, conversation_id, deleted_messages, deleted_summaries }`. |

### Personas — `/api/v1/personas`

Personas let each user define custom AI personalities that shape the Composer Agent's tone and behavior on a per-conversation basis. Three default personas (**Friend**, **Teacher**, **Lover**) are seeded automatically on first access.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/personas` | JWT | List all personas for the authenticated user (seeds defaults on first call). Returns `{ personas: [...] }`. |
| `POST` | `/personas` | JWT | Create a new persona. Body: `{ name, description }`. Returns `{ persona }`. |
| `PUT` | `/personas/{persona_id}` | JWT | Update an existing persona. Body: `{ name?, description? }`. Returns `{ persona }`. |
| `DELETE` | `/personas/{persona_id}` | JWT | Delete a persona. Returns `{ status }`. |

When a persona is selected in the frontend, its `persona_id` is sent with each query. The Composer Agent receives a system-level instruction derived from the persona's description, influencing the tone and style of the final answer.

### OAuth Connectors — `/api/v1/connectors`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/providers` | No | List all available OAuth connector providers (Gmail, GitHub, etc.) and their configuration status. |
| `GET` | `/connections` | JWT | List all OAuth connections for the authenticated user. |
| `GET` | `/{provider}/auth-url` | JWT | Get the OAuth authorization URL for a provider. Frontend opens this in a popup. Returns `{ auth_url, provider }`. |
| `GET` | `/{provider}/callback` | No | OAuth redirect callback. Exchanges auth code for tokens, stores connection, returns HTML that auto-closes the popup. |
| `DELETE` | `/connections/{connection_id}` | JWT | Disconnect and revoke an OAuth connection. Returns `{ status, connection_id }`. |

### Voice — `/api/v1/voice`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/voice/transcribe` | JWT | Transcribe an audio file via the configured STT provider (default: AssemblyAI). Accepts multipart `audio` file (max 25 MB). Returns `{ text, confidence, language }`. |
| `POST` | `/voice/synthesize` | JWT | Synthesize text to speech via the configured TTS provider (default: AWS Polly Neural). Body: `{ text, voice? }`. Returns `audio/mpeg` binary stream. |
| `POST` | `/tts/speak` | JWT | Text-to-speech for assistant responses. Strips sources, citations, and markdown before synthesis. Body: `{ text, voice? }`. Returns `audio/mpeg` binary stream. |

The `/voice/synthesize` endpoint exists for direct API consumers. The `/tts/speak` endpoint is optimized for chat responses — it automatically cleans the text (removes `Sources:` block, `[N]` citations, markdown formatting) before sending to Polly. In the voice input flow, TTS audio is delivered inline via SSE (`voice_audio` event) — no extra HTTP round-trip needed.

### Analytics — `/api/v1`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/analytics` | JWT | Get per-user agent analytics. Query params: `days` (default 30). Returns `{ agent_usage, token_timeline, response_times, query_patterns, token_breakdown }`. |

The analytics endpoint aggregates data from `agent_executions`, `messages`, and `conversations` tables to provide:
- **Agent Usage** — Most-used agents with success/failure counts.
- **Token Timeline** — Daily token consumption over the requested period.
- **Response Times** — Average, min, and max response times per agent.
- **Query Patterns** — Total executions, success rate, conversation and message counts.
- **Token Breakdown** — Per-agent token totals extracted from the `token_details` JSONB column.

### Scheduled Jobs — `/api/v1/scheduled-jobs`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/` | JWT | Create a new scheduled job with explicit steps. Body: `{ name, description?, cron_expression, timezone?, steps: [{agent_name, task, tools?, depends_on_steps?}], notification_mode?, notification_target? }`. Validates cron syntax and step dependency graph. Syncs to Celery Beat. Returns the created job with steps. |
| `POST` | `/from-prompt` | JWT | Parse natural language into a scheduled job preview using LLM. Body: `{ prompt, timezone?, notification_mode? }`. Returns `{ preview: { name, cron_expression, cron_human, steps, ... } }`. The preview can be reviewed before creating. |
| `GET` | `/` | JWT | List all scheduled jobs for the authenticated user (excludes soft-deleted). Returns array of jobs with steps, schedule info, and last/next run timestamps. |
| `GET` | `/presets` | JWT | Get available cron preset schedules. Returns `{ every_hour: {cron, label}, every_morning: {cron, label}, ... }`. |
| `GET` | `/{job_id}` | JWT | Get a single scheduled job with its steps. Ownership verified. Returns job object. |
| `PUT` | `/{job_id}` | JWT | Update a scheduled job (name, cron, steps, notification, etc.). Body: any subset of `ScheduledJobUpdate` fields. Re-syncs Celery Beat if schedule changes. Returns updated job. |
| `DELETE` | `/{job_id}` | JWT | Soft-delete a scheduled job. Removes from Celery Beat. Returns `{ status: "deleted" }`. |
| `POST` | `/{job_id}/pause` | JWT | Pause an active job. Removes from Celery Beat (stops cron triggers). Returns `{ status: "paused" }`. |
| `POST` | `/{job_id}/resume` | JWT | Resume a paused job. Re-syncs to Celery Beat. Returns `{ status: "active" }`. |
| `POST` | `/{job_id}/trigger` | JWT | Manually trigger a job (run now). Enqueues a Celery task immediately regardless of cron schedule. Returns `{ status: "queued", task_id, job_id }`. |
| `GET` | `/{job_id}/runs` | JWT | List run history (paginated, newest first). Query params: `limit` (default 20), `offset`. Returns array of run objects. |
| `GET` | `/{job_id}/runs/{run_id}` | JWT | Get a single run with per-step results. Returns run object with `step_results` array. |

### Artifacts — `/api/v1`

Code agent generates file artifacts (PDFs, charts, CSVs) that are sent to the frontend as base64-encoded SSE events. Users can optionally save artifacts to S3 for persistent storage.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/artifacts/save` | JWT | Save an artifact to S3.But removed from frontend to acess for no use of saving in cloud. Body: `{ conversation_id, filename, artifact_type, content_type, base64_data, agent_id?, agent_name?, preview_data? }`. Decodes base64, uploads to S3, persists DB row. Returns `{ artifact_id, download_url }`. |
| `GET` | `/artifacts/{artifact_id}/download` | JWT | Get a presigned S3 download URL for a saved artifact. Ownership verified. Returns `{ url, filename, content_type, file_size_bytes, expires_in }`. |
| `GET` | `/conversations/{conversation_id}/artifacts` | JWT | List all saved artifacts for a conversation (used for history reload). Ownership verified. Returns `{ artifacts: [...] }`. |

**SSE Event:** During streaming, artifacts arrive as `artifact_preview` events with `{ agent_id, filename, artifact_type, content_type, file_size_bytes, preview_data, base64_data }`. The frontend renders inline preview cards with a Download button.

---

## Voice Input / Output Architecture

The system supports full voice interaction: **microphone input** (Speech-to-Text) and **spoken responses** (Text-to-Speech). The key design decision is that voice summary generation runs **in parallel** with text streaming — the user hears the audio response while text is still being typed.

### How It Works

```mermaid
flowchart TD
    Mic["🎤 User presses mic button<br/><i>MediaRecorder API</i>"]
    Mic -->|"audio blob"| STT["POST /voice/transcribe<br/><i>AssemblyAI STT</i>"]
    STT -->|"transcribed text"| Stream["POST /query/stream<br/><i>source: 'voice'</i>"]

    Stream --> Build["Build ComposerInput<br/><i>agent results, sources,<br/>memory, persona</i>"]

    Build --> Fork["asyncio — two parallel tasks"]

    Fork --> TextTask
    Fork --> VoiceTask

    subgraph TextTask["Text Streaming (Composer LLM)"]
        direction TB
        TextLLM["Composer LLM call<br/><i>full markdown answer<br/>with citations</i>"]
        TextLLM --> Tokens["SSE token events<br/><i>streamed to frontend</i>"]
    end

    subgraph VoiceTask["Voice Task (asyncio.create_task)"]
        direction TB
        VoiceLLM["Voice Summary LLM call<br/><i>3-4 sentence spoken summary<br/>from raw ComposerInput</i>"]
        VoiceLLM --> TTS["AWS Polly Neural TTS<br/><i>synthesize → MP3 bytes</i>"]
        TTS --> B64["Base64 encode audio"]
    end

    Tokens -->|"each chunk"| Check{"voice_task<br/>.done()?"}
    Check -->|"No"| Continue["Continue streaming<br/>next token"]
    Check -->|"Yes"| Emit["Emit SSE voice_audio event<br/><i>base64 MP3 + content_type</i>"]

    Continue --> Tokens
    Emit --> Play["🔊 Frontend auto-plays audio<br/><i>+ adds replay button</i>"]

    Tokens -->|"streaming done"| Fallback{"Voice task<br/>still pending?"}
    Fallback -->|"Yes"| Await["await voice_task<br/>then emit voice_audio"]
    Fallback -->|"No"| Done["Done — both complete"]
    Await --> Done

    style Mic fill:#2196F3,color:#fff
    style STT fill:#2196F3,color:#fff
    style TextLLM fill:#9C27B0,color:#fff
    style VoiceLLM fill:#FF9800,color:#fff
    style TTS fill:#FF9800,color:#fff
    style Play fill:#4CAF50,color:#fff
    style Emit fill:#4CAF50,color:#fff
```

### Why Parallel, Not Sequential

A sequential approach (wait for full text → summarize → TTS) adds **8-12 seconds** of delay after the text finishes. The parallel design eliminates this:

| Approach | Flow | User-Perceived Latency |
|----------|------|----------------------|
| **Sequential** | Text streaming → Voice LLM → TTS → play | +8-12s after text done |
| **Parallel (our approach)** | Text streaming ‖ (Voice LLM → TTS) → play mid-stream | ~0s — audio arrives during text streaming |

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Voice LLM uses raw `ComposerInput`, not finished text** | The voice summary doesn't depend on the text answer. Both LLM calls receive the same input (agent results, sources, memory, persona) but produce different outputs — rich markdown vs. concise spoken summary. This enables true parallelism. |
| **Mid-stream emission** | After each token chunk, the server checks `voice_task.done()`. The moment TTS finishes, the `voice_audio` SSE event is emitted while text is still streaming. The frontend plays audio immediately. |
| **Fallback await** | If the voice task takes longer than the entire text stream (unlikely), it falls back to `await voice_task` after streaming completes — guaranteeing audio is always delivered. |
| **Base64 in SSE, not separate HTTP** | Audio is embedded in the SSE stream as a base64-encoded MP3. No extra HTTP round-trip, no race conditions, no CORS issues. |
| **Persona-aware voice** | The voice summary prompt includes the active persona's description, so the spoken response matches the persona's tone (casual friend vs. formal teacher). |

### Provider Architecture

STT and TTS use a **provider-agnostic pattern** with abstract base classes:

```
voice/
├── __init__.py              # get_stt_provider(), get_tts_provider()
├── stt/
│   ├── base.py              # BaseSTTProvider (ABC) + STTResult
│   └── assemblyai.py        # AssemblyAI implementation
└── tts/
    ├── base.py              # BaseTTSProvider (ABC) + TTSResult
    └── aws_polly.py         # AWS Polly Neural implementation
```

| Layer | Class | Description |
|-------|-------|-------------|
| **STT Base** | `BaseSTTProvider` | ABC with `transcribe(audio_bytes, content_type, language) → STTResult` |
| **STT Provider** | `AssemblyAISTTProvider` | 3-step async flow: upload audio → create transcript → poll until completed |
| **TTS Base** | `BaseTTSProvider` | ABC with `synthesize(text, voice, language) → TTSResult` |
| **TTS Provider** | `AWSPollyTTSProvider` | Neural engine via aioboto3, configurable voice ID, returns MP3 |

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `STT_PROVIDER` | `assemblyai` | STT provider (`assemblyai`) |
| `ASSEMBLYAI_API_KEY` | — | AssemblyAI API key |
| `TTS_PROVIDER` | `aws_polly` | TTS provider (`aws_polly`) |
| `AWS_ACCESS_KEY_ID` | — | AWS credentials for Polly |
| `AWS_SECRET_ACCESS_KEY` | — | AWS credentials for Polly |
| `AWS_REGION` | `us-east-1` | AWS region |
| `AWS_POLLY_VOICE_ID` | `Matthew` | Polly voice (e.g., `Matthew`, `Joanna`, `Kajal`) |
| `VOICE_MAX_AUDIO_SIZE_MB` | `25` | Max upload size for STT |
| `VOICE_MAX_DURATION_SEC` | `60` | Max recording duration |

---

## System Architecture

```mermaid
graph TB
    User["User"] -->|query / upload| API["FastAPI API Layer"]

    API -->|POST /query| LoadMem["Existing Load Long-Term Memory<br/><i>SELECT from PostgreSQL</i>"]
    API -->|POST /upload| DocPipeline["Document Pipeline"]

    LoadMem --> ParallelBlock

    subgraph ParallelBlock["Parallel (asyncio.gather)"]
        direction LR
        Master["Master Agent<br/><i>plan + route</i>"]
        Extractor["Long Term Memory Extractor<br/><i>LLM detects personal data<br/>in query → upsert to DB</i>"]
    end

    Master -->|execution plan| Orchestrator["⚙️ Orchestrator<br/><i>Kahn's toposort → stages</i>"]
    Extractor -->|"new facts"| PG

    Orchestrator -->|stage 1..N| RAG["RAG Agent"]
    Orchestrator -->|stage 1..N| Code["Code Agent"]
    Orchestrator -->|stage 1..N| Mail["Mail Agent<br/><i>Gmail API</i>"]
    Orchestrator -->|stage 1..N| Web["Web Agent<br/><i>Tavily API</i>"]

    RAG -->|results| State["Shared State"]
    Code -->|results| State
    Mail -->|results| State
    Web -->|results| State

    State -->|all outputs| Composer["Composer Agent<br/><i>synthesise + cite</i>"]
    Composer -->|SSE stream| API
    API -->|streamed answer| User

    DocPipeline -->|vectors| Qdrant[("Qdrant<br/>Vector DB")]
    RAG <-->|search| Qdrant

    subgraph Persistence
        PG[("PostgreSQL")]
        Qdrant
    end

    Memory["Memory Manager"] <--> PG
    Orchestrator <--> Memory
    LoadMem <-.->|"existing memory"| PG
```

---

## Auto-Generated Conversation Titles

New conversations get an LLM-generated title automatically. Instead of showing the raw user query truncated to 120 characters, the system asks the LLM to produce a short, descriptive title (max 7 words) from the first message. Title generation runs **in parallel** with the main pipeline — zero added latency.

### How It Works

```mermaid
flowchart TD
    Query["User sends first message<br/><i>conversation_id = null</i>"]
    Query --> Create["Create new conversation<br/><i>Chat title = query[:120]</i>"]
    Create --> Fork["asyncio — parallel tasks"]

    Fork --> Pipeline["Main Pipeline<br/><i>Master → Orchestrator →<br/>Agents → Composer</i>"]
    Fork --> TitleTask["Title Generation Task"]

    subgraph TitleTask["Title Task (asyncio.create_task)"]
        direction TB
        Prompt["LLM prompt<br/><i>'Generate a short title<br/>(max 7 words) for this message'</i>"]
        Prompt --> Generated["Generated title<br/><i>e.g. 'Q3 Revenue Analysis Report'</i>"]
    end

    Pipeline --> Stream["SSE token events<br/><i>streamed to user</i>"]
    Stream --> Done{"Streaming<br/>complete?"}
    Done --> Await["Await title_task"]
    Await --> Update["UPDATE conversations<br/>SET title = generated_title<br/><i>PostgreSQL</i>"]
    Update --> Emit["SSE done event<br/><i>includes conversation_title</i>"]
    Emit --> Frontend["Frontend updates sidebar<br/><i>conv-item title replaced</i>"]

    style Query fill:#2196F3,color:#fff
    style Prompt fill:#FF9800,color:#fff
    style Generated fill:#FF9800,color:#fff
    style Update fill:#4CAF50,color:#fff
    style Frontend fill:#4CAF50,color:#fff
```

### Flow Summary

| Step | Streaming Route | Non-Streaming Route |
|------|----------------|---------------------|
| **1. Detect new conversation** | `request.conversation_id` is null | Same |
| **2. Launch title task** | `asyncio.create_task(generate_title(...))` — runs in parallel with agents | Same — background task after response |
| **3. Await + persist** | Awaited before `done` event → `UPDATE conversations SET title` | Fire-and-forget `asyncio.create_task` |
| **4. Deliver to frontend** | `done` SSE event includes `conversation_title` field | Next `loadConversations()` call picks it up |
| **5. Update sidebar** | `updateConvItemTitle()` replaces text in the active conv-item | Sidebar refreshes on next load |

### Configuration

The bot name and title generation use settings from `config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `BOT_NAME` | `MRAG` | The chatbot's name — used in system prompts so the bot can identify itself |

Title generation uses whichever LLM model is active for the request (user-selected or default), with `temperature=0.3` for focused output. The prompt and generator live in `core/title_generator.py`.

---

## Multi-Modal Image Analysis

Upload images alongside documents — the system uses a **Vision LLM** to generate a comprehensive text description of each image, then pipes that text through the existing chunk → embed → store pipeline. The RAG agent can then answer questions about uploaded images just like any other document.

### Supported Formats

`.png` · `.jpg` / `.jpeg` · `.gif` · `.webp` · `.bmp`

### Architecture

```mermaid
flowchart TD
    Upload["User uploads image<br/><i>POST /documents/upload</i>"]
    Upload --> S3["Store original in S3<br/><i>uploads/{user_id}/{doc_id}/{filename}</i>"]
    Upload --> Celery["Celery Task<br/><i>process_document_task</i>"]

    Celery --> Parser["parser.parse_document()"]

    Parser --> ExtCheck{"File extension<br/>in _IMAGE_EXTENSIONS?"}
    ExtCheck -->|No| TextParse["Existing text/PDF/CSV<br/>parsers"]
    ExtCheck -->|Yes| Vision["_parse_image()"]

    subgraph Vision["Vision LLM Analysis"]
        direction TB
        Dispatch{"Config:<br/>vision_model_provider"}
        Dispatch -->|openai| OpenAI["OpenAI Vision API<br/><i>base64 image_url<br/>detail: high</i>"]
        Dispatch -->|google| Gemini["Google Gemini API<br/><i>inline_data<br/>mime_type + bytes</i>"]
        OpenAI --> Description["Text description<br/><i>content overview, extracted text,<br/>data/numbers, visual elements</i>"]
        Gemini --> Description
    end

    Description --> Chunk["chunker.chunk_text()<br/><i>split into overlapping chunks</i>"]
    Chunk --> Embed["embedder.embed_chunks()<br/><i>batch embedding</i>"]
    Embed --> Store["vector_store.store_chunks()<br/><i>Qdrant collection<br/>user_{id}_documents</i>"]

    TextParse --> Chunk

    style Upload fill:#2196F3,color:#fff
    style Dispatch fill:#FF9800,color:#fff
    style Description fill:#FF9800,color:#fff
    style Store fill:#4CAF50,color:#fff
```

### How It Works

1. **Upload** — Image file arrives via the existing `/documents/upload` endpoint. The original is stored in S3 (Backblaze B2).
2. **Parse** — `parser.parse_document()` detects the image extension and calls `_parse_image()`.
3. **Vision LLM** — `document_pipeline/vision.py` sends the image to the configured vision model. The prompt asks for: content overview, text extraction (OCR), data/numbers, visual elements, and contextual meaning.
4. **Pipeline continues** — The returned text description flows through **chunker → embedder → vector store**, exactly like any text document.
5. **Query** — The RAG agent retrieves relevant chunks. Image-derived chunks carry `doc_type: "png"` (or `jpg`, etc.) in metadata for traceability.

### Frontend

- The file picker accepts image formats alongside documents.
- Uploaded images show a **thumbnail preview** (loaded via presigned S3 URL) in the file list instead of the generic file icon.

### Configuration

All vision settings live in `config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `VISION_MODEL_PROVIDER` | `openai` | Which provider to use — `openai` or `google` |
| `VISION_MODEL` | `gpt-4o` | The vision-capable model name |
| `VISION_MAX_TOKENS` | `2048` | Maximum tokens for the vision response |

To switch from OpenAI to Gemini, change **only** the config:

```env
VISION_MODEL_PROVIDER=google
VISION_MODEL=gemini-2.0-flash
```

No code changes required — both providers are handled by `document_pipeline/vision.py`.

---

## Web Scraping Collections

Scrape entire websites (with configurable crawl depth) and make the content searchable by the RAG agent — alongside your uploaded documents. Each user can create multiple **web scrape collections**, each containing up to **5 URLs**. Collections have a toggle: when **ON**, the RAG agent includes that collection's content in its search; when **OFF**, it's ignored.

### End-to-End Flow

```mermaid
flowchart TD
    User["User creates collection<br/><i>name + up to 5 URLs<br/>each with depth 1-3</i>"]
    User -->|"POST /web-scrape/collections"| API["FastAPI API Layer"]

    API --> Validate["Validate<br/><i>max 5 URLs, depth 1-3,<br/>valid URL format</i>"]
    Validate --> DB1["Insert into PostgreSQL<br/><i>web_scrape_collections (pending)<br/>+ web_scrape_urls (pending)</i>"]
    DB1 --> Enqueue["Enqueue Celery task<br/><i>scrape_web_collection_task.delay()</i>"]
    Enqueue --> Return["Return collection<br/><i>status: pending</i>"]
    Return -->|"SSE stream"| FE["Frontend updates<br/>collection card in real-time"]

    Enqueue --> Worker["Celery Worker picks up task"]

    subgraph CeleryTask["Celery Worker — scrape_web_collection_task"]
        direction TB
        Loop["For each URL in collection"]
        Loop --> Scrape["scrape_url(url, depth)"]

        subgraph Crawler["BFS Web Crawler (httpx + BeautifulSoup)"]
            direction TB
            Fetch["Fetch page via httpx<br/><i>follow redirects, timeout 15s</i>"]
            Fetch --> Parse["Parse HTML<br/><i>strip nav, footer, scripts,<br/>styles, forms</i>"]
            Parse --> Extract["Extract title + body text"]
            Extract --> Links{"depth > 1?"}
            Links -->|"Yes"| Follow["Follow same-domain links<br/><i>BFS queue, max 20 pages</i>"]
            Follow --> Fetch
            Links -->|"No"| Pages["Return scraped pages"]
        end

        Scrape --> Crawler
        Crawler --> Process["process_scraped_pages()"]

        subgraph Pipeline["Chunk → Embed → Store"]
            direction TB
            Chunk["StructureAwareChunker<br/><i>chunk_size=1024</i>"]
            Chunk --> Embed["OpenAI Embeddings<br/><i>batch embed chunks</i>"]
            Embed --> Store["Qdrant vector_store.add_web_page()<br/><i>source_type: 'web_scrape'<br/>web_collection_id: UUID</i>"]
        end

        Process --> Pipeline
        Pipeline --> Status["Update DB status<br/><i>url → ready / failed</i>"]
        Status --> Publish["Redis Pub/Sub<br/><i>web_scrape_status:{user_id}</i>"]
    end

    Publish -->|"SSE push"| FE

    style User fill:#2196F3,color:#fff
    style Worker fill:#FF9800,color:#fff
    style Fetch fill:#FF9800,color:#fff
    style Store fill:#4CAF50,color:#fff
    style FE fill:#4CAF50,color:#fff
```

### RAG Integration — Toggle ON/OFF

```mermaid
flowchart LR
    Query["User sends query"]
    Query --> FE2["Frontend collects<br/>active collection IDs<br/><i>toggled ON + status ready</i>"]
    FE2 -->|"active_web_collection_ids"| Stream["POST /query/stream"]
    Stream --> Orch["Orchestrator<br/><i>passes IDs in agent metadata</i>"]
    Orch --> RAG["RAG Agent"]

    RAG --> Search["two_level_search()"]

    subgraph Stage1["Stage 1 — Document Entry Search"]
        direction TB
        Filter["Qdrant filter:<br/><i>source_type = 'document'<br/>OR web_collection_id IN [...]</i>"]
        Filter --> Match["Matched document entries<br/><i>uploaded docs + active web pages</i>"]
    end

    Search --> Stage1
    Stage1 --> Stage2["Stage 2 — Hybrid Search<br/><i>dense + BM25 within<br/>matched docs → RRF merge</i>"]
    Stage2 --> Rerank["LLM Reranking<br/><i>(optional)</i>"]
    Rerank --> Synthesis["LLM Synthesis<br/><i>answer with citations</i>"]
    Synthesis --> Answer["Streamed answer<br/><i>includes web sources</i>"]

    style Query fill:#2196F3,color:#fff
    style Filter fill:#FF9800,color:#fff
    style Synthesis fill:#9C27B0,color:#fff
    style Answer fill:#4CAF50,color:#fff
```

### How It Works

1. **Create Collection** — User provides a name and up to 5 URLs with crawl depth (1-3). The API creates DB records and enqueues a Celery task.
2. **Scrape** — The Celery worker crawls each URL using BFS (`httpx` + `BeautifulSoup`), following same-domain links up to the specified depth. Max 20 pages per URL.
3. **Process** — Each scraped page is chunked (`StructureAwareChunker`), embedded (OpenAI), and stored in the **same** Qdrant collection as uploaded documents (`user_{id}_documents`), distinguished by `source_type: "web_scrape"` and `web_collection_id` metadata.
4. **Real-time Updates** — Status changes are published via Redis Pub/Sub → SSE stream. The frontend updates collection/URL cards live as scraping progresses.
5. **Toggle** — Each collection has an `is_active` toggle. When ON, the RAG agent's Stage 1 search includes that collection's document entries via Qdrant `should` filter clauses. When OFF, they're excluded.
6. **Delete** — Deleting a collection removes all Qdrant points (filtered by `web_collection_id`) and cascades the DB delete (collection + URLs).

### Architecture Decision: Same Qdrant Collection

Scraped web content is stored in the **same** `user_{user_id}_documents` Qdrant collection alongside uploaded documents, not in separate per-collection Qdrant collections. This is achieved by adding metadata fields to every point:

| Field | Uploaded Documents | Web Scrape Pages |
|-------|-------------------|-----------------|
| `source_type` | `"document"` | `"web_scrape"` |
| `web_collection_id` | — | `UUID` of the collection |

This avoids managing many tiny Qdrant collections and lets the existing two-level retrieval pipeline work with minimal changes.

### API Endpoints — `/api/v1/web-scrape`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/collections` | JWT | Create a new web scrape collection. Body: `{ name, urls: [{ url, depth }] }`. Max 5 URLs, depth 1-3. Returns collection with pending status. Enqueues Celery scrape task. |
| `GET` | `/collections` | JWT | List all web scrape collections for the authenticated user (with URL details). |
| `PUT` | `/collections/{id}/toggle` | JWT | Toggle a collection's active status. Body: `{ is_active: bool }`. Returns `{ collection_id, is_active }`. |
| `DELETE` | `/collections/{id}` | JWT | Delete a collection (Qdrant vectors + DB records). Ownership verified. |
| `POST` | `/collections/{id}/rescrape` | JWT | Re-trigger scraping for an existing collection. Resets statuses and enqueues a new Celery task. |
| `GET` | `/status/stream` | JWT | SSE stream of real-time web scrape status updates via Redis Pub/Sub. |

### Database Schema

```sql
-- Collections
CREATE TABLE web_scrape_collections (
    collection_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name              VARCHAR(256) NOT NULL,
    is_active         BOOLEAN NOT NULL DEFAULT FALSE,
    status            VARCHAR(16) NOT NULL DEFAULT 'pending',
                      -- pending | scraping | ready | partial | failed
    total_pages       INT DEFAULT 0,
    total_chunks      INT DEFAULT 0,
    error_message     TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Individual URLs within a collection
CREATE TABLE web_scrape_urls (
    url_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id     UUID NOT NULL REFERENCES web_scrape_collections(collection_id) ON DELETE CASCADE,
    url               TEXT NOT NULL,
    depth             INT NOT NULL DEFAULT 1,
    status            VARCHAR(16) NOT NULL DEFAULT 'pending',
                      -- pending | scraping | ready | failed
    pages_scraped     INT DEFAULT 0,
    chunks_created    INT DEFAULT 0,
    error_message     TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### Collection Status Flow

```
pending → scraping → ready      (all URLs succeeded)
                   → partial    (some URLs succeeded, some failed)
                   → failed     (all URLs failed or timed out)
```

### Scraper Limits

| Setting | Value | Purpose |
|---------|-------|---------|
| Max URLs per collection | 5 | Prevent abuse |
| Max crawl depth | 3 | Limit crawl scope |
| Max pages per URL | 20 | Prevent runaway crawling |
| Request timeout | 15s | Per-page HTTP timeout |
| Celery rate limit | 10/m | Prevent overloading external sites |
| Celery max retries | 2 | Auto-retry with exponential backoff |

---

## Document Chat Mode — Scoped RAG Search

Document Chat Mode lets users **select specific documents** from the Files panel to scope their RAG queries. Instead of searching across all uploaded documents, the system restricts retrieval to only the selected files — ideal for asking questions about a specific report, contract, or dataset.

### End-to-End Flow

```mermaid
sequenceDiagram
    participant U as User (Frontend)
    participant API as FastAPI Streaming
    participant Orch as Orchestrator
    participant RAG as RAG Agent
    participant Tool as rag_tools.py
    participant Q as Qdrant

    U->>U: Select docs via checkboxes in Files panel
    Note over U: Scope bar shows "Scoped to N document(s)"
    U->>API: POST /query/stream<br/>{ query, selected_doc_ids: [id1, id2] }
    API->>Orch: context includes selected_doc_ids
    Orch->>RAG: metadata.selected_doc_ids = [id1, id2]
    RAG->>Tool: two_level_search(selected_doc_ids=[id1, id2])

    alt selected_doc_ids provided
        Note over Tool: Skip Stage 1 (document discovery)<br/>Go directly to Stage 2
        Tool->>Q: Hybrid search (dense + BM25)<br/>filtered to selected doc_ids only
    else no doc selection
        Note over Tool: Normal two-stage retrieval
        Tool->>Q: Stage 1: Discover relevant documents
        Tool->>Q: Stage 2: Hybrid search across discovered docs
    end

    Q-->>Tool: Matching chunks
    Tool-->>RAG: { chunks, matched_documents }
    RAG-->>API: Agent output with sources
    API-->>U: SSE stream with answer + sources
```

### How It Works

1. **Frontend** — Each document in the Files panel has a checkbox. Selecting documents activates a blue scope bar: *"Scoped to N document(s)"* with a clear button.
2. **Query Payload** — `selected_doc_ids` (array of UUID strings) is sent alongside the query in both streaming and non-streaming endpoints.
3. **Data Pipeline** — The IDs flow through: `QueryRequest` → `streaming.py` context → `orchestrator` metadata → `rag_agent` → `two_level_search()`.
4. **Stage-1 Skip** — When `selected_doc_ids` is provided, the expensive Stage 1 document discovery is **entirely skipped**. The system goes directly to Stage 2 hybrid search, filtered to only the user-selected documents.
5. **Hybrid Search** — Stage 2 runs both dense vector similarity and BM25 keyword matching within the selected documents, then fuses results with Reciprocal Rank Fusion (RRF).

### Architecture — Stage-1 Skip Optimization

```mermaid
flowchart TD
    A[User Query + selected_doc_ids] --> B{selected_doc_ids<br/>provided?}

    B -- Yes --> C[Skip Stage 1]
    C --> D[Stage 2: Hybrid Search<br/>dense + BM25]
    D --> E[Filter: doc_id IN selected_doc_ids]
    E --> F[RRF Fusion → Top-K Chunks]

    B -- No --> G[Stage 1: Document Discovery<br/>Find relevant doc_ids]
    G --> H[Stage 2: Hybrid Search<br/>dense + BM25]
    H --> I[Filter: discovered doc_ids]
    I --> F

    F --> J[Return chunks + sources]

    style C fill:#2ecc71,color:#fff
    style G fill:#3498db,color:#fff
```

Skipping Stage 1 when the user has already selected documents provides **faster responses** and **more predictable results** — the user knows exactly which documents are being searched.

### Data Flow Through the Pipeline

| Layer | File | What happens |
|-------|------|-------------|
| Schema | `utils/schemas.py` | `QueryRequest.selected_doc_ids: List[str]` field |
| API | `api/streaming.py` | Passes `selected_doc_ids` into context dict |
| Orchestrator | `core/orchestrator.py` | Forwards `selected_doc_ids` in agent metadata |
| RAG Agent | `agents/rag_agent/agent.py` | Extracts from metadata, passes to search tool |
| Search Tool | `tools/rag_tools.py` | Checks `selected_doc_ids` → skips Stage 1 if present |
| Frontend | `frontend/index.html` | Checkboxes, scope bar, `_selectedDocIds` state |

---

## Message Edit & Rerun

Users can edit any previous message in a conversation and re-run the query from that point. Everything after the edited message is cleanly truncated — messages, agent executions, HITL requests, and stale conversation summaries.

### How It Works

```mermaid
sequenceDiagram
    participant U as User (Frontend)
    participant API as FastAPI
    participant DB as PostgreSQL
    participant SSE as Streaming Pipeline

    U->>U: Click pencil icon on a user message
    U->>U: Edit text in inline textarea
    U->>API: PUT /conversations/{conv}/messages/{msg}/edit<br/>{ content: "edited text" }

    API->>DB: Verify conversation ownership
    API->>DB: Verify message is a user message
    API->>DB: Count turn number (user messages ≤ this one)
    API->>DB: Update message content
    API->>DB: DELETE messages WHERE created_at > edit_timestamp
    API->>DB: DELETE agent_executions WHERE started_at > edit_timestamp
    API->>DB: DELETE hitl_requests WHERE created_at > edit_timestamp
    API->>DB: Walk summaries → delete any covering turns ≥ edit turn
    API-->>U: { deleted_messages, deleted_summaries }

    U->>U: Remove all DOM messages after edited one
    U->>SSE: POST /query/stream { query: "edited text" }
    SSE-->>U: New streamed response
```

### Truncation Logic

```mermaid
flowchart TD
    A[User edits message at Turn N] --> B[Update message content]
    B --> C[Delete all messages after Turn N]
    C --> D[Delete agent executions after Turn N]
    D --> E[Delete HITL requests after Turn N]
    E --> F[Walk conversation summaries in order]

    F --> G{Summary range<br/>reaches Turn N?}
    G -- No --> H[Keep summary]
    G -- Yes --> I[Delete summary]

    H --> J{More summaries?}
    I --> J
    J -- Yes --> F
    J -- No --> K[Flush to DB]
    K --> L[Frontend re-runs edited query via streaming]

    style B fill:#2ecc71,color:#fff
    style C fill:#e74c3c,color:#fff
    style D fill:#e74c3c,color:#fff
    style E fill:#e74c3c,color:#fff
    style I fill:#e74c3c,color:#fff
    style H fill:#2ecc71,color:#fff
```

### Summary Trimming — Precise Approach

Summaries are not blindly deleted. Each summary covers a fixed number of turns (default: 5). The system walks through summaries in chronological order, accumulating the turn range:

| Summary | Turns Covered | Cumulative Range | Edit at Turn 7? |
|---------|--------------|-----------------|-----------------|
| Summary 1 | 5 | Turns 1–5 | **Keep** (range < 7) |
| Summary 2 | 5 | Turns 6–10 | **Delete** (range reaches 7) |
| Summary 3 | 5 | Turns 11–15 | **Delete** (range past 7) |

Any summary whose cumulative range reaches or exceeds the edited turn number is deleted. Earlier summaries are preserved — they contain valid history that doesn't need regeneration.

### Frontend Behavior

- **Edit button** — pencil icon (✎) appears on hover, only on user messages loaded from history (not live-sent messages without a persisted ID)
- **Inline editing** — replaces the message bubble with a textarea + Cancel / Save & Rerun buttons
- **After save** — all messages below the edited one are removed from the DOM, then the edited query is re-sent through the normal streaming pipeline
- **Error handling** — if the backend call fails, the edit is cancelled and the original message is restored

---

## Response Text-to-Speech & Follow-up Suggestions

Two features that enhance the chat experience: a **Speak** button on every assistant response that reads it aloud via AWS Polly, and **follow-up suggestion chips** that appear after each response to keep the conversation flowing.

### Response TTS — How It Works

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as POST /tts/speak
    participant Polly as AWS Polly Neural

    U->>FE: Clicks "Speak" on assistant message
    FE->>FE: Check cached audio blob on element

    alt Cache hit
        FE->>FE: Play cached audio (no API call)
    else Cache miss
        FE->>FE: Show "Loading..." state
        FE->>API: POST { text: rawTextForTTS }
        API->>API: Strip Sources block
        API->>API: Strip [N] citations
        API->>API: Strip markdown (bold, headers, links, code)
        API->>API: Collapse whitespace, truncate to 3000 chars
        API->>Polly: synthesize_speech(clean_text, neural, mp3)
        Polly-->>API: audio/mpeg bytes
        API-->>FE: audio/mpeg stream
        FE->>FE: Cache blob on msgEl._ttsAudioBlob
        FE->>FE: Play audio, show "Stop" button
    end

    U->>FE: Clicks "Stop"
    FE->>FE: Pause audio, reset to "Speak"
```

### Text Cleaning Pipeline

The `/tts/speak` endpoint strips non-speakable content before sending to Polly:

| Step | Regex | What It Removes |
|------|-------|-----------------|
| 1 | `\n\n?Sources:\n[\s\S]*$` | Trailing sources block (`Sources:\n[1] file.pdf\n...`) |
| 2 | `\[\d+\]` | Inline citation references (`[1]`, `[2]`) |
| 3 | `\*{1,3}(.+?)\*{1,3}` | Markdown bold/italic markers |
| 4 | `^#{1,6}\s*` | Markdown headers |
| 5 | `\[([^\]]+)\]\([^)]+\)` | Markdown links → keeps link text only |
| 6 | `` ```...``` `` and `` `...` `` | Code fences and inline code |
| 7 | `\n{3,}` | Excessive newlines → collapse to `\n\n` |

### Follow-up Suggestions — How It Works

```mermaid
sequenceDiagram
    participant LLM as Composer LLM
    participant BE as Streaming Pipeline
    participant FE as Frontend
    participant U as User

    BE->>LLM: Composer streaming completes
    BE->>LLM: Quick generate() call with Q&A context
    Note right of LLM: "Suggest 3 follow-up<br/>questions for this Q&A"
    LLM-->>BE: 3 questions (one per line)
    BE->>FE: SSE event: suggestions { questions: [...] }
    BE->>FE: SSE event: done { ... }
    FE->>FE: finalizeAssistant() renders message
    FE->>FE: showSuggestionChips() renders clickable pills
    U->>FE: Clicks a suggestion chip
    FE->>FE: Fills input, removes chips, calls sendQuery()
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Audio caching on DOM element** | `msgEl._ttsAudioBlob` stores the Polly response. Clicking Speak again replays from cache — zero API cost, instant playback. |
| **Server-side text cleaning** | Cleaning happens on the backend so the frontend sends raw text and the endpoint handles all stripping. Consistent behavior across any client. |
| **3000 char truncation** | AWS Polly Neural engine has a per-request character limit. Long responses are silently truncated to stay within bounds. |
| **Suggestions via SSE, not separate call** | Follow-ups are emitted as an SSE event (`suggestions`) in the same stream, before `done`. No extra HTTP call, no auth issues, no race conditions. |
| **Stored on element, rendered after finalize** | Suggestions arrive before `done`, stored as `msgEl._suggestions`. Rendered after `finalizeAssistant()` so the DOM order is deterministic: content → sources → speak → tokens → suggestion chips. |
| **Chips self-destruct on click** | Clicking a suggestion removes all chips and immediately sends the query. Keeps the UI clean. |

### Files Changed

| Layer | File | Change |
|-------|------|--------|
| API | `api/routes.py` | `POST /tts/speak` endpoint with text cleaning pipeline |
| Streaming | `api/streaming.py` | Follow-up suggestion generation + `suggestions` SSE event |
| Frontend | `frontend/index.html` | Speak button, audio caching, suggestion chips, CSS |

---

## Response Comparison — Side-by-Side Model Evaluation

Compare any assistant response against a different model. A **Compare** button on each message opens a model picker; the selected model re-runs the full pipeline and streams its answer next to the original. The user picks the better response — "Keep Original" dismisses the card, "Use This" replaces the original in-place.

### How It Works

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as /query/stream
    participant MA as Master Agent
    participant Agents as Specialized Agents
    participant C as Composer

    U->>FE: Click Compare on message
    FE->>FE: Show model picker dropdown
    U->>FE: Select different model
    FE->>FE: Create side-by-side card<br/>(left = original, right = empty)
    FE->>API: POST /query/stream<br/>{ compare: true, model: "new-model" }

    Note over API: compare=true skips:<br/>• title generation<br/>• memory extraction<br/>• DB saves (messages, agents)<br/>• follow-up suggestions<br/>• voice TTS task

    API->>MA: Plan with new model
    MA->>Agents: Execute agents (all use new model)
    Agents->>C: Stream composed answer
    C-->>FE: SSE tokens → right pane

    FE->>FE: Render markdown in right pane
    FE->>U: Enable "Keep Original" / "Use This"

    alt Keep Original
        U->>FE: Click "Keep Original"
        FE->>FE: Remove comparison card
    else Use This
        U->>FE: Click "Use This"
        FE->>FE: Replace original msg-content<br/>Clear TTS audio cache
    end
```

### Model Override Architecture

When the user selects a model from the chat dropdown (or picks one in the comparison picker), **all** pipeline components use that model — not just the Composer:

```mermaid
flowchart LR
    subgraph "User Selection"
        D[Model Dropdown]
    end

    subgraph "agent_factory.py"
        F["build_agent_instances(registry, model_override)"]
        L["_llm() helper"]
        F --> L
    end

    D -->|model_override| F

    subgraph "Pipeline Components"
        M[Master Agent]
        R[RAG Agent]
        W[Web Search Agent]
        CO[Code Agent]
        MA[Mail Agent]
        G[GitHub Agent]
        CM[Composer Agent]
    end

    L -->|override provider| M
    L -->|override provider| R
    L -->|override provider| W
    L -->|override provider| CO
    L -->|override provider| MA
    L -->|override provider| G
    D -->|model_override| CM

    style D fill:#4a90d9,color:#fff
    style F fill:#2d2d2d,color:#fff
    style L fill:#2d2d2d,color:#fff
```

> Agents do **not** pass `model=` in their `.generate()` calls. The model is baked into the `llm_provider` at construction time by `build_agent_instances()`, so a single `model_override` propagates to every component automatically.

### Compare Mode — What Gets Skipped

| Operation | Normal Query | Compare Mode | Why |
|-----------|:---:|:---:|-----|
| Master Agent plan | Yes | Yes | Need full pipeline |
| Agent execution | Yes | Yes | Need full pipeline |
| Composer streaming | Yes | Yes | Need the answer |
| Title generation | Yes | **No** | Conversation already has a title |
| Memory extraction | Yes | **No** | Don't pollute memory with comparison |
| Message DB save | Yes | **No** | Comparison is ephemeral |
| Agent execution DB save | Yes | **No** | Comparison is ephemeral |
| Follow-up suggestions | Yes | **No** | Only needed for committed responses |
| Voice TTS task | Yes | **No** | User can TTS after choosing |

### Frontend Interaction Details

| Element | Behavior |
|---------|----------|
| **Compare button** | Appears after TTS button on every assistant message |
| **Model picker** | Dropdown grouped by provider; disables current model & unavailable providers |
| **Side-by-side card** | CSS Grid (2 columns on desktop, stacked on mobile < 768px) |
| **Close button (X)** | Aborts in-flight stream, removes card |
| **Keep Original** | Removes comparison card, no changes |
| **Use This** | Replaces original `msg-content` innerHTML, clears `_ttsAudioBlob` cache, updates `_rawTextForTTS` |
| **Guard** | `sendQuery()` blocked while comparison is active; only one comparison at a time |
| **AbortController** | Separate `_compareAbort` from main `_streamAbort`; cleaned up on conversation switch, new chat, logout |

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Ephemeral (no DB save) | Comparisons are exploratory — saving would double storage and pollute analytics |
| Reuse `/query/stream` | Same endpoint, same pipeline — just a `compare: true` flag to skip side effects |
| Separate AbortController | Main stream and comparison stream are independent; aborting one shouldn't cancel the other |
| Clear TTS cache on "Use This" | Old audio no longer matches the new text; force re-synthesis on next play |
| Skip memory extraction | Comparison answers shouldn't create duplicate memory entries |

### Files Changed

| Layer | File | Change |
|-------|------|--------|
| Schema | `utils/schemas.py` | Added `compare: bool = False` to `QueryRequest` |
| Streaming | `api/streaming.py` | Compare skip guards around title, memory, DB saves, suggestions, voice |
| Factory | `core/agent_factory.py` | `model_override` param — all agents use user-selected model |
| Master | `core/master_agent.py` | Removed explicit `model=config.*` from `.generate()` |
| Composer | `core/composer_agent.py` | Removed explicit `model=config.*` from voice summary `.generate()` |
| RAG Agent | `agents/rag_agent/agent.py` | Removed explicit `model=config.*` from `.generate()` |
| Web Agent | `agents/web_search_agent/agent.py` | Removed explicit `model=config.*` from `.generate()` (2 calls) |
| Code Agent | `agents/code_agent/agent.py` | Removed explicit `model=config.*` from `.generate()` |
| Mail Agent | `agents/mail_agent/agent.py` | Removed explicit `model=config.*` from `.generate()` (3 calls) |
| GitHub Agent | `agents/github_agent/agent.py` | Removed explicit `model=config.*` from `.generate()` |
| Frontend | `frontend/index.html` | Compare button, model picker, side-by-side card, SSE handler, CSS |

---

## Per-Message Token Tracking

Every assistant message stores **total tokens consumed** and a **per-component breakdown** (JSONB) — using actual token counts from each provider's API response, not estimates.

### What Gets Tracked

| Component | Token Source | Provider Support |
|-----------|-------------|-----------------|
| **Master Agent** | `LLMResult.usage` from `plan()` | OpenAI, Anthropic, Gemini |
| **Each Agent** (RAG, Code, Mail, Web) | `AgentOutput.resource_usage["tokens_used"]` — accumulated across all LLM calls within the agent | OpenAI, Anthropic, Gemini |
| **Composer** | `LLMResult.usage` (sync) or `StreamResult.usage` (streaming) | OpenAI, Anthropic, Gemini |

### Database Schema

Two columns added to the `messages` table (on the **assistant** row only):

```sql
total_tokens   INTEGER DEFAULT 0        -- sum of all component tokens
token_details  JSONB   DEFAULT '{}'     -- per-component breakdown
```

Example `token_details` value:

```json
{
  "master_agent": {"prompt_tokens": 820, "completion_tokens": 145, "total_tokens": 965},
  "rag_agent_a3f1b2c4": {"total_tokens": 2340},
  "web_search_agent_e7d9c0a1": {"total_tokens": 1180},
  "composer": {"prompt_tokens": 1950, "completion_tokens": 680, "total_tokens": 2630}
}
```

### Architecture

```mermaid
flowchart LR
    MA["Master Agent<br/><i>result.usage</i>"] --> TD["token_details JSONB"]
    A1["Agent 1<br/><i>resource_usage.tokens_used</i>"] --> TD
    A2["Agent 2<br/><i>resource_usage.tokens_used</i>"] --> TD
    CO["Composer<br/><i>StreamResult.usage</i>"] --> TD
    TD --> SUM["total_tokens = Σ"]
    SUM --> DB["messages table<br/><i>assistant row</i>"]
    TD --> DB
    DB --> SSE["SSE done event<br/><i>token_details + tokens_used</i>"]
    SSE --> FE["Frontend<br/><i>token badge + popup</i>"]

    style MA fill:#2196F3,color:#fff
    style CO fill:#FF9800,color:#fff
    style DB fill:#4CAF50,color:#fff
    style FE fill:#4CAF50,color:#fff
```

### Streaming Token Capture

All three providers return real token usage from their streaming APIs:

| Provider | Mechanism |
|----------|-----------|
| **OpenAI** | `stream_options={"include_usage": True}` — final chunk contains `chunk.usage` |
| **Anthropic** | `stream.get_final_message()` after iteration — returns `response.usage` with `input_tokens`, `output_tokens` |
| **Google Gemini** | `response.resolve()` after iteration — `response.usage_metadata` with `prompt_token_count`, `candidates_token_count` |

### Frontend

- Each assistant message shows a **token badge** (e.g., "2,340 tokens") next to the sources button.
- Clicking the badge opens a **popup** with per-component breakdown (Master Agent, each agent by ID, Composer).
- Token data is persisted in the DB and restored when loading past conversations.

---

## Chat Artifacts — File Generation & Download

The **Code Agent** can generate files (PDFs, charts, CSVs, images) that appear as **inline preview cards** in the chat. Users can download files directly — no cloud storage required.

### How It Works

```mermaid
flowchart TD
    subgraph UserRequest["User Request"]
        Q["Create a PDF report on AI trends"]
    end

    subgraph Pipeline["Multi-Agent Pipeline"]
        MA["Master Agent<br/><i>Plans: web_search → code_agent</i>"]
        WS["Web Search Agent<br/><i>Gathers research data</i>"]
        CA["Code Agent<br/><i>Generates PDF with reportlab</i>"]
    end

    subgraph FileCapture["File Capture (code_tools.py)"]
        TD["TempDir created<br/><i>OUTPUT_DIR injected</i>"]
        EX["Code executes<br/><i>Writes files to OUTPUT_DIR</i>"]
        SC["Scan for files<br/><i>Read → Base64 encode</i>"]
        CL["TempDir auto-deleted<br/><i>Only base64 remains in memory</i>"]
    end

    subgraph SSE["SSE Events"]
        AP["artifact_preview event<br/><i>filename, type, base64_data</i>"]
    end

    subgraph Frontend["Frontend"]
        PC["Preview Card<br/><i>Icon + filename + size</i>"]
        DL["Download Button<br/><i>Blob from base64</i>"]
    end

    Q --> MA
    MA --> WS
    WS -->|"Research data"| CA
    CA --> TD
    TD --> EX
    EX --> SC
    SC --> CL
    CL --> AP
    AP --> PC
    PC --> DL
```

### Supported Artifact Types

| Type | Extensions | Preview | Libraries |
|------|-----------|---------|-----------|
| **PDF** | `.pdf` | "Download to view" | `reportlab` |
| **Chart** | `.png`, `.jpg` (with plot/chart/graph in name) | Inline image | `matplotlib` |
| **Image** | `.png`, `.jpg`, `.gif`, `.svg` | Inline image | — |
| **CSV** | `.csv` | Table preview (headers + 5 rows) | `csv`, `pandas` |
| **JSON** | `.json` | Formatted snippet | `json` |
| **Code** | `.py`, `.js`, `.sql`, etc. | Syntax snippet (20 lines) | — |

### Persona-Aware Content Generation

When a **persona** is selected, the Code Agent generates content that matches the persona's tone and style:

```mermaid
flowchart LR
    subgraph Context["Context Passed to Code Agent"]
        P["Persona<br/><i>name + description</i>"]
        D["Dependency Data<br/><i>Web search results, docs</i>"]
        U["User Profile<br/><i>detail_level, tone</i>"]
    end

    subgraph CodeAgent["Code Agent"]
        PR["Prompt includes:<br/>- Persona instructions<br/>- Upstream data<br/>- Detail level"]
        GEN["Generate code that<br/>writes persona-styled content"]
    end

    subgraph Output["Generated Artifact"]
        PDF["PDF with persona tone<br/><i>e.g., 'Dear friend...' vs<br/>'Executive Summary...'</i>"]
    end

    P --> PR
    D --> PR
    U --> PR
    PR --> GEN
    GEN --> PDF
```

**Example — Same Request, Different Personas:**

| Persona | PDF Content Style |
|---------|------------------|
| **Professor** | "This comprehensive analysis examines the methodological implications of AI adoption rates across sectors, as evidenced by recent empirical studies..." |
| **Friend** | "Hey! So I looked into this AI stuff for you, and honestly, the numbers are pretty wild. Here's what's going on..." |
| **Lover** | "My dear, I've put together this special report just for you. Let me walk you through these fascinating findings with care..." |
| **Einstein** | "Consider, if you will, a thought experiment: what happens when artificial minds begin to outnumber tasks requiring human cognition?..." |

### Data Flow from Upstream Agents

The Code Agent receives **dependency_outputs** containing data from agents that ran earlier in the pipeline:

```python
# In code_agent prompt, dependency_outputs might contain:
{
    "web_search_agent_39bfdb4f": {
        "search_query": "AI adoption rates 2026",
        "tavily_answer": "According to recent reports...",
        "search_results": [...],
        "sources": [...]
    }
}
```

The Code Agent then:
1. Parses the search results
2. Extracts key facts, quotes, and statistics
3. Generates a properly formatted PDF/chart using the data
4. Applies the persona's tone to all text content

### Frontend Artifact Card

```
┌─────────────────────────────────────────────┐
│ 📄 impact_report.pdf                        │
│    2.1 KB · PDF                             │
├─────────────────────────────────────────────┤
│ Preview not available for PDF               │
├─────────────────────────────────────────────┤
│                              [Download]     │
└─────────────────────────────────────────────┘
```

- **Download**: Creates a Blob from base64 data and triggers browser download
- **No Save button**: Artifacts are ephemeral — download if needed, gone on refresh
- **No cloud storage**: Files exist only as base64 in the SSE stream

### Key Files

| File | Purpose |
|------|---------|
| `tools/code_tools.py` | `execute_code()` — runs Python, captures files from OUTPUT_DIR |
| `agents/code_agent/prompts.py` | Prompt with persona instructions, dependency data formatting |
| `agents/code_agent/agent.py` | Wires persona from metadata, builds ArtifactPreview objects |
| `api/streaming.py` | Emits `artifact_preview` SSE events |
| `frontend/index.html` | `renderArtifactPreview()`, `downloadArtifactLocal()` |

---

## Long term Memory Extraction Pipeline

Every user query passes through a **parallel memory extraction layer** that runs alongside the Master Agent — zero added latency. The extractor uses an LLM to detect personal data (name, company, job title, tone preferences, etc.) and persists any new findings to PostgreSQL for future conversations.

```mermaid
flowchart TD
    Query["User query arrives<br/><i>POST /query or /query/stream</i>"]

    Query --> LoadExisting["Load existing memory<br/><i>SELECT from user_long_term_memory</i>"]

    LoadExisting --> Fork["asyncio.gather()"]

    Fork --> MasterPlan["Master Agent<br/><i>plan(query, memory)</i>"]
    Fork --> Extract["Memory Extractor LLM"]

    subgraph Extract["Memory Extractor"]
        direction TB
        Prompt["Lightweight prompt<br/><i>temp=0.1, max_tokens=512</i>"]
        Prompt --> Check{"Found personal<br/>data?"}
        Check -->|"found: false"| NoOp["No-op<br/><i>skip silently</i>"]
        Check -->|"found: true"| Parse[" Check if already there or not .Parse critical_facts<br/>+ preferences"]
        Parse --> Upsert["INSERT ... ON CONFLICT<br/>DO UPDATE<br/><i>user_long_term_memory</i>"]
    end

    MasterPlan --> Continue["Continue pipeline<br/><i>Orchestrate → Compose</i>"]
    NoOp -.-> Continue
    Upsert -.->|"available next request"| Continue

    style Query fill:#2196F3,color:#fff
    style Fork fill:#FF9800,color:#fff
    style Prompt fill:#FF9800,color:#fff
    style Upsert fill:#4CAF50,color:#fff
    style MasterPlan fill:#9C27B0,color:#fff
```

### What gets extracted

| Category | Fields | Example trigger |
|----------|--------|-----------------|
| **CriticalFacts** | `user_name`, `company_name`, `job_title`, `recent_projects` | *"I am Vinay working on RAG Project at Google"* |
| **UserPreferences** | `tone`, `detail_level`, `language` | *"Keep answers short and casual"* |

---

## Query Execution Patterns

The Master Agent produces an execution plan where agents can depend on each other. The Orchestrator uses **Kahn's topological sort** to group agents into parallel stages. Here are the three main patterns:

### 1. Parallel Execution (No Dependencies)

When agents are independent, they all run in **Stage 1** simultaneously.

> *Example: "Search my docs for Q3 revenue AND find the latest news about our competitor"*

```mermaid
graph LR
    Master["Master Agent<br/>plan query"] --> Orchestrator

    subgraph "Stage 1 (parallel)"
        RAG["RAG Agent<br/><i>search docs for Q3 revenue</i>"]
        Web["Web Agent<br/><i>search competitor news</i>"]
    end

    Orchestrator --> RAG
    Orchestrator --> Web
    RAG --> State["Shared State"]
    Web --> State
    State --> Composer["Composer<br/><i>combine both results</i>"]

    style RAG fill:#4CAF50,color:#fff
    style Web fill:#2196F3,color:#fff
```

**Dependency graph:**
```
RAG Agent  → (no depdends)  ─┐
                          ├─► Composer
Web Agent  → (no depdends)  ─┘
```

---

### 2. Sequential Execution (Linear Chain)

Each agent depends on the previous one's output, creating a strict pipeline.

> *Example: "Find sales data in my docs, write Python code to analyze the trend, then email the report to my manager"*

```mermaid
graph LR
    Master["Master Agent"] --> Orchestrator

    subgraph "Stage 1"
        RAG["RAG Agent<br/><i>find sales data</i>"]
    end

    subgraph "Stage 2"
        Code["Code Agent<br/><i>analyze trend<br/>using RAG output</i>"]
    end

    subgraph "Stage 3"
        Mail["Mail Agent<br/><i>email report<br/>using Code output</i>"]
    end

    Orchestrator --> RAG
    RAG -->|"sales data"| Code
    Code -->|"analysis + chart"| Mail
    Mail --> Composer["Composer"]

    style RAG fill:#4CAF50,color:#fff
    style Code fill:#FF9800,color:#fff
    style Mail fill:#9C27B0,color:#fff
```

**Dependency graph:**
```
RAG Agent ──► Code Agent ──► Mail Agent ──► Composer
 (stage 1)    (stage 2)      (stage 3)
```

---

### 3. Diamond Execution (Fan-out + Fan-in)

Multiple agents run in parallel, then a downstream agent combines their results.

> *Example: "Search my docs for the budget, also find public benchmarks online, then write code to compare them"*

```mermaid
graph TD
    Master["Master Agent"] --> Orchestrator

    subgraph "Stage 1 (parallel)"
        RAG["RAG Agent<br/><i>find internal budget</i>"]
        Web["Web Agent<br/><i>find public benchmarks</i>"]
    end

    subgraph "Stage 2 (waits for both)"
        Code["Code Agent<br/><i>compare budget vs benchmarks</i>"]
    end

    Orchestrator --> RAG
    Orchestrator --> Web
    RAG -->|"budget data"| Code
    Web -->|"benchmarks"| Code
    Code --> Composer["Composer"]

    style RAG fill:#4CAF50,color:#fff
    style Web fill:#2196F3,color:#fff
    style Code fill:#FF9800,color:#fff
```

**Dependency graph:**
```
RAG Agent  ──┐
             ├──► Code Agent ──► Composer
Web Agent  ──┘
 (stage 1)      (stage 2)
```

---

### How Kahn's Algorithm Creates Stages

```mermaid
flowchart TD
    Input["List of agents +<br/>depends_on edges"] --> Build["Build in-degree map<br/>+ adjacency graph"]
    Build --> Init["Queue all agents<br/>with in-degree = 0"]
    Init --> Loop{"Queue empty?"}
    Loop -->|No| Drain["Drain current queue<br/>→ one parallel Stage"]
    Drain --> Decrement["Decrement in-degree<br/>of dependents"]
    Decrement --> Enqueue["Enqueue newly<br/>zero-degree agents"]
    Enqueue --> Loop
    Loop -->|Yes| Check{"Any nodes<br/>remaining?"}
    Check -->|"Yes → cycle!"| Error["Circular dependency<br/>ValueError"]
    Check -->|No| Done["Return stages<br/>list of lists"]

    style Error fill:#f44336,color:#fff
    style Done fill:#4CAF50,color:#fff
```

---

## RAG Document Pipeline — How Documents Are Processed

Document processing is **fully asynchronous**. The upload endpoint saves a DB record with `storage_key` and `storage_bucket`, then dispatches a **Celery task** per file via **Redis**. Each Celery worker handles both the **S3 upload** and the heavy pipeline (parsing, chunking, embedding) — so the API response is instant regardless of how many files are uploaded. Multiple workers process files **in parallel**. Real-time status updates (`pending → processing → ready / failed`) are pushed to the frontend via **Redis Pub/Sub → SSE**.

### Upload & Storage Flow

```mermaid
flowchart TD
    Upload["User uploads file(s)<br/><i>PDF / DOCX / Excel / CSV / TXT / MD</i>"]

    Upload --> API["FastAPI upload endpoint<br/><i>POST /documents/upload</i>"]

    API --> DB["Save DB record per file<br/><i>status=pending, storage_key,<br/>storage_bucket, file_size_bytes,<br/>content_type</i>"]

    DB -->|"Celery .delay() per file<br/>(file bytes hex-encoded)"| Queue["Redis Task Queue<br/><i>broker — one task per file</i>"]

    API -->|"instant response"| Response["Return { documents: [{doc_id, filename, status}] }<br/><i>No S3 upload blocking the API</i>"]

    Queue --> Worker

    subgraph Worker["Celery Workerk<br/><i>autoretry ×3 · exponential backoff · soft 5 min / hard 6 min<br/>multiple files processed in parallel across workers</i>"]
        direction TB

        S3["Upload to S3<br/><i>Backblaze B2 / AWS S3 / MinIO</i><br/>key: uploads/{user_id}/{doc_id}/{filename}"]

        S3 --> Status1["DB → status = processing<br/>Redis Pub/Sub → SSE"]
        Status1 --> Parser["Parser<br/><i>Extract raw text</i>"]

        Parser -->|"PDF"| PyMuPDF["PyMuPDF<br/>page.get_text()<br/><i>+ &lt;!-- PAGE N --&gt; markers</i>"]
        Parser -->|"DOCX"| PythonDocx["python-docx<br/>paragraphs"]
        Parser -->|"Excel/CSV"| Pandas["pandas<br/>df.to_string()"]
        Parser -->|"TXT/MD"| RawText["Read bytes<br/>UTF-8 decode"]

        PyMuPDF --> RawContent["Raw text content<br/>+ metadata"]
        PythonDocx --> RawContent
        Pandas --> RawContent
        RawText --> RawContent

        RawContent --> Parallel

        subgraph Parallel["Parallel: description ‖ chunking"]
        
            direction LR
            Describe["LLM generates<br/>document description<br/><i>2-3 sentences covering<br/>type, topics, scope</i>"]
            Chunker["Structure-Aware Chunker<br/><i>regex structure analysis,<br/>section splitting,<br/>1024-token chunks,<br/>table chunks kept intact,<br/>UUID per chunk,<br/>parent-child relationships,<br/>page number extraction</i>"]
        end

        Parallel --> BatchEmbed["Batch Embedding<br/><i>text-embedding-3-small<br/>1536 dims, embed_batch()</i>"]

        BatchEmbed --> Store

        subgraph Store["Store in Qdrant"]
            direction TB
            Collection["Create / get collection<br/><i>user_{id}_documents</i>"]
            Collection --> DocPoint["Upsert document-level point<br/><i>UUID, description embedding,<br/>filename, doc_type, total_chunks</i>"]
            DocPoint --> ChunkPoints["Upsert chunk points<br/><i>UUID, chunk embedding,<br/>text, metadata (page, section),<br/>doc_id at payload level</i>"]
            ChunkPoints --> PayloadIdx["Create payload indexes<br/><i>doc_type, date, section_title</i>"]
        end

        Store --> Status2["DB → status = ready<br/>Redis Pub/Sub → SSE<br/><i>or status = failed on error</i>"]
    end

    style Upload fill:#2196F3,color:#fff
    style API fill:#2196F3,color:#fff
    style Response fill:#2196F3,color:#fff
    style S3 fill:#FF9800,color:#fff
    style DB fill:#FF9800,color:#fff
    style Queue fill:#e53935,color:#fff
    style Worker fill:#e53935,color:#fff
    style Describe fill:#FF9800,color:#fff
    style Chunker fill:#FF9800,color:#fff
    style BatchEmbed fill:#9C27B0,color:#fff
    style Status1 fill:#4CAF50,color:#fff
    style Status2 fill:#4CAF50,color:#fff
```

### Document Viewing — Pre-signed URLs

When a user queries documents and clicks a source in the **Sources popup**, the system generates a **time-limited pre-signed URL** so the user can view the original file directly in the browser — without exposing storage credentials.

```mermaid
flowchart LR
    Click["User clicks source<br/>in Sources popup"]
    Click --> FE["Frontend calls<br/><i>GET /documents/{doc_id}/view</i><br/>with JWT"]
    FE --> Auth["JWT verified<br/>→ user_id"]
    Auth --> Own["DB ownership check<br/><i>WHERE doc_id = ? AND user_id = ?</i>"]
    Own --> Gen["Generate pre-signed URL<br/><i>boto3 generate_presigned_url()</i><br/>TTL: 5 min (configurable)"]
    Gen --> Resp["Return {url, filename,<br/>content_type, expires_in}"]
    Resp --> Open["Browser opens<br/>document in new tab"]

    style Click fill:#2196F3,color:#fff
    style FE fill:#2196F3,color:#fff
    style Auth fill:#4CAF50,color:#fff
    style Own fill:#4CAF50,color:#fff
    style Gen fill:#FF9800,color:#fff
    style Resp fill:#9C27B0,color:#fff
    style Open fill:#2196F3,color:#fff
```

**Security layers:**

| Layer | Detail |
|-------|--------|
| **JWT authentication** | Every request requires a valid bearer token |
| **DB ownership check** | `get_document_for_user()` verifies the document belongs to the requesting user |
| **S3 path scoping** | Storage keys are namespaced: `uploads/{user_id}/{doc_id}/{filename}` |
| **Time-limited URL** | Pre-signed URLs expire after `s3_presign_expiry` seconds (default 300 = 5 min) |
| **Private bucket** | Bucket is not publicly accessible; only pre-signed URLs grant read access |

**Key implementation details:**

| Aspect | Detail |
|--------|--------|
| **Cloud storage** | S3-compatible (Backblaze B2, AWS S3, MinIO) via `boto3` with S3v4 signatures |
| **Storage module** | `storage/s3.py` — `upload_file()`, `generate_presigned_url()`, `build_storage_key()` |
| **DB columns** | `storage_key`, `storage_bucket`, `file_size_bytes`, `content_type` on `documents` table |
| **Task queue** | Celery 5.x with Redis broker (`celery_app.py`) |
| **Worker config** | `acks_late=True`, `prefetch_multiplier=1`, `concurrency=4` |
| **Retry policy** | Up to 3 retries with exponential backoff (10 s → 20 s → 40 s, capped at 120 s, jittered) |
| **Rate limiting** | `20/m` per worker to avoid embedding API throttling |
| **Time limits** | 5 min soft (raises `SoftTimeLimitExceeded`), 6 min hard kill |
| **Status updates** | Redis Pub/Sub channel `doc_status:{user_id}` → SSE `GET /documents/status/stream` |
| **DB sessions** | Fresh `create_async_engine` per task (each `asyncio.run()` creates a new event loop) |
| **Serialisation** | File bytes sent as hex string for JSON safety |
| **Page tracking** | PDF parser injects `<!-- PAGE N -->` markers; chunker extracts page numbers into metadata |
| **Source linking** | Chunk payloads carry `doc_id` → RAG agent includes `doc_id` in source objects → frontend links to view endpoint |

---

## RAG Query Pipeline — Two-Stage Retrieval

The RAG Agent uses a **two-stage retrieval** approach: first narrow down *which documents* are relevant (Stage 1), then search *within those documents* for the best chunks (Stage 2). This dramatically reduces noise and improves precision, especially when a user has many uploaded documents.

```mermaid
flowchart TD
    Query["User query arrives<br/><i>via Master → Orchestrator</i>"]

    Query --> Stage1

    subgraph Stage1["Stage 1 — Document Discovery"]
        direction TB
        EmbedQ1["Embed query<br/><i>text-embedding-3-small</i>"]
        EmbedQ1 --> DescSearch["Dense search over<br/><b>document_entry</b> points<br/><i>cosine similarity on<br/>description embeddings</i>"]
        DescSearch --> Filter["Filter by score threshold<br/><i>default ≥ 0.40</i><br/>top-K = 5"]
        Filter --> DocIDs["Matched doc_ids<br/><i>+ filenames, descriptions, scores</i>"]
    end

    DocIDs --> Decision{"≥ 2 docs<br/>above threshold?"}

    Decision -->|"Yes"| Stage2Scoped["Stage 2 — Scoped Search<br/><i>search only matched doc_ids</i>"]
    Decision -->|"No"| Stage2Fallback["Stage 2 — Fallback<br/><i>scoped + unscoped hybrid,<br/>deduplicate by chunk_id</i>"]

    Stage2Scoped --> Hybrid
    Stage2Fallback --> Hybrid

    subgraph Hybrid["Stage 2 — Hybrid Chunk Search (parallel)"]
        direction TB
        Dense["Dense Vector Search<br/><i>embed query → cosine similarity<br/>filter type=chunk + doc_ids</i>"]
        Sparse["BM25 Sparse Search<br/><i>keyword matching<br/>scoped to doc_ids</i>"]
        Dense --> RRF["Reciprocal Rank Fusion<br/><i>score = Σ 1/(k + rank)<br/>merge + deduplicate</i>"]
        Sparse --> RRF
    end

    RRF --> Top20["Top 20 candidate chunks"]

    Top20 --> Rerank

    subgraph Rerank["LLM Reranking (configurable)"]
        direction TB
        Check{"rag_use_llm_reranking<br/>= True?"}
        Check -->|"Yes"| RePrompt["Prompt LLM with query<br/>+ all 20 chunk previews"]
        RePrompt --> Score["LLM ranks by:<br/>• Direct relevance<br/>• Information quality<br/>• Context completeness"]
        Score --> Top8["Return top 8<br/>ranked_indices"]
        Check -->|"No"| PassThrough["Return top 8<br/>by RRF score<br/><i>(no LLM call)</i>"]
    end

    Top8 --> Sources["Extract unique sources<br/><i>filename, page, section, doc_id</i>"]
    PassThrough --> Sources
    Top8 --> Confidence["Calculate confidence<br/><i>avg chunk scores</i>"]
    PassThrough --> Confidence

    Sources --> Output["AgentOutput<br/><i>chunks, sources (with doc_id),<br/>query, confidence_score,<br/>matched_documents</i>"]
    Confidence --> Output

    Output -->|"passed to"| Composer["Composer Agent<br/><i>synthesise answer<br/>with [1], [2] citations</i>"]

    Composer --> FE["Frontend Sources popup<br/><i>clickable filenames →<br/>GET /documents/{doc_id}/view<br/>→ pre-signed URL → new tab</i>"]

    style Query fill:#2196F3,color:#fff
    style EmbedQ1 fill:#2196F3,color:#fff
    style DescSearch fill:#2196F3,color:#fff
    style Filter fill:#2196F3,color:#fff
    style DocIDs fill:#2196F3,color:#fff
    style Dense fill:#4CAF50,color:#fff
    style Sparse fill:#8BC34A,color:#fff
    style RRF fill:#FF9800,color:#fff
    style RePrompt fill:#FF9800,color:#fff
    style Composer fill:#9C27B0,color:#fff
    style Decision fill:#FF9800,color:#fff
    style FE fill:#2196F3,color:#fff
```

**How it works:**

| Stage | What happens | Why |
|-------|-------------|-----|
| **Stage 1 — Document Discovery** | The query embedding is compared against **document-level description embeddings** (`type=document_entry`). Top-K documents above the score threshold are selected. | Narrows search space from *all* chunks across *all* documents to only relevant documents. Descriptions are LLM-generated summaries rich in entities, topics, and keywords. |
| **Fallback check** | If fewer than 2 documents pass the threshold, the system also runs an **unscoped hybrid search** (all chunks) and merges/deduplicates results. | Prevents empty results when descriptions don't closely match the query wording. |
| **Stage 2 — Scoped Hybrid Search** | Dense vector search + BM25 keyword search run **in parallel** (`asyncio.gather`), both scoped to chunks belonging to matched `doc_ids`. Results are merged via Reciprocal Rank Fusion. | Scoping BM25 to matched doc_ids avoids scrolling the entire collection — critical at scale (e.g., 100K chunks → only ~500 scoped chunks). |
| **LLM Reranking** | Configurable via `rag_use_llm_reranking` (default: `True`). When enabled, an LLM scores each chunk for relevance, quality, and completeness. When disabled, top chunks by RRF score are returned directly. | Adds semantic understanding but costs an extra LLM call. Can be toggled off for cost/latency savings. |
| **Source extraction** | Unique sources are extracted from chunks with `filename`, `page` (from `<!-- PAGE N -->` markers), `section`, and `doc_id` (from Qdrant payload). The Composer includes `[1], [2]` inline citations. | Enables the frontend Sources popup where users click a filename to fetch a pre-signed URL (`GET /documents/{doc_id}/view`) and open the original document in a new tab. |

**Configuration** (in `config/settings.py` or `.env`):

| Setting | Default | Description |
|---------|---------|-------------|
| `rag_description_top_k` | `5` | Max documents to consider in Stage 1 |
| `rag_description_score_threshold` | `0.40` | Min cosine similarity for a document to be "matched" |
| `rag_use_llm_reranking` | `True` | Whether to use LLM-based reranking after RRF |

**Qdrant point types** stored per user collection (`user_{id}_documents`):

| Point type | Key payload fields | Purpose |
|------------|-------------------|---------|
| `document_entry` | `doc_id`, `filename`, `description`, `doc_type`, `uploaded_at`, `total_chunks` | Document-level vector for Stage 1 broad retrieval |
| `chunk` | `doc_id`, `text`, `metadata` (page, section_title, chunk_index, parent_id) | Chunk-level vector for Stage 2 fine-grained retrieval |

---

## Gmail Agent Flow

```mermaid
flowchart TD
    Task["Mail task from<br/>Orchestrator"]

    Task --> Planner["LLM Action Planner<br/><i>decides what to do</i>"]

    Planner -->|"search_inbox"| SearchInbox["search_messages()<br/><i>Gmail search syntax<br/>e.g. from:x subject:y</i>"]
    Planner -->|"search_sent"| SearchSent["search_sent_messages()"]
    Planner -->|"search_drafts"| SearchDrafts["search_drafts()"]
    Planner -->|"read"| ReadMsg["get_message_by_id()<br/><i>full message content</i>"]
    Planner -->|"send"| Compose

    subgraph Compose["Compose + Send"]
        direction TB
        LLMCompose["LLM composes<br/>professional email<br/><i>greeting, body, sign-off</i>"]
        LLMCompose --> SendGmail["send_email()<br/><i>Gmail API → sent</i>"]
    end

    Planner -->|"draft"| Draft["draft_email()<br/><i>saved to Gmail Drafts</i>"]
    Planner -->|"reply"| Reply["reply_to_message()<br/><i>thread-aware reply</i>"]

    SearchInbox --> Summarise["LLM summarises<br/>results for user"]
    SearchSent --> Summarise
    SearchDrafts --> Summarise
    ReadMsg --> Summarise
    Compose --> Summarise
    Draft --> Summarise
    Reply --> Summarise

    Summarise --> Output["AgentOutput"]

    style Task fill:#2196F3,color:#fff
    style Planner fill:#FF9800,color:#fff
    style LLMCompose fill:#FF9800,color:#fff
    style Summarise fill:#9C27B0,color:#fff
```

---

## Tavily Web Search Agent Flow

```mermaid
flowchart TD
    Task["Web task from<br/>Orchestrator"]

    Task --> Strategy["LLM Search Strategist<br/><i>choose query + search type</i>"]

    Strategy -->|"basic"| Basic["web_search()<br/><i>Tavily basic search</i>"]
    Strategy -->|"news"| News["web_search_news()<br/><i>topic=news, recent articles</i>"]
    Strategy -->|"deep"| Deep["web_search_deep()<br/><i>advanced depth +<br/>raw page content</i>"]

    Basic --> Results["Search Results<br/><i>title, url, snippet,<br/>score, Tavily answer</i>"]
    News --> Results
    Deep --> Results

    Results --> FollowUp{"Follow-up<br/>URLs needed?"}
    FollowUp -->|Yes| Fetch["fetch_url()<br/><i>Tavily Extract API<br/>→ clean parsed content</i>"]
    FollowUp -->|No| Synth

    Fetch --> Synth["LLM Synthesis<br/><i>combine search results<br/>+ fetched pages<br/>+ Tavily answer</i>"]

    Synth --> Output["AgentOutput<br/><i>answer with [1],[2] citations<br/>sources, confidence</i>"]

    style Task fill:#2196F3,color:#fff
    style Strategy fill:#FF9800,color:#fff
    style Fetch fill:#4CAF50,color:#fff
    style Synth fill:#9C27B0,color:#fff
```

---

## Database Storage Reference

### PostgreSQL — Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| **`users`** | Registered users | `user_id` (UUID PK), `email` (UNIQUE), `display_name`, `password_hash`, `created_at` |
| **`sessions`** | Login sessions (closed on logout) | `session_id` (UUID PK), `user_id` (FK), `started_at`, `ended_at`, `is_active` (BOOL) |
| **`conversations`** | Chat threads within a session | `conversation_id` (UUID PK), `session_id` (FK), `user_id` (FK), `title`, `created_at`, `updated_at` |
| **`messages`** | Individual messages | `message_id` (UUID PK), `conversation_id` (FK), `role` (`user`/`assistant`/`system`), `content`, `metadata` (JSONB), `created_at` |
| **`agent_executions`** | Audit log of every agent run | `execution_id` (UUID PK), `conversation_id` (FK), `agent_name`, `task_description`, `status` (`pending`/`running`/`success`/`failed`), `input_payload` (JSONB), `output_payload` (JSONB), `error_message`, `started_at`, `completed_at` |
| **`documents`** | Uploaded document metadata & processing status | `doc_id` (UUID PK), `user_id` (FK), `filename`, `doc_type`, `description`, `total_chunks`, `qdrant_collection`, `processing_status` (`pending`/`processing`/`ready`/`failed`), `error_message`, `uploaded_at` |
| **`conversation_summaries`** | Rolling summaries of every 3 conversation turns | `summary_id` (UUID PK), `conversation_id` (FK), `summary_text`, `turns_covered`, `created_at` |
| **`user_long_term_memory`** | Persistent user profile extracted by the Memory Extractor | `user_id` (UUID PK, FK), `critical_facts` (JSONB), `preferences` (JSONB), `updated_at` |
| **`hitl_requests`** | HITL approval requests for agents with dangerous tools | `request_id` (UUID PK), `conversation_id` (FK), `agent_id`, `agent_name`, `tool_names` (TEXT[]), `task_description`, `status` (`pending`/`approved`/`denied`/`timed_out`/`expired`), `user_instructions`, `created_at`, `responded_at`, `expires_at` |
| **`user_connections`** | OAuth connections per user per provider | `connection_id` (UUID PK), `user_id` (FK), `provider` (VARCHAR), `account_label`, `account_id`, `access_token`, `refresh_token`, `token_type`, `expires_at`, `scopes` (TEXT[]), `provider_meta` (JSONB), `status` (`active`/`expired`/`revoked`/`error`), `connected_at`, `last_refreshed`, `last_used_at`, `error_message` — UNIQUE(`user_id`, `provider`, `account_id`) |
| **`personas`** | User-defined AI personalities | `persona_id` (UUID PK), `user_id` (FK), `name`, `description`, `is_default` (BOOL), `created_at`, `updated_at` |
| **`artifacts`** | Code agent file artifacts (PDF, charts, CSV) | `artifact_id` (UUID PK), `conversation_id` (FK), `message_id` (FK, nullable), `agent_id`, `agent_name`, `filename`, `artifact_type`, `content_type`, `file_size_bytes` (BIGINT), `storage_key`, `storage_bucket`, `preview_data` (JSONB), `metadata` (JSONB), `created_at` |
| **`web_scrape_collections`** | User web scraping collections | `collection_id` (UUID PK), `user_id` (FK), `name`, `description`, `status`, `created_at`, `updated_at` |
| **`web_scrape_urls`** | URLs within a scraping collection | `url_id` (UUID PK), `collection_id` (FK), `url`, `status`, `title`, `chunk_count`, `error_message`, `created_at`, `scraped_at` |
| **`scheduled_jobs`** | Cron-based scheduled agent jobs | `job_id` (UUID PK), `user_id` (FK), `name`, `description`, `cron_expression`, `timezone`, `status` (`active`/`paused`/`deleted`), `notification_mode`, `notification_target`, `created_at`, `updated_at` |
| **`scheduled_job_steps`** | Steps within a scheduled job | `step_id` (UUID PK), `job_id` (FK), `step_order`, `agent_name`, `task`, `tools` (TEXT[]), `depends_on_steps` (INT[]), `created_at` |
| **`scheduled_job_runs`** | Execution history for scheduled jobs | `run_id` (UUID PK), `job_id` (FK), `status`, `started_at`, `completed_at`, `error_message` |
| **`scheduled_job_step_results`** | Per-step results within a run | `result_id` (UUID PK), `run_id` (FK), `step_id` (FK), `status`, `output_payload` (JSONB), `error_message`, `started_at`, `completed_at` |

#### `user_long_term_memory` — JSONB Column Detail

```
critical_facts                          preferences
┌────────────────────────────────┐      ┌──────────────────────────┐
│ user_name    : "vinu"    │      │ tone         : "casual"  │
│ company_name : "Google"     │      │ detail_level : "detailed"│
│ job_title    : "AI Engineer" │      │ language     : "en"      │
│ recent_projects: ["RAGA",     │      └──────────────────────────┘
│                   "Beta"]      │
└────────────────────────────────┘
```

### Qdrant — Vector Collections

Each user gets a dedicated collection: **`user_{user_id}_documents`**

| Point Type | Stored Payload Fields | Purpose |
|------------|----------------------|---------|
| **`document_entry`** | `doc_id`, `filename`, `description`, `doc_type`, `uploaded_at`, `total_chunks` | Document-level vector for Stage 1 broad retrieval |
| **`chunk`** | `doc_id`, `text`, `metadata` (`filename`, `page`, `section_title`, `doc_type`, `date`, …) | Individual text chunk for Stage 2 fine-grained search |

**Vector config:** `text-embedding-3-small` (1536 dimensions), cosine distance.

**Payload indexes:** `type` (keyword), `doc_id` (keyword), `metadata.doc_type` (keyword), `metadata.date` (keyword), `metadata.section_title` (text).

---

## Agent Input / Output Reference

Every agent receives an **`AgentInput`** and returns an **`AgentOutput`**. These are Pydantic models defined in `utils/schemas.py`.

### Common Input — `AgentInput`

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Unique instance ID (e.g. `rag_agent-a1b2c3`) |
| `agent_name` | `str` | Agent type (`rag_agent`, `code_agent`, …) |
| `task` | `str` | Natural-language task description from the Master Agent |
| `entities` | `Dict` | Extracted entities (`date_range`, `metric`, `names`, `doc_type`, …) |
| `tools` | `List[str]` | Tools this agent is allowed to use |
| `conversation_history` | `List[Dict]` | Recent conversation context |
| `long_term_memory` | `Dict` | `{critical: {...}, preferences: {...}}` — user profile |
| `dependency_outputs` | `Dict` | Outputs from upstream agents (`dep_agent_id → data`) |
| `retry_config` | `Dict` | `{max_retries, timeout, backoff_multiplier}` |
| `metadata` | `Dict` | `{user_id, query_id, session_id, conversation_id, persona, active_web_collection_ids, selected_doc_ids}` |

### Common Output — `AgentOutput`

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Same ID from input |
| `agent_name` | `str` | Agent type |
| `task_description` | `str` | What was asked |
| `task_done` | `bool` | `true` on success, `false` on failure/timeout |
| `data` | `Dict` | Full results (agent-specific, see below) |
| `partial_data` | `Dict` | Partial results on timeout/failure |
| `error` | `str` | Error message if failed |
| `confidence_score` | `float` | 0.0 – 1.0 |
| `depends_on` | `List[str]` | IDs of upstream agents this depended on |
| `resource_usage` | `Dict` | `{time_taken_ms, tokens_used, api_calls_made, …}` |
| `metadata` | `Dict` | `{sources, citations, model_used}` |
| `artifacts` | `List[ArtifactPreview]` | File artifacts generated by code_agent (PDF, charts, CSV) — empty `[]` for other agents |

### Per-Agent `data` Output

#### RAG Agent

```json
{
  "chunks": [
    {"chunk_id": "uuid", "score": 0.92, "text": "...", "metadata": {"filename": "report.pdf", "page": 3, "section_title": "Revenue"}}
  ],
  "sources": [
    {"type": "document", "source": "report.pdf", "page": 3, "section": "Revenue"}
  ],
  "query": "What was Q3 revenue?"
}
```

| Key | Description |
|-----|-------------|
| `chunks` | Top-8 reranked document chunks with text, score, and metadata |
| `sources` | Deduplicated list of source documents cited |
| `query` | The original search query |

#### Code Agent

```json
{
  "code": "import json\ndata = {...}\nprint(json.dumps(data, indent=2))",
  "stdout": "{\"total\": 42}",
  "stderr": "",
  "exit_code": 0
}
```

| Key | Description |
|-----|-------------|
| `code` | The generated Python code |
| `stdout` | Standard output from execution |
| `stderr` | Standard error (if any) |
| `exit_code` | `0` = success, non-zero = failure |

#### Mail Agent

```json
{
  "action": "search_inbox",
  "messages": [
    {"id": "msg-id", "from": "bob@example.com", "subject": "Q3 Report", "date": "2026-01-15", "snippet": "..."}
  ],
  "count": 5
}
```

| Action | Extra `data` keys | Description |
|--------|-------------------|-------------|
| `search_inbox` / `search_sent` | `messages`, `count` | List of matching messages |
| `search_drafts` | `drafts`, `count` | List of matching drafts |
| `read` | `message` | Full message content |
| `send` | `sent` | Confirmation + message ID |
| `draft` | `draft` | Created draft details |
| `reply` | `reply` | Reply confirmation |

Also returns a `result` field with an LLM-generated human-readable summary.

#### Web Search Agent

```json
{
  "search_query": "Python async best practices 2026",
  "search_type": "basic",
  "tavily_answer": "Short AI-generated answer from Tavily...",
  "search_results": [
    {"title": "...", "url": "https://...", "snippet": "...", "score": 0.95}
  ],
  "fetched_pages": 2,
  "sources": [
    {"type": "web", "source": "Article Title", "url": "https://...", "excerpt": "...", "relevance_score": 0.95}
  ]
}
```

| Key | Description |
|-----|-------------|
| `search_query` | The optimised query sent to Tavily |
| `search_type` | `basic`, `news`, or `deep` |
| `tavily_answer` | Tavily's AI-generated direct answer (if available) |
| `search_results` | Raw results with title, URL, snippet, score |
| `fetched_pages` | Number of pages deep-fetched for richer content |
| `sources` | Formatted source list for citations |

Also returns a `result` field with the LLM-synthesised final answer.

### LLM Provider — `LLMResult`

All `generate()` calls return an `LLMResult` dataclass (defined in `utils/llm_providers.py`) instead of raw `str` / `dict`:

```python
@dataclass
class LLMResult:
    text:  str                    # Raw text response from the LLM
    data:  Dict[str, Any] | None  # Parsed JSON dict when output_schema was provided, else None
    usage: Dict[str, int]         # {"prompt_tokens": …, "completion_tokens": …, "total_tokens": …}
    model: str                    # The model that actually served the request
```

**Usage patterns:**

```python
# ── Text response (no output_schema) ──────────────────────────────
result = await self.llm.generate(prompt=prompt, temperature=0.7)
answer = result.text                      # plain string
tokens = result.usage["total_tokens"]     # real API token count

# ── Structured response (with output_schema) ──────────────────────
result = await self.llm.generate(
    prompt=prompt,
    output_schema={"action": "string", "params": "dict"},
)
plan = result.data                        # parsed dict
tokens = result.usage["total_tokens"]
```

> **Do not pass `model=` explicitly** in agent `.generate()` calls. The model is set at construction time via the `llm_provider` injected by `build_agent_instances()`. This allows the user's model dropdown selection to override all agents at once. Only pass `temperature` and `output_schema`.

### Master Agent → Composer Pipeline

```
User Query
    │
    ▼
┌────────────┐   ResolvedMasterOutput    ┌──────────────┐
│ Master     │ ────────────────────────▶ │ Orchestrator │
│ Agent      │   • analysis.intent       │              │
│            │   • analysis.entities     │  runs agents │
│            │   • execution_plan        │  in stages   │
└────────────┘     (agent tasks +        └──────┬───────┘
                    dependencies)                │
                                                 ▼
                                     Dict[agent_id → AgentOutput]
                                                 │
                                                 ▼
                                         ┌──────────────┐   ComposerOutput
                                         │  Composer    │ ──────────────▶ User
                                         │  Agent       │   • answer (str)
                                         │              │   • sources
                                         └──────────────┘
```

---

## Auth & Connectors Architecture

The system uses a modular architecture for authentication and third-party OAuth integrations.

### Project Structure

```
MRAG/
├── auth/                            # User authentication module
│   ├── __init__.py
│   ├── routes.py                    # POST /register, /login
│   ├── jwt.py                       # JWT create/verify (HMAC-SHA256)
│   ├── models.py                    # User ORM (re-exports from database/models.py)
│   ├── dependencies.py              # get_current_user_id() FastAPI dependency
│   └── password.py                  # bcrypt hash/verify
│
├── connectors/                      # OAuth connector module
│   ├── __init__.py
│   ├── routes.py                    # /providers, /auth-url, /callback, /connections
│   ├── base.py                      # BaseConnector abstract class
│   ├── gmail.py                     # GmailConnector(BaseConnector)
│   ├── models.py                    # UserConnection ORM (re-exports from database/models.py)
│   ├── registry.py                  # ConnectorRegistry (auto-discover at startup)
│   ├── token_manager.py             # get/store/refresh/revoke per-user tokens
│   └── encryption.py                # Fernet AES encryption for tokens at rest
│
├── api/                             # AI routes (unchanged)
│   ├── routes.py                    # /query, /hitl/respond
│   ├── streaming.py                 # /query/stream (SSE)
│   └── middleware.py
│
├── agents/                          # Agent implementations
├── core/                            # Orchestrator, Master, Composer
├── tools/                           # Tool functions (mail_tools uses connectors)
└── ...
```

### Security Configuration

All secrets are loaded from environment variables via `config/settings.py`:

| Env Var | Purpose | Generate with |
|---------|---------|---------------|
| `JWT_SECRET` | HMAC-SHA256 signing key for auth tokens | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `JWT_EXPIRY_SECONDS` | Token TTL (default: 604800 = 7 days) | — |
| `OAUTH_STATE_SECRET` | HMAC key for OAuth CSRF state tokens | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `TOKEN_ENCRYPTION_KEY` | Fernet key for encrypting OAuth tokens at rest | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |

#### Password Hashing

Passwords are hashed with **bcrypt** (work factor 12, auto-salted):

```python
# auth/password.py
from auth.password import hash_password, verify_password

hashed = hash_password("user-password")       # $2b$12$...
ok     = verify_password("user-password", hashed)  # True
```

#### Token Encryption at Rest

OAuth access tokens and refresh tokens are encrypted with **Fernet (AES-128-CBC + HMAC-SHA256)** before being stored in PostgreSQL. This is handled transparently by `connectors/encryption.py`:

```
Store flow:   plaintext_token → encrypt_token() → ciphertext → DB column
Read flow:    DB column → ciphertext → decrypt_token() → plaintext_token
```

- If `TOKEN_ENCRYPTION_KEY` is not set, tokens are stored as plaintext (with a warning)
- Tokens stored before encryption was enabled are handled gracefully (auto-detected and returned as-is)
- All encryption/decryption is done inside `token_manager.py` — no other code needs to know about it

---

## Security Guardrails

The system implements multi-layer defense against prompt injection, code execution attacks, and API abuse. The `security/` module provides reusable utilities for all agents.

### Architecture Overview

```mermaid
flowchart TB
    subgraph Input["User Input Layers"]
        API[API Request] --> RL[Rate Limiter<br/>Redis-backed]
        RL --> IV[Input Validator<br/>Blocklist patterns]
        IV --> SAN[Sanitization<br/>Escape + detect]
    end

    subgraph Prompts["Prompt Construction"]
        SAN --> DEL[Delimiters<br/>XML-style wrappers]
        DEL --> PROMPT[Agent Prompt<br/>+ DELIMITER_SYSTEM_PROMPT]
    end

    subgraph Execution["Code Execution"]
        CODE[LLM-generated code] --> CV[Code Validator<br/>AST + regex]
        CV -->|Safe| EXEC[Subprocess]
        CV -->|Blocked| REJECT[Reject + log]
    end

    subgraph Response["Response Layer"]
        PROMPT --> LLM[LLM Call]
        LLM --> HEADERS[Security Headers<br/>CSP, X-Frame-Options]
        HEADERS --> ERR[Error Sanitization<br/>No stack traces]
    end
```

### Security Module Files

| File | Purpose |
|------|---------|
| `security/sanitization.py` | Input sanitization, injection pattern detection |
| `security/delimiters.py` | XML-style delimiters to mark user content |
| `security/validators.py` | Query blocklist, input validation |
| `security/code_validator.py` | AST-based code safety checking |
| `security/rate_limiter.py` | Redis-backed per-user rate limiting |
| `security/middleware.py` | Security headers, error sanitization |
| `security/config.py` | CORS config, secret validation |

### Prompt Injection Defense

All agent prompts must sanitize user inputs and wrap them with delimiters:

```python
from security.sanitization import sanitize_user_input
from security.delimiters import (
    wrap_user_query,
    wrap_task,
    wrap_entities,
    wrap_conversation_history,
    DELIMITER_SYSTEM_PROMPT,
)

# 1. Sanitize user input — escapes control chars, detects injection patterns
safe_task = sanitize_user_input(task).text
safe_entity_str = ", ".join(
    f"{sanitize_user_input(str(k)).text}={sanitize_user_input(str(v)).text}"
    for k, v in entities.items()
) if entities else "none"

# 2. Wrap with delimiters — marks content as DATA, not instructions
# 3. Prepend DELIMITER_SYSTEM_PROMPT — tells LLM to treat delimited content as data only

prompt = f"""{DELIMITER_SYSTEM_PROMPT}

You are an Agent in a multi-agent RAG system.

{wrap_task(safe_task)}

{wrap_entities(safe_entity_str)}
... rest of prompt ...
"""
```

#### Sanitization Functions

| Function | Use Case | Max Length |
|----------|----------|-----------|
| `sanitize_user_input(text)` | User queries, task descriptions, entities | 50,000 |
| `sanitize_document_chunk(text, source)` | RAG-retrieved document content | 8,000 |
| `sanitize_web_content(text, url)` | Web-scraped content (highest risk) | 10,000 |

Each returns a `SanitizationResult` with:
- `text` — sanitized output
- `was_modified` — boolean
- `detected_patterns` — list of matched injection pattern names
- `risk_score` — 0.0 to 1.0

#### Injection Patterns Detected

- `ignore previous instructions`, `disregard all`, `you are now`
- Role injection: `system:`, `assistant:`, `human:`
- Model tokens: `<|...|>`, `[INST]`, `<<SYS>>`
- Jailbreak attempts: `DAN`, `developer mode`

#### Delimiter Wrappers

| Wrapper | Output Tags |
|---------|-------------|
| `wrap_user_query(query)` | `<\|user_query\|>...<\/\|user_query\|>` |
| `wrap_task(task)` | `<\|task_description\|>...<\/\|task_description\|>` |
| `wrap_entities(str)` | `<\|entities\|>...<\/\|entities\|>` |
| `wrap_conversation_history(str)` | `<\|conversation_history\|>...<\/\|conversation_history\|>` |
| `wrap_document_chunk(text, source, idx)` | `<\|document_chunk\|>...<\/\|document_chunk\|>` |
| `wrap_web_content(text, url)` | `<\|web_content\|>...<\/\|web_content\|>` |

### Code Execution Safety

The `code_agent` uses HITL approval + pre-execution validation:

```python
# tools/code_tools.py
from security.code_validator import validate_and_log

@tool("code_agent", requires_approval=True)
async def execute_code(code: str, language: str = "python", timeout: int = 30):
    # Security: validate BEFORE subprocess
    validation = validate_and_log(code, "code_agent")
    if not validation.is_safe:
        return {"error": "Code validation failed", "violations": validation.violations}

    # ... proceed with subprocess execution ...
```

#### Blocked Imports

`os`, `sys`, `subprocess`, `shutil`, `socket`, `requests`, `urllib`, `pickle`, `ctypes`, `multiprocessing`, `importlib`, `builtins`, `code`, `pty`, `signal`, `gc`

#### Allowed Imports

`json`, `math`, `datetime`, `collections`, `re`, `csv`, `io`, `pandas`, `matplotlib`, `numpy`, `reportlab`, `statistics`

#### Dangerous Patterns Blocked

- `os.system(`, `subprocess.`, `Popen(`
- `eval(`, `exec(`, `compile(`
- `__builtins__`, `__subclasses__`, `__globals__`
- Path traversal: `../`, `.env`, `credentials`, `id_rsa`

### Rate Limiting

Redis-backed sliding window rate limiter:

| Category | Limit | Window |
|----------|-------|--------|
| `standard` | 100 requests | 60 seconds |
| `streaming` | 20 requests | 60 seconds |
| `code_execution` | 10 requests | 60 seconds |
| `auth` | 10 requests | 60 seconds |

```python
# Usage in endpoints
from security.rate_limiter import check_rate_limit

@router.post("/query/stream")
async def stream_query(
    request: QueryRequest,
    raw_request: Request,
    ...
):
    await check_rate_limit(raw_request, "streaming")  # Raises 429 if exceeded
    ...
```

### Security Headers

All responses include:

| Header | Value |
|--------|-------|
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `X-XSS-Protection` | `1; mode=block` |
| `Content-Security-Policy` | `default-src 'self'; ...` |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |
| `Permissions-Policy` | `geolocation=(), microphone=(), camera=()` |

### Error Sanitization

Unhandled exceptions are caught and sanitized:
- Full stack trace logged internally with `request_id`
- Client receives: `{"error": "internal_error", "request_id": "abc123"}`
- No file paths, line numbers, or internal details exposed

### Startup Secret Validation

At startup, the system warns about insecure defaults:

```python
# main.py lifespan
from security.config import validate_secrets
for warning in validate_secrets():
    logger.warning("SECURITY: %s", warning)
```

Warnings issued for:
- Default `JWT_SECRET` (`"change-me-jwt-secret-key"`)
- `JWT_SECRET` shorter than 32 characters
- Default `OAUTH_STATE_SECRET`
- Default `TOKEN_ENCRYPTION_KEY`

### OAuth Connector Flow

```mermaid
sequenceDiagram
    participant User as User (Frontend)
    participant FE as Sidebar UI
    participant API as /api/v1/connectors
    participant Google as Google OAuth
    participant DB as PostgreSQL

    User->>FE: Click "Connect Gmail"
    FE->>API: GET /gmail/auth-url
    API-->>FE: {auth_url: "https://accounts.google.com/..."}
    FE->>Google: Open popup → auth_url
    Google-->>User: Consent screen
    User->>Google: Approve
    Google->>API: GET /gmail/callback?code=XXX&state=YYY
    API->>API: Verify state (HMAC + TTL)
    API->>Google: POST /token (exchange code)
    Google-->>API: {access_token, refresh_token, expires_in}
    API->>API: encrypt_token(access_token)
    API->>DB: INSERT/UPDATE user_connections
    API-->>Google: Return auto-closing HTML
    Note over FE: postMessage('oauth-callback')
    FE->>API: GET /connections (refresh list)
    API-->>FE: [{provider: gmail, status: active, ...}]
```

### How Tools Use Connector Tokens (Production Pattern)

When any connector-backed agent runs, it retrieves a fresh per-user token on every request. Here's the production flow using the Gmail implementation as reference:

```
MailAgent.execute(task_config)
  → prepare_gmail_service(user_id)               # called once per request
    → get_active_token(user_id, "gmail")          # DB lookup + auto-refresh
      → SELECT from user_connections WHERE user_id + provider + status='active'
      → decrypt_token(access_token)
      → if expired → connector.refresh_access_token(decrypt(refresh_token))
                   → encrypt_token(new_access_token) → UPDATE DB
      → return plaintext access_token
    → asyncio.to_thread(build, "gmail", "v1", credentials=...)  # non-blocking
    → get_user_connections(user_id)               # resolve sender email
    → _current_gmail_service.set(service)         # ContextVar — request-scoped
    → _current_sender_email.set(email)            # ContextVar — request-scoped
  → tool functions call _get_service()            # reads ContextVar
    → await asyncio.to_thread(api_call.execute)   # non-blocking sync I/O
  → request ends → ContextVars garbage-collected  # no memory leak
```

**Key design principles:**

| Principle | Implementation |
|-----------|---------------|
| Fresh token every request | `get_active_token()` runs on every `execute()` call — never cached globally |
| Auto-refresh | `token_manager` checks `expires_at` with 120s buffer, refreshes transparently |
| Non-blocking I/O | All sync SDK calls wrapped in `asyncio.to_thread()` |
| Request-scoped state | `ContextVar` — automatically GC'd when the async task ends |
| No global singletons | No `_user_services` dict, no `_gmail_service` global |
| Graceful errors | If user has no connection → `RuntimeError` with clear "connect via UI" message |

This pattern applies to **all** connector-backed agents. Each provider follows the same structure — only the SDK and API calls change.

---

## How to Add a New Connector

Adding a new OAuth connector with tools and HITL is a **7-step process**. This guide shows the complete walkthrough with **two real examples**: **Slack** (REST API + `httpx`) and **GitHub** (REST API + `httpx`) — demonstrating both read-only and HITL-protected tools.

### Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Step 1 │ connectors/<provider>.py    — implement the OAuth connector   │
│  Step 2 │ connectors/registry.py      — register the connector          │
│  Step 3 │ config/settings.py + .env   — add client_id / secret          │
│  Step 4 │ tools/<provider>_tools.py   — tool functions + HITL flags     │
│  Step 5 │ agents/<provider>_agent/    — agent.py + prompts.py           │
│  Step 6 │ config/agent_registry.yaml  — register agent capabilities     │
│  Step 7 │ core/agent_factory.py       — wire agent instance             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1 — Create the Connector

Subclass `BaseConnector` and implement the OAuth methods.

#### Slack Example — `connectors/slack.py`

```python
"""SlackConnector — OAuth2 for Slack workspace access."""

from __future__ import annotations
import logging
from typing import Any, Dict, List
from urllib.parse import urlencode

import httpx
from config.settings import config
from connectors.base import BaseConnector

logger = logging.getLogger(__name__)

_SLACK_AUTH_URL = "https://slack.com/oauth/v2/authorize"
_SLACK_TOKEN_URL = "https://slack.com/api/oauth.v2.access"
_SLACK_REVOKE_URL = "https://slack.com/api/auth.revoke"


class SlackConnector(BaseConnector):

    @property
    def provider_name(self) -> str:
        return "slack"

    @property
    def display_name(self) -> str:
        return "Slack"

    @property
    def scopes(self) -> List[str]:
        return ["channels:read", "channels:history", "chat:write", "users:read"]

    @property
    def icon(self) -> str:
        return "💬"

    def is_configured(self) -> bool:
        return bool(config.slack_client_id and config.slack_client_secret)

    def _redirect_uri(self) -> str:
        return f"{config.oauth_redirect_base}/api/v1/connectors/slack/callback"

    def get_auth_url(self, state: str) -> str:
        params = {
            "client_id": config.slack_client_id,
            "redirect_uri": self._redirect_uri(),
            "scope": ",".join(self.scopes),
            "state": state,
        }
        return f"{_SLACK_AUTH_URL}?{urlencode(params)}"

    async def handle_callback(self, code: str) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(_SLACK_TOKEN_URL, data={
                "code": code,
                "client_id": config.slack_client_id,
                "client_secret": config.slack_client_secret,
                "redirect_uri": self._redirect_uri(),
            })
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                raise ValueError(f"Slack OAuth error: {data.get('error')}")

        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token"),
            "expires_in": data.get("expires_in", 43200),
            "scopes": data.get("scope", "").split(","),
            "account_id": data.get("team", {}).get("id", ""),
            "account_label": data.get("team", {}).get("name", "Slack Workspace"),
            "provider_meta": {
                "team_id": data.get("team", {}).get("id"),
                "team_name": data.get("team", {}).get("name"),
                "bot_user_id": data.get("bot_user_id"),
            },
        }

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        # Slack V2 tokens with token rotation — implement per Slack docs
        raise NotImplementedError("Slack token refresh not yet supported")

    async def revoke_token(self, access_token: str) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    _SLACK_REVOKE_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                return resp.json().get("ok", False)
        except Exception:
            logger.warning("Slack token revocation failed", exc_info=True)
            return False
```

#### GitHub Example — `connectors/github.py`

```python
"""GitHubConnector — OAuth2 for GitHub API access."""

from __future__ import annotations
import logging
from typing import Any, Dict, List
from urllib.parse import urlencode

import httpx
from config.settings import config
from connectors.base import BaseConnector

logger = logging.getLogger(__name__)

_GH_AUTH_URL = "https://github.com/login/oauth/authorize"
_GH_TOKEN_URL = "https://github.com/login/oauth/access_token"
_GH_API = "https://api.github.com"


class GitHubConnector(BaseConnector):

    @property
    def provider_name(self) -> str:
        return "github"

    @property
    def display_name(self) -> str:
        return "GitHub"

    @property
    def scopes(self) -> List[str]:
        return ["repo", "read:user", "user:email"]

    @property
    def icon(self) -> str:
        return "🐙"

    def is_configured(self) -> bool:
        return bool(config.github_client_id and config.github_client_secret)

    def _redirect_uri(self) -> str:
        return f"{config.oauth_redirect_base}/api/v1/connectors/github/callback"

    def get_auth_url(self, state: str) -> str:
        params = {
            "client_id": config.github_client_id,
            "redirect_uri": self._redirect_uri(),
            "scope": " ".join(self.scopes),
            "state": state,
        }
        return f"{_GH_AUTH_URL}?{urlencode(params)}"

    async def handle_callback(self, code: str) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            # Exchange code for token
            resp = await client.post(
                _GH_TOKEN_URL,
                data={
                    "client_id": config.github_client_id,
                    "client_secret": config.github_client_secret,
                    "code": code,
                    "redirect_uri": self._redirect_uri(),
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise ValueError(f"GitHub OAuth error: {data.get('error_description', data['error'])}")

            # Fetch user profile
            user_resp = await client.get(
                f"{_GH_API}/user",
                headers={
                    "Authorization": f"Bearer {data['access_token']}",
                    "Accept": "application/vnd.github+json",
                },
            )
            user_resp.raise_for_status()
            user = user_resp.json()

        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token"),
            "expires_in": data.get("expires_in", 28800),  # 8h for GitHub Apps
            "scopes": data.get("scope", "").split(","),
            "account_id": str(user.get("id", "")),
            "account_label": user.get("login", ""),
            "provider_meta": {
                "login": user.get("login"),
                "name": user.get("name"),
                "avatar_url": user.get("avatar_url"),
                "email": user.get("email"),
            },
        }

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """GitHub App refresh — non-app tokens don't expire."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _GH_TOKEN_URL,
                data={
                    "client_id": config.github_client_id,
                    "client_secret": config.github_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise ValueError(f"GitHub token refresh error: {data.get('error_description', data['error'])}")

        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token"),
            "expires_in": data.get("expires_in", 28800),
        }

    async def revoke_token(self, access_token: str) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.delete(
                    f"{_GH_API}/applications/{config.github_client_id}/token",
                    auth=(config.github_client_id, config.github_client_secret),
                    json={"access_token": access_token},
                )
                return resp.status_code == 204
        except Exception:
            logger.warning("GitHub token revocation failed", exc_info=True)
            return False
```

**Key rules:**
- `provider_name` must be a unique slug — used in URLs (`/connectors/<provider>/callback`)
- `is_configured()` gates registration — connector is skipped at startup if credentials are missing
- `handle_callback()` must return the standardised dict: `access_token`, `refresh_token`, `expires_in`, `scopes`, `account_id`, `account_label`, `provider_meta`
- All HTTP calls use `httpx.AsyncClient` — never block the event loop

---

### Step 2 — Register in ConnectorRegistry

Add the import and instance to `_ALL_CONNECTORS` in `connectors/registry.py`:

```python
from connectors.slack import SlackConnector
from connectors.github import GitHubConnector

_ALL_CONNECTORS: List[BaseConnector] = [
    GmailConnector(),
    SlackConnector(),      # ← add
    GitHubConnector(),     # ← add
]
```

The registry auto-discovers configured connectors at startup. Unconfigured ones are skipped with a log warning — no crash.

---

### Step 3 — Add Config

Add credential fields to `config/settings.py`:

```python
# Inside class Settings:

    # ── Slack OAuth ───────────────────────────────────────────────────
    slack_client_id: str = ""
    slack_client_secret: str = ""

    # ── GitHub OAuth ──────────────────────────────────────────────────
    github_client_id: str = ""
    github_client_secret: str = ""
```

Add env vars to `.env`:

```env
# ── Slack OAuth ──────────────────────────────────────
# 1. https://api.slack.com/apps → Create New App
# 2. OAuth & Permissions → Add redirect URL:
#    http://localhost:8000/api/v1/connectors/slack/callback
SLACK_CLIENT_ID=
SLACK_CLIENT_SECRET=

# ── GitHub OAuth ─────────────────────────────────────
# 1. https://github.com/settings/developers → New OAuth App
# 2. Set callback URL:
#    http://localhost:8000/api/v1/connectors/github/callback
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
```

---

### Step 4 — Create Tool Functions with HITL

Create `tools/<provider>_tools.py`. Mark read-only tools with `@tool("agent_name")` and irreversible tools with `@tool("agent_name", requires_approval=True)`.

**The HITL system auto-detects** tools with `requires_approval=True` at startup — no orchestrator changes needed. When the user triggers an irreversible tool, the HITL dialog fires automatically.

#### Slack Tools — `tools/slack_tools.py`

Tools use the token directly (no SDK — just `httpx`), so no ContextVar service object is needed. The agent fetches the token and passes it to each tool call.

```python
"""Slack tools — channel reading, search, messaging."""

from __future__ import annotations
import logging
from typing import Any, Dict, List

import httpx
from tools import tool

logger = logging.getLogger(__name__)
_SLACK_API = "https://slack.com/api"


async def _slack_api(token: str, method: str, **kwargs) -> Dict[str, Any]:
    """Call a Slack Web API method."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_SLACK_API}/{method}",
            headers={"Authorization": f"Bearer {token}"},
            json=kwargs,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise ValueError(f"Slack API error: {data.get('error')}")
        return data


# ── Read-only tools (no approval) ─────────────────────────────────────

@tool("slack_agent")
async def list_channels(token: str) -> List[Dict[str, Any]]:
    """List public channels in the Slack workspace."""
    data = await _slack_api(token, "conversations.list", types="public_channel", limit=100)
    return [{"id": ch["id"], "name": ch["name"],
             "topic": ch.get("topic", {}).get("value", "")}
            for ch in data.get("channels", [])]


@tool("slack_agent")
async def read_channel_history(token: str, channel_id: str, limit: int = 20) -> List[Dict]:
    """Read recent messages from a channel."""
    data = await _slack_api(token, "conversations.history", channel=channel_id, limit=limit)
    return [{"user": m.get("user"), "text": m.get("text"), "ts": m.get("ts")}
            for m in data.get("messages", [])]


@tool("slack_agent")
async def search_slack_messages(token: str, query: str, count: int = 10) -> List[Dict]:
    """Search messages across the Slack workspace."""
    data = await _slack_api(token, "search.messages", query=query, count=count)
    return [{"channel": m.get("channel", {}).get("name"), "user": m.get("username"),
             "text": m.get("text"), "permalink": m.get("permalink")}
            for m in data.get("messages", {}).get("matches", [])]


# ── Irreversible tools (HITL required) ────────────────────────────────

@tool("slack_agent", requires_approval=True)
async def send_slack_message(token: str, channel_id: str, text: str) -> Dict[str, Any]:
    """
    Post a message to a Slack channel.
    ⚠️ requires_approval — triggers HITL dialog before execution.
    """
    data = await _slack_api(token, "chat.postMessage", channel=channel_id, text=text)
    return {"channel": data.get("channel"), "ts": data.get("ts"),
            "message": data.get("message", {}).get("text")}


@tool("slack_agent", requires_approval=True)
async def send_slack_dm(token: str, user_id: str, text: str) -> Dict[str, Any]:
    """
    Send a direct message to a Slack user.
    ⚠️ requires_approval — triggers HITL dialog before execution.
    """
    dm = await _slack_api(token, "conversations.open", users=user_id)
    channel_id = dm["channel"]["id"]
    data = await _slack_api(token, "chat.postMessage", channel=channel_id, text=text)
    return {"channel": channel_id, "ts": data.get("ts"), "message": text}
```

> **Note:** The `_slack_api` helper uses `json=kwargs` for all methods. Some Slack endpoints (notably `search.messages`) only accept `application/x-www-form-urlencoded`. In production, use `data=kwargs` for search methods or add method-specific handling.

#### GitHub Tools — `tools/github_tools.py`

Same pattern — `httpx` REST calls, token passed from agent. Note the HITL distinction: reading repo info is safe, creating repos/PRs is not.

```python
"""GitHub tools — repo management, PR creation, info retrieval."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import httpx
from tools import tool

logger = logging.getLogger(__name__)
_GH_API = "https://api.github.com"


def _gh_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


# ── Read-only tools (no approval) ─────────────────────────────────────

@tool("github_agent")
async def list_user_repos(
    token: str, sort: str = "updated", limit: int = 10,
) -> List[Dict[str, Any]]:
    """List repositories for the authenticated user."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_GH_API}/user/repos",
            headers=_gh_headers(token),
            params={"sort": sort, "per_page": min(limit, 30)},
        )
        resp.raise_for_status()
    return [{"full_name": r["full_name"], "description": r.get("description"),
             "language": r.get("language"), "stars": r["stargazers_count"],
             "private": r["private"], "html_url": r["html_url"],
             "updated_at": r.get("updated_at")}
            for r in resp.json()]


@tool("github_agent")
async def get_repo_info(token: str, owner: str, repo: str) -> Dict[str, Any]:
    """Get repository metadata (description, stars, language, visibility)."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{_GH_API}/repos/{owner}/{repo}", headers=_gh_headers(token))
        resp.raise_for_status()
        data = resp.json()
    return {
        "full_name": data["full_name"], "description": data.get("description"),
        "language": data.get("language"), "stars": data["stargazers_count"],
        "forks": data["forks_count"], "open_issues": data["open_issues_count"],
        "visibility": data.get("visibility", "public"),
        "default_branch": data["default_branch"],
        "html_url": data["html_url"],
        "created_at": data.get("created_at"), "updated_at": data.get("updated_at"),
    }


@tool("github_agent")
async def list_repo_issues(
    token: str, owner: str, repo: str, state: str = "open", limit: int = 10,
) -> List[Dict[str, Any]]:
    """List issues in a repository."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_GH_API}/repos/{owner}/{repo}/issues",
            headers=_gh_headers(token),
            params={"state": state, "per_page": min(limit, 30)},
        )
        resp.raise_for_status()
    return [{"number": i["number"], "title": i["title"], "state": i["state"],
             "user": i["user"]["login"], "labels": [l["name"] for l in i.get("labels", [])],
             "created_at": i.get("created_at"), "html_url": i["html_url"]}
            for i in resp.json() if "pull_request" not in i]


@tool("github_agent")
async def list_pull_requests(
    token: str, owner: str, repo: str, state: str = "open", limit: int = 10,
) -> List[Dict[str, Any]]:
    """List pull requests in a repository."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_GH_API}/repos/{owner}/{repo}/pulls",
            headers=_gh_headers(token),
            params={"state": state, "per_page": min(limit, 30)},
        )
        resp.raise_for_status()
    return [{"number": p["number"], "title": p["title"], "state": p["state"],
             "user": p["user"]["login"], "head": p["head"]["ref"], "base": p["base"]["ref"],
             "draft": p.get("draft", False),
             "created_at": p.get("created_at"), "html_url": p["html_url"]}
            for p in resp.json()]


# ── Irreversible tools (HITL required) ────────────────────────────────

@tool("github_agent", requires_approval=True)
async def create_repo(
    token: str, name: str, description: str = "", private: bool = False,
) -> Dict[str, Any]:
    """
    Create a new GitHub repository.
    ⚠️ requires_approval — creates a real repository on the user's account.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_GH_API}/user/repos",
            headers=_gh_headers(token),
            json={"name": name, "description": description, "private": private, "auto_init": True},
        )
        resp.raise_for_status()
        data = resp.json()
    logger.info("create_repo → %s", data["full_name"])
    return {"full_name": data["full_name"], "html_url": data["html_url"],
            "private": data["private"], "default_branch": data["default_branch"]}


@tool("github_agent", requires_approval=True)
async def create_pull_request(
    token: str, owner: str, repo: str, title: str, head: str, base: str,
    body: str = "", draft: bool = False,
) -> Dict[str, Any]:
    """
    Create a pull request.
    ⚠️ requires_approval — opens a real PR on the repository.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_GH_API}/repos/{owner}/{repo}/pulls",
            headers=_gh_headers(token),
            json={"title": title, "head": head, "base": base, "body": body, "draft": draft},
        )
        resp.raise_for_status()
        data = resp.json()
    logger.info("create_pull_request → %s#%d", f"{owner}/{repo}", data["number"])
    return {"number": data["number"], "title": data["title"], "html_url": data["html_url"],
            "state": data["state"], "head": head, "base": base, "draft": data.get("draft", False)}
```

**HITL decision guide — when to use `requires_approval=True`:**

| Category | Examples | `requires_approval` |
|----------|----------|-------------------|
| **Read / list / search** | `get_repo_info`, `list_channels`, `search_messages` | `False` |
| **Create / write / send** | `create_repo`, `create_pull_request`, `send_message` | `True` |
| **Delete / modify** | `delete_repo`, `close_issue`, `merge_pr` | `True` |

---

### Step 5 — Create the Agent

Create `agents/<provider>_agent/` with three files: `__init__.py`, `prompts.py`, `agent.py`.

**Two architecture patterns** depending on the SDK:

| Pattern | When to use | Token flow | Example |
|---------|-------------|-----------|---------|
| **ContextVar service** | SDK requires a client/service object (e.g. `googleapiclient`) | `prepare_*_service()` → ContextVar → `_get_service()` | Gmail |
| **Token pass-through** | REST API via `httpx` — no SDK object needed | `get_active_token()` → pass `token` to each tool call | Slack, GitHub |

#### Slack Agent — `agents/slack_agent/prompts.py`

Connector agents need prompts for action planning. The LLM decides which API action to take based on the user's task.

```python
# agents/slack_agent/prompts.py

"""Slack agent prompts — action planning for Slack workspace operations."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict

from security.sanitization import sanitize_user_input
from security.delimiters import wrap_task, wrap_entities, wrap_conversation_history, DELIMITER_SYSTEM_PROMPT
from utils.prompt_utils import format_user_profile


class SlackPrompts:

    @staticmethod
    def action_prompt(
        task: str,
        entities: Dict[str, Any],
        dependency_outputs: Dict[str, Any] | None = None,
        long_term_memory: Dict[str, Any] | None = None,
        conversation_history: list | None = None,
    ) -> str:
        dep_context = ""
        if dependency_outputs:
            dep_context = "\n### Context from other agents\n"
            for aid, data in dependency_outputs.items():
                dep_context += f"- **{aid}**: {str(data)[:400]}\n"

        # Security: sanitize entities
        entity_str = ", ".join(
            f"{sanitize_user_input(str(k)).text}={sanitize_user_input(str(v)).text}"
            for k, v in entities.items()
        ) if entities else "none"

        profile = format_user_profile(long_term_memory or {})

        # Security: wrap conversation history
        conv_section = ""
        if conversation_history:
            conv_lines = ""
            for msg in conversation_history[-10:]:
                role = msg.get("role", "user")
                text = str(msg.get("content", ""))[:800]
                conv_lines += f"- {role}: {text}\n"
            conv_section = wrap_conversation_history(conv_lines)

        # Security: sanitize task
        safe_task = sanitize_user_input(task).text

        return f"""{DELIMITER_SYSTEM_PROMPT}

You are the Slack Action Planner for a multi-agent RAG system.

### Current Date
{datetime.now(timezone.utc).strftime('%A, %B %d, %Y')}

{wrap_task(safe_task)}

{wrap_entities(entity_str)}
{dep_context}
{profile}
{conv_section}

### Available Actions
| Action | When to use |
|--------|-------------|
| search_messages | Search for messages across workspace |
| list_channels | List available channels |
| read_channel | Read recent messages from a channel |
| send_message | Send a message to a channel |

### Output (JSON)
Return EXACTLY:
{{
    "action": "search_messages | list_channels | read_channel | send_message",
    "params": {{"channel": "...", "query": "...", "message": "..."}},
    "reasoning": "brief explanation"
}}"""
```

#### Slack Agent — `agents/slack_agent/agent.py` (token pass-through)

```python
"""Slack Agent — workspace search, channel reading, messaging."""

from __future__ import annotations
import time
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from agents.slack_agent.prompts import SlackPrompts
from config.settings import config
from connectors.token_manager import get_active_token
from utils.schemas import AgentInput, AgentOutput

import logging
logger = logging.getLogger("slack_agent")


class SlackAgent(BaseAgent):
    def __init__(self, tool_registry, llm_provider):
        super().__init__("slack_agent", tool_registry)
        self.llm = llm_provider
        self.prompts = SlackPrompts()

    def get_required_tools(self) -> List[str]:
        return ["list_channels", "read_channel_history", "search_slack_messages",
                "send_slack_message", "send_slack_dm"]

    async def execute(self, task_config: AgentInput) -> AgentOutput:
        start = time.perf_counter()
        user_id = task_config.metadata.get("user_id", "")

        # ── Fresh token every request (no caching, no globals) ────────
        token = await get_active_token(user_id, "slack")
        if not token:
            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=task_config.task,
                task_done=False,
                data={"error": "not_connected"},
            )

        try:
            # HITL-aware effective task (handles enhance/override)
            effective_task = await self._effective_task(task_config)

            plan_result = await self.llm.generate(
                prompt=self.prompts.action_prompt(
                    task=effective_task,
                    entities=task_config.entities,
                    dependency_outputs=task_config.dependency_outputs,
                    long_term_memory=task_config.long_term_memory,
                    conversation_history=task_config.conversation_history,
                ),
                temperature=config.get_agent_model_config("slack_agent")["temperature"],
                output_schema={
                    "action": "list_channels | read_history | search | send_message | send_dm",
                    "params": "dict of tool parameters",
                    "reasoning": "string",
                },
            )
            tokens_used = plan_result.usage.get("total_tokens", 0)
            plan = plan_result.data    # LLMResult.data — parsed JSON dict
            if not isinstance(plan, dict):
                plan = {"action": "search", "params": {"query": task_config.task}}

            action = plan.get("action", "search")
            params = plan.get("params", {})
            result_data: Dict[str, Any] = {"action": action}
            success = True

            # Token is passed directly to each tool — no ContextVar needed
            if action == "list_channels":
                tool_fn = self.get_tool("list_channels")
                result_data["channels"] = await tool_fn(token=token)

            elif action == "read_history":
                tool_fn = self.get_tool("read_channel_history")
                result_data["messages"] = await tool_fn(
                    token=token, channel_id=params.get("channel_id", ""), limit=params.get("limit", 20),
                )

            elif action == "search":
                tool_fn = self.get_tool("search_slack_messages")
                result_data["results"] = await tool_fn(
                    token=token, query=params.get("query", task_config.task),
                )

            elif action == "send_message":
                tool_fn = self.get_tool("send_slack_message")
                result_data["sent"] = await tool_fn(
                    token=token, channel_id=params.get("channel_id", ""), text=params.get("text", ""),
                )

            elif action == "send_dm":
                tool_fn = self.get_tool("send_slack_dm")
                result_data["sent"] = await tool_fn(
                    token=token, user_id=params.get("user_id", ""), text=params.get("text", ""),
                )

            else:
                result_data["error"] = f"Unknown action: {action}"
                success = False

            logger.info("[SlackAgent] Output: %s", result_data)
            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=effective_task,
                task_done=success,
                data=result_data,
                confidence_score=0.85 if success else 0.4,
                resource_usage={
                    "time_taken_ms": int((time.perf_counter() - start) * 1000),
                    "tokens_used": tokens_used,
                },
                depends_on=list(task_config.dependency_outputs.keys()),
                artifacts=[],  # Only code_agent populates this
            )
        except Exception:
            logger.exception("[SlackAgent] Error")
            raise
```

#### GitHub Agent — `agents/github_agent/prompts.py` (HITL-aware)

GitHub agent has HITL-protected tools (`create_repo`, `create_pr`). The prompts must handle both read-only operations and write operations that need user approval.

```python
# agents/github_agent/prompts.py

"""GitHub agent prompts — action planning for repo management, issues, and PRs."""

from __future__ import annotations
from typing import Any, Dict

from security.sanitization import sanitize_user_input
from security.delimiters import wrap_task, wrap_conversation_history, DELIMITER_SYSTEM_PROMPT
from utils.prompt_utils import format_user_profile


class GitHubPrompts:

    @staticmethod
    def action_prompt(
        task: str,
        entities: Dict[str, Any],
        dependency_outputs: Dict[str, Any] | None = None,
        long_term_memory: Dict[str, Any] | None = None,
        conversation_history: list | None = None,
    ) -> str:
        dep_context = ""
        if dependency_outputs:
            dep_context = "\n### Context from other agents\n"
            for aid, data in dependency_outputs.items():
                dep_context += f"- **{aid}**: {str(data)[:500]}\n"

        profile = format_user_profile(long_term_memory or {})

        # Security: wrap conversation history
        history_context = ""
        if conversation_history:
            conv_lines = ""
            for msg in conversation_history[-10:]:
                role = msg.get("role", "user")
                text = str(msg.get("content", ""))[:800]
                conv_lines += f"- {role}: {text}\n"
            history_context = "\n### Recent conversation\n" + wrap_conversation_history(conv_lines)

        # Security: sanitize entities
        entities_str = ""
        if entities:
            entities_str = "\n### Extracted entities\n"
            for k, v in entities.items():
                entities_str += f"- {sanitize_user_input(str(k)).text}: {sanitize_user_input(str(v)).text}\n"

        # Security: sanitize task
        safe_task = sanitize_user_input(task).text

        return f"""{DELIMITER_SYSTEM_PROMPT}

You are a GitHub Integration Agent in a multi-agent system.

{wrap_task(safe_task)}
{dep_context}
{profile}
{history_context}
{entities_str}

### Available Actions
- **list_repos** — list the authenticated user's repositories (no owner/repo needed)
- **repo_info**   — get metadata for a specific repository (owner, repo required)
- **list_issues** — list issues in a repository (owner, repo required)
- **list_prs**    — list pull requests in a repository (owner, repo required)
- **create_repo** — create a new repository (name required) ⚠️ requires HITL approval
- **create_pr**   — create a pull request (owner, repo, title, head, base required) ⚠️ requires HITL approval

### Instructions
1. Parse the user's request to determine which action to take.
2. Extract the required parameters (owner, repo, name, etc.) from the task and entities.
3. If the user mentions a repo like "owner/repo", split it into owner and repo.
4. If the user asks about their repos without specifying one, use **list_repos**.
5. For create operations, include a clear description/body.

### Output format
Return JSON:
```json
{{
  "action": "list_repos | repo_info | list_issues | list_prs | create_repo | create_pr",
  "params": {{...}},
  "reasoning": "why this action"
}}
```"""
```

#### GitHub Agent — `agents/github_agent/agent.py` (token pass-through)

```python
"""GitHub Agent — repo info, issues, PRs, repo/PR creation."""

from __future__ import annotations
import time
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from agents.github_agent.prompts import GitHubPrompts
from config.settings import config
from connectors.token_manager import get_active_token
from utils.schemas import AgentInput, AgentOutput

import logging
logger = logging.getLogger("github_agent")


class GitHubAgent(BaseAgent):
    def __init__(self, tool_registry, llm_provider):
        super().__init__("github_agent", tool_registry)
        self.llm = llm_provider
        self.prompts = GitHubPrompts()

    def get_required_tools(self) -> List[str]:
        return ["list_user_repos", "get_repo_info", "list_repo_issues",
                "list_pull_requests", "create_repo", "create_pull_request"]

    async def execute(self, task_config: AgentInput) -> AgentOutput:
        start = time.perf_counter()
        user_id = task_config.metadata.get("user_id", "")

        # ── Fresh token every request ─────────────────────────────────
        token = await get_active_token(user_id, "github")
        if not token:
            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=task_config.task,
                task_done=False,
                data={"error": "not_connected"},
            )

        try:
            effective_task = await self._effective_task(task_config)

            plan_result = await self.llm.generate(
                prompt=self.prompts.action_prompt(
                    task=effective_task,
                    entities=task_config.entities,
                    dependency_outputs=task_config.dependency_outputs,
                    long_term_memory=task_config.long_term_memory,
                    conversation_history=task_config.conversation_history,
                ),
                temperature=config.get_agent_model_config("github_agent")["temperature"],
                output_schema={
                    "action": "list_repos | repo_info | list_issues | list_prs | create_repo | create_pr",
                    "params": "dict of tool parameters",
                    "reasoning": "string",
                },
            )
            tokens_used = plan_result.usage.get("total_tokens", 0)
            plan = plan_result.data    # LLMResult.data — parsed JSON dict
            if not isinstance(plan, dict):
                plan = {"action": "repo_info", "params": {}}

            action = plan.get("action", "repo_info")
            params = plan.get("params", {})
            result_data: Dict[str, Any] = {"action": action}
            success = True

            if action == "list_repos":
                tool_fn = self.get_tool("list_user_repos")
                result_data["repos"] = await tool_fn(
                    token=token, sort=params.get("sort", "updated"),
                )

            elif action == "repo_info":
                tool_fn = self.get_tool("get_repo_info")
                result_data["repo"] = await tool_fn(
                    token=token, owner=params.get("owner", ""), repo=params.get("repo", ""),
                )

            elif action == "list_issues":
                tool_fn = self.get_tool("list_repo_issues")
                result_data["issues"] = await tool_fn(
                    token=token, owner=params.get("owner", ""), repo=params.get("repo", ""),
                    state=params.get("state", "open"),
                )

            elif action == "list_prs":
                tool_fn = self.get_tool("list_pull_requests")
                result_data["pull_requests"] = await tool_fn(
                    token=token, owner=params.get("owner", ""), repo=params.get("repo", ""),
                    state=params.get("state", "open"),
                )

            elif action == "create_repo":
                tool_fn = self.get_tool("create_repo")
                result_data["created"] = await tool_fn(
                    token=token, name=params.get("name", ""),
                    description=params.get("description", ""),
                    private=params.get("private", False),
                )

            elif action == "create_pr":
                tool_fn = self.get_tool("create_pull_request")
                result_data["pull_request"] = await tool_fn(
                    token=token, owner=params.get("owner", ""), repo=params.get("repo", ""),
                    title=params.get("title", ""), head=params.get("head", ""),
                    base=params.get("base", "main"), body=params.get("body", ""),
                    draft=params.get("draft", False),
                )

            else:
                result_data["error"] = f"Unknown action: {action}"
                success = False

            logger.info("[GitHubAgent] Output: %s", result_data)
            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=effective_task,
                task_done=success,
                data=result_data,
                confidence_score=0.85 if success else 0.4,
                resource_usage={
                    "time_taken_ms": int((time.perf_counter() - start) * 1000),
                    "tokens_used": tokens_used,
                },
                depends_on=list(task_config.dependency_outputs.keys()),
                artifacts=[],  # Only code_agent populates this
            )
        except Exception:
            logger.exception("[GitHubAgent] Error")
            raise
```

#### When to Use ContextVar vs Token Pass-Through

If the provider has a **sync SDK that builds a client object** (like `googleapiclient`), use the Gmail pattern:

```python
# In tools/<provider>_tools.py — ContextVar pattern (for sync SDKs)

_current_service: contextvars.ContextVar[Any] = contextvars.ContextVar("_current_service", default=None)

async def prepare_service(user_id: str) -> None:
    """Called once per request by the agent's execute()."""
    token = await get_active_token(user_id, "provider_name")
    if not token:
        raise RuntimeError("Provider not connected.")
    creds = Credentials(token=token)
    service = await asyncio.to_thread(build_sdk_client, creds)   # non-blocking
    _current_service.set(service)

def _get_service():
    """Called by each tool function."""
    svc = _current_service.get(None)
    if svc is None:
        raise RuntimeError("Service not initialised. Call prepare_service() first.")
    return svc

# In the agent's execute():
async def execute(self, task_config):
    await prepare_service(user_id)     # fresh token → ContextVar
    # ... tools call _get_service() internally
```

If the provider uses a **REST API via `httpx`** (no SDK needed), pass the token directly — simpler and equally safe:

```python
# In the agent's execute() — token pass-through pattern (for REST APIs)
async def execute(self, task_config):
    token = await get_active_token(user_id, "provider_name")
    # ... pass token= to each tool call
```

Both patterns guarantee: fresh token per request, auto-refresh, no global state, no memory leak.

---

### Step 6 — Register the Agent

Add entries to `config/agent_registry.yaml`:

```yaml
  slack_agent:
    description: "Interacts with Slack — read channels, search messages, send messages"
    capabilities:
      - slack_read
      - slack_search
      - slack_send
    tools:
      - list_channels
      - read_channel_history
      - search_slack_messages
      - send_slack_message
      - send_slack_dm
    typical_use_cases:
      - "Search Slack for messages about deployment"
      - "Send a message to #general"
      - "Read the latest messages in #engineering"
    default_timeout: 30
    max_retries: 2

  github_agent:
    description: "Interacts with GitHub — repo info, issues, PRs, repo/PR creation"
    capabilities:
      - github_read
      - github_issues
      - github_prs
      - github_create
    tools:
      - list_user_repos
      - get_repo_info
      - list_repo_issues
      - list_pull_requests
      - create_repo
      - create_pull_request
    typical_use_cases:
      - "Show me info about the MRAG repo"
      - "List open PRs on our project"
      - "Create a new repository called my-tool"
      - "Open a PR from feature-branch to main"
    default_timeout: 30
    max_retries: 2
```

Add model config to `config/settings.py`:

```python
    slack_model_provider: str = "openai"
    slack_model: str = "gpt-4o-mini"
    slack_temperature: float = 0.3

    github_model_provider: str = "openai"
    github_model: str = "gpt-4o-mini"
    github_temperature: float = 0.2
```

Then add the mapping in `get_agent_model_config()` so `config.get_agent_model_config("slack_agent")` returns your configured values instead of the fallback default:

```python
    # config/settings.py — inside get_agent_model_config()
    mapping = {
        # ... existing agents ...
        "slack_agent": (self.slack_model_provider, self.slack_model, self.slack_temperature),
        "github_agent": (self.github_model_provider, self.github_model, self.github_temperature),
    }
```

---

### Step 7 — Wire the Agent Instance

All agents are built centrally in `core/agent_factory.py`. Add your agents to `build_agent_instances()`:

```python
# core/agent_factory.py

from agents.slack_agent.agent import SlackAgent
from agents.github_agent.agent import GitHubAgent

def build_agent_instances(
    registry: ToolRegistry,
    model_override: str | None = None,    # ← from user model dropdown
) -> Dict[str, BaseAgent]:
    # When model_override is set, ALL agents use that model
    if model_override and model_override in _VALID_MODELS:
        _override_llm = get_llm_provider(model_provider_for(model_override), default_model=model_override)
    else:
        _override_llm = None

    def _llm(agent_name: str):
        if _override_llm:
            return _override_llm                          # user-selected model
        cfg = config.get_agent_model_config(agent_name)
        return get_llm_provider(cfg["provider"], default_model=cfg["model"])

    return {
        # ... existing agents ...
        "slack_agent": SlackAgent(
            tool_registry=registry,
            llm_provider=_llm("slack_agent"),
        ),
        "github_agent": GitHubAgent(
            tool_registry=registry,
            llm_provider=_llm("github_agent"),
        ),
    }
```

`api/routes.py` and `api/streaming.py` both call `build_agent_instances()` — you do **not** edit them. The streaming endpoint automatically passes the user's model selection as `model_override`.

---

### What Happens at Runtime

#### Example 1: "Send a deployment summary to #general on Slack"

```
1. Master Agent  → routes to slack_agent (capabilities match "slack_send")
2. Orchestrator  → checks slack_agent tools → send_slack_message has requires_approval=True
3. HITL triggers → SSE event "hitl_required" sent to frontend
4. User sees     → "slack_agent wants to use send_slack_message. Approve?"
5. User approves → optionally adds "add a 🚀 emoji at the end"
6. _effective_task() → classifies instructions as "enhance" → agent gets both tasks
7. Agent calls   → get_active_token(user_id, "slack") → fresh token from DB
8. Tool executes → send_slack_message(token, channel_id, text) → message posted
9. Composer      → synthesises "I posted the deployment summary to #general ✅"
```

#### Example 2: "Create a new repo called my-service on GitHub"

```
1. Master Agent  → routes to github_agent (capabilities match "github_create")
2. Orchestrator  → checks github_agent tools → create_repo has requires_approval=True
3. HITL triggers → "github_agent wants to use create_repo. Approve?"
4. User approves → "make it private"
5. _effective_task() → classifies as "enhance" → original task + user instruction merged
6. Agent calls   → get_active_token(user_id, "github") → fresh token
7. Tool executes → create_repo(token, name="my-service", private=True)
8. Composer      → "Created private repo my-service ✅ — https://github.com/you/my-service"
```

#### Example 3: "Show me open PRs on our project" (read-only, no HITL)

```
1. Master Agent  → routes to github_agent
2. Orchestrator  → checks tools → list_pull_requests has requires_approval=False
3. No HITL       → agent executes immediately
4. Agent calls   → get_active_token() → list_pull_requests(token, owner, repo)
5. Composer      → formats PR list as a readable summary
```

---

### Checklist

| # | File | Action |
|---|------|--------|
| 1 | `connectors/<provider>.py` | Subclass `BaseConnector` — OAuth flow |
| 2 | `connectors/registry.py` | Add instance to `_ALL_CONNECTORS` |
| 3 | `config/settings.py` + `.env` | Add `<provider>_client_id`, `<provider>_client_secret`, model config |
| 4 | `tools/<provider>_tools.py` | `@tool` (safe) and `@tool(requires_approval=True)` (irreversible) |
| 5 | `agents/<provider>_agent/` | `prompts.py` + `agent.py` with `_effective_task()` + token retrieval |
| 6 | `config/agent_registry.yaml` | Register capabilities, tools, use cases |
| 7 | `core/agent_factory.py` | Add agent to `build_agent_instances()` |

### What You Get for Free

| Feature | How |
|---------|-----|
| **OAuth UI** | "Connect" button auto-appears in sidebar for every registered connector |
| **Token management** | Auto-refresh, Fernet encryption at rest, revocation on disconnect |
| **HITL approval** | Any `requires_approval=True` tool triggers the approval dialog automatically |
| **HITL instructions** | User can modify task during approval — auto-classified as enhance/override |
| **Dashboard** | `user_connections` table shows all connections in admin dashboard |
| **"Not configured" UI** | If env vars missing → grayed out in sidebar with "Setup required" |
| **No orchestrator changes** | HITL, toposort, parallel execution — all work identically for new agents |

---

## Quick Start

```bash
cp .env.example .env
docker compose up -d postgres qdrant
uv sync
.venv\Scripts\activate
pip install -r requirements.txt
uv run uvicorn main:app --reload
```
---

## How to Add a New Agent with no HITL Tool basic one:

Adding a new agent is a **7-step process** touching 6 files (3 new, 3 existing). Below is a complete walkthrough using a fictional **`summary_agent`** as the example.

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1 │ config/agent_registry.yaml  — register capabilities      │
│  Step 2 │ config/settings.py          — add model/temp settings    │
│  Step 3 │ .env                        — set model env vars         │
│  Step 4 │ tools/summary_tools.py      — implement tool functions   │
│  Step 5 │ agents/summary_agent/       — prompts.py + agent.py      │
│  Step 6 │ core/agent_factory.py        — inject agent instance     │
│  Step 7 │ tests/                      — add agent tests   # Optional         │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Step 1 — Register in `config/agent_registry.yaml`

The Master Agent reads this file to know what agents exist and when to select them. Append a new block:

```yaml
# config/agent_registry.yaml

agents:
  # ... existing agents ...

  summary_agent:
    description: "Generates concise summaries of long-form text, documents, or agent outputs"
    capabilities:
      - text_summarisation
      - key_point_extraction
      - executive_briefing
    tools:
      - summarise_text        # must match @tool function names in Step 4
      - extract_key_points
    typical_use_cases:
      - "Summarise this document in 3 bullet points"
      - "Give me an executive brief of last quarter's report"
      - "Condense these search results into a paragraph"
    default_timeout: 30
    max_retries: 2
```

**Key rules:**
- `capabilities` drives Master Agent's selection logic (what the agent *can* do)
- `tools` must exactly match the function names decorated with `@tool` in Step 4
- `typical_use_cases` are injected into the Master Agent prompt so it knows *when* to select this agent

---

### Step 2 — Add Model Settings in `config/settings.py`

Each agent gets its own model/provider/temperature triple:

```python
# config/settings.py — inside class Settings(BaseSettings):

    # ── Summary Agent Model ─────────────────────────────────────────────
    summary_model_provider: str = "openai"
    summary_model: str = "gpt-4o-mini"
    summary_temperature: float = 0.3
```

Then add the mapping in `get_agent_model_config()`:

```python
    mapping = {
        # ... existing agents ...
        "summary_agent": (self.summary_model_provider, self.summary_model, self.summary_temperature),
    }
```

---

### Step 3 — Add Env Vars to `.env` / `.env.example`

```env
# Summary Agent
SUMMARY_MODEL_PROVIDER=openai
SUMMARY_MODEL=gpt-4o-mini
SUMMARY_TEMPERATURE=0.3
```

---

### Step 4 — Create Tool Functions in `tools/summary_tools.py`

Every tool is a standalone async function decorated with `@tool("agent_name")`. The decorator declares which agents may call it.

```python
# tools/summary_tools.py

"""Summary tools — registered via @tool decorator."""

from __future__ import annotations
from typing import Any, Dict, List
from tools import tool


@tool("summary_agent")
async def summarise_text(
    text: str,
    max_length: int = 500,
    style: str = "bullet_points",
) -> Dict[str, Any]:
    """
    Summarise the given text.

    Parameters
    ----------
    text        : the content to summarise
    max_length  : target summary length (chars)
    style       : "bullet_points" | "paragraph" | "executive"
    """
    # Your implementation here — call an LLM, use a library, etc.
    # Return a dict consumable by the agent.
    return {
        "summary": "...",
        "word_count": 0,
        "style": style,
    }


@tool("summary_agent")
async def extract_key_points(
    text: str,
    num_points: int = 5,
) -> List[str]:
    """Extract the top-N key points from the text."""
    return ["point 1", "point 2"]
```

**Key rules:**
- File must be named `*_tools.py` inside `tools/` — the `ToolRegistry.auto_discover_tools()` scans `tools/*_tools.py` automatically.
- The `@tool("summary_agent")` string must match the agent name in `agent_registry.yaml`.
- Multiple agents can share a tool: `@tool("summary_agent", "rag_agent")`.

---

### Step 5 — Create the Agent Package

Create a folder `agents/summary_agent/` with three files:

#### 5a. `agents/summary_agent/__init__.py`

```python
# empty — makes it a package
```

#### 5b. `agents/summary_agent/prompts.py`

Every prompt class follows the same pattern: static methods that accept task data + `long_term_memory` and return a formatted prompt string. Use the shared `format_user_profile()` from `utils/prompt_utils.py` — **do not** duplicate it in your prompt class.

**IMPORTANT — Security Guardrails**: All user-supplied data (task, entities, conversation history) must be sanitized and wrapped with delimiters to prevent prompt injection attacks. See the [Security Guardrails](#security-guardrails) section for details.

```python
# agents/summary_agent/prompts.py

"""Summary agent prompts — production-quality, personalised, secure."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict

from security.sanitization import sanitize_user_input
from security.delimiters import (
    wrap_task,
    wrap_entities,
    wrap_conversation_history,
    DELIMITER_SYSTEM_PROMPT,
)
from utils.prompt_utils import format_user_profile


class SummaryPrompts:

    @staticmethod
    def summarise_prompt(
        task: str,
        source_text: str,
        entities: Dict[str, Any],
        dependency_outputs: Dict[str, Any] | None = None,
        long_term_memory: Dict[str, Any] | None = None,
        conversation_history: list | None = None,
    ) -> str:
        dep_context = ""
        if dependency_outputs:
            dep_context = "\n### Context from other agents\n"
            for aid, data in dependency_outputs.items():
                dep_context += f"- **{aid}**: {str(data)[:500]}\n"

        # Security: sanitize entities
        entity_str = ", ".join(
            f"{sanitize_user_input(str(k)).text}={sanitize_user_input(str(v)).text}"
            for k, v in entities.items()
        ) if entities else "none"

        profile = format_user_profile(long_term_memory or {})

        # Security: wrap conversation history with delimiters
        history_context = ""
        if conversation_history:
            history_lines = ""
            for msg in conversation_history[-10:]:
                role = msg.get("role", "user")
                text = str(msg.get("content", ""))[:800]
                history_lines += f"- {role}: {text}\n"
            history_context = "\n### Recent conversation\n" + wrap_conversation_history(history_lines)

        current_date = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

        # Security: sanitize task and wrap with delimiters
        safe_task = sanitize_user_input(task).text

        # Security: prepend DELIMITER_SYSTEM_PROMPT to instruct LLM
        return f"""{DELIMITER_SYSTEM_PROMPT}

You are a Summarisation Expert in a multi-agent RAG system.

### Current Date
{current_date}

{wrap_task(safe_task)}

{wrap_entities(entity_str)}
{dep_context}
{profile}
{history_context}
### Source Text
{source_text[:5000]}

### Instructions
1. Produce a clear, accurate summary that captures the key information.
2. Match the user's preferred detail level and tone from the User Profile.
3. Respond in the user's preferred language if specified.
4. Cite specific facts, figures, and dates from the source.

### Summary"""
```

#### 5c. `agents/summary_agent/agent.py`

Inherit from `BaseAgent`, implement `execute()` and `get_required_tools()`.

```python
# agents/summary_agent/agent.py

"""Summary Agent — generates summaries from documents or upstream agent data."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from agents.summary_agent.prompts import SummaryPrompts
from config.settings import config
from utils.schemas import AgentInput, AgentOutput

import logging
logger = logging.getLogger("summary_agent")


class SummaryAgent(BaseAgent):
    def __init__(self, tool_registry, llm_provider):
        super().__init__("summary_agent", tool_registry)   # ← must match registry key
        self.llm = llm_provider
        self.prompts = SummaryPrompts()

    def get_required_tools(self) -> List[str]:
        """Declare tools this agent needs (validated at startup)."""
        return ["summarise_text", "extract_key_points"]

    async def execute(self, task_config: AgentInput) -> AgentOutput:
        start = time.perf_counter()

        try:
            # HITL-aware effective task (returns task unchanged if no HITL context)
            effective_task = await self._effective_task(task_config)

            # Build the source text from dependency outputs or the task itself
            source_text = ""
            if task_config.dependency_outputs:
                for dep_data in task_config.dependency_outputs.values():
                    if isinstance(dep_data, dict):
                        source_text += str(dep_data.get("chunks", dep_data)) + "\n"
                    else:
                        source_text += str(dep_data) + "\n"
            else:
                source_text = effective_task

            # Ask LLM to summarise
            prompt = self.prompts.summarise_prompt(
                task=effective_task,
                source_text=source_text,
                entities=task_config.entities,
                dependency_outputs=task_config.dependency_outputs,
                long_term_memory=task_config.long_term_memory,
                conversation_history=task_config.conversation_history,
            )
            llm_result = await self.llm.generate(
                prompt=prompt,
                temperature=config.summary_temperature,
            )
            summary = llm_result.text    # LLMResult.text — raw string response
            tokens_used = llm_result.usage.get("total_tokens", 0)

            logger.info("[SummaryAgent] Output: %s", summary[:200])
            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=effective_task,
                task_done=True,
                data={"summary": summary},
                confidence_score=0.9,
                resource_usage={
                    "time_taken_ms": int((time.perf_counter() - start) * 1000),
                    "tokens_used": tokens_used,
                },
                depends_on=list(task_config.dependency_outputs.keys()),
                artifacts=[],  # Only code_agent populates this with file artifacts
            )

        except Exception:
            logger.exception("[SummaryAgent] Error for input: %s", task_config)
            raise   # let execute_with_retry handle retries
```

> **Note on Persona Support**
>
> Most agents do **NOT** need to handle persona — only two agents use it:
> - **Composer Agent** — Applies persona to the final user-facing response
> - **Code Agent** — Applies persona to generated files (PDFs, charts) that are directly served to users
>
> The Composer Agent handles persona styling for all other agents' outputs, so you don't need to extract or pass persona in your custom agent. The persona is already available in `task_config.metadata["persona"]` if you ever need it for a special use case (e.g., an agent that generates user-facing content directly).

---

### Step 6 — Wire the Agent Instance

All agents are built centrally in `core/agent_factory.py`. Add your agent to the `build_agent_instances()` function — `api/routes.py` and `api/streaming.py` both call this function, so you do **not** touch them:

```python
# core/agent_factory.py

from agents.summary_agent.agent import SummaryAgent

def build_agent_instances(
    registry: ToolRegistry,
    model_override: str | None = None,    # ← from user model dropdown
) -> Dict[str, BaseAgent]:
    # When model_override is set, ALL agents use that model
    if model_override and model_override in _VALID_MODELS:
        _override_llm = get_llm_provider(model_provider_for(model_override), default_model=model_override)
    else:
        _override_llm = None

    def _llm(agent_name: str):
        if _override_llm:
            return _override_llm                          # user-selected model
        cfg = config.get_agent_model_config(agent_name)
        return get_llm_provider(cfg["provider"], default_model=cfg["model"])

    return {
        # ... existing agents ...
        "summary_agent": SummaryAgent(
            tool_registry=registry,
            llm_provider=_llm("summary_agent"),
        ),
    }
```

---

### Step 7 — Add Tests

Create `tests/test_summary_agent.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from agents.summary_agent.agent import SummaryAgent
from utils.schemas import AgentInput

@pytest.mark.asyncio
async def test_summary_agent_execute():
    mock_registry = MagicMock()
    mock_registry.get_tool.return_value = AsyncMock(return_value={"summary": "test"})

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        text="This is a concise summary.",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )

    agent = SummaryAgent(mock_registry, mock_llm)

    task_input = AgentInput(
        agent_id="summary-001",
        agent_name="summary_agent",
        task="Summarise the quarterly report",
        entities={},
        tools=["summarise_text"],
        dependency_outputs={},
        long_term_memory={"critical": {"user_name": "Alice"}, "preferences": {"tone": "casual"}},
    )

    result = await agent.execute(task_input)

    assert result.task_done is True
    assert "summary" in result.data
    mock_llm.generate.assert_called_once()
```

---

### Checklist

| # | File | Action |
|---|------|--------|
| 1 | `config/agent_registry.yaml` | Add agent block with capabilities, tools, use cases |
| 2 | `config/settings.py` | Add `*_model_provider`, `*_model`, `*_temperature` fields + mapping entry |
| 3 | `.env` / `.env.example` | Add corresponding env vars |
| 4 | `tools/<name>_tools.py` | Create `@tool("agent_name")` functions |
| 5a | `agents/<name>/__init__.py` | Empty package file |
| 5b | `agents/<name>/prompts.py` | Prompt class using `format_user_profile()` from `utils/prompt_utils.py` |
| 5c | `agents/<name>/agent.py` | Subclass `BaseAgent`, implement `execute()` + `get_required_tools()` |
| 6 | `core/agent_factory.py` | Add agent to `build_agent_instances()` return dict |
| 7 | `tests/test_<name>_agent.py` | Unit test for the new agent |

### Architecture Rules

- **`agent_name` must be consistent** across: `agent_registry.yaml` key, `BaseAgent.__init__("name")`, `@tool("name")`, and `agent_instances["name"]`.
- **Tools are auto-discovered** — any `tools/*_tools.py` file is scanned at startup. No manual registration needed.
- **User profile formatting** lives in `utils/prompt_utils.py` → `format_user_profile()`. Import it in your prompts — never duplicate it.
- **Long-term memory** is automatically loaded and injected by the Orchestrator into every `AgentInput.long_term_memory`. Pass it to your prompt methods.
- **Retry/timeout** is handled by `BaseAgent.execute_with_retry()`. Your `execute()` should raise on failure, not catch-and-suppress.
- **Dependencies** are declared by the Master Agent in the execution plan (`depends_on`). The topological sort ensures your agent only runs after its dependencies complete.
- **Always inject the current date** into your agent's prompt. Without it, the LLM has no temporal awareness — queries like "recent news", "this week", or "current status" will return stale results. Use `datetime.now(timezone.utc).strftime('%A, %B %d, %Y')` and add a `### Current Date` section near the top of your prompt. This is already done in: Master Agent, Web Search Agent, Mail Agent, and Composer Agent.

---

## Adding HITL (Human-in-the-Loop) to an Agent

Some tools perform **irreversible actions** — sending emails, executing code, making API calls. The HITL system lets you gate these tools behind a **user approval step** before the agent runs. All state is persisted in PostgreSQL — no in-memory coordination, survives server restarts, works across multiple workers.

### How HITL Works — Full Flow

```mermaid
sequenceDiagram
    participant User as User (Frontend)
    participant SSE as SSE Stream<br/>(streaming.py)
    participant Orch as Orchestrator
    participant DB as PostgreSQL<br/>(hitl_requests)
    participant HITL as POST /hitl/respond
    participant Agent as Mail Agent

    User->>SSE: POST /query/stream<br/>"send the Q3 report to john@"
    SSE->>Orch: execute_plan(plan, context, on_hitl_needed)

    Note over Orch: Stage 1: rag_agent → runs normally

    Orch->>Orch: Stage 2: mail_agent has<br/>send_email (requires_approval=True)
    Orch->>DB: INSERT hitl_requests<br/>(status='pending', expires_at=+120s)
    DB-->>Orch: request_id = abc-123

    Orch->>SSE: Queue hitl_required SSE event
    SSE-->>User: event: hitl_required<br/>{request_id, agent, tools, task}

    Note over User: User sees approval dialog

    loop Poll DB every 6s
        Orch->>DB: SELECT status WHERE request_id=abc-123
        DB-->>Orch: status = 'pending'
    end

    User->>HITL: POST /hitl/respond<br/>{request_id, decision: "approved",<br/>instructions: "only send to john@"}
    HITL->>DB: UPDATE status='approved',<br/>user_instructions='only send to john@'

    Orch->>DB: SELECT status WHERE request_id=abc-123
    DB-->>Orch: status = 'approved'

    Orch->>SSE: Queue hitl_approved SSE event
    SSE-->>User: event: hitl_approved

    Orch->>Agent: execute(AgentInput with hitl_context)
    Agent-->>Orch: AgentOutput (task_done=true)

    Note over Orch: Stage 3: composer runs

    SSE-->>User: event: token (streamed answer)
    SSE-->>User: event: done
```

### What Happens When User Denies

```mermaid
sequenceDiagram
    participant User as User
    participant HITL as POST /hitl/respond
    participant SSE as SSE Stream
    participant Orch as Orchestrator
    participant DB as PostgreSQL

    Orch->>DB: INSERT hitl_requests (pending)
    SSE-->>User: event: hitl_required

    User->>HITL: POST /hitl/respond<br/>{decision: "denied"}
    HITL->>DB: UPDATE status='denied'

    Orch->>DB: Poll → status='denied'
    SSE-->>User: event: hitl_denied

    Note over Orch: Create synthetic AgentOutput:<br/>task_done=False<br/>error="denied_by_user"<br/>partial_data={reason, tools}

    Note over Orch: Downstream agents see<br/>dependency_outputs[mail_agent_0] =<br/>{reason: "denied_by_user"}<br/>→ Composer explains what happened
```

### What Happens on Timeout

```mermaid
sequenceDiagram
    participant User as User (Frontend)
    participant SSE as SSE Stream
    participant Orch as Orchestrator
    participant DB as PostgreSQL

    Orch->>DB: INSERT hitl_requests (pending, expires_at=+120s)
    SSE-->>User: event: hitl_required

    loop 120 seconds of polling
        Orch->>DB: SELECT status → 'pending'
    end

    Note over DB: expires_at reached

    Orch->>DB: UPDATE status='timed_out'
    SSE-->>User: event: hitl_timeout

    Note over Orch: Treated as denial:<br/>task_done=False, error="hitl_timeout"
```

### HITL Architecture — Internal Components

```mermaid
graph TD
    subgraph "Frontend"
        FE["SSE EventSource listener"]
        Dialog["Approval Dialog<br/><i>Approve / Deny / instructions</i>"]
        FE -->|"hitl_required event"| Dialog
        Dialog -->|"POST /hitl/respond"| HitlEndpoint
    end

    subgraph "API Layer"
        Stream["POST /query/stream<br/><i>streaming.py</i>"]
        HitlEndpoint["POST /hitl/respond<br/><i>routes.py</i>"]
    end

    subgraph "Core"
        Orch["Orchestrator<br/><i>on_hitl_needed callback</i>"]
        CB["HITL Callback<br/><i>_make_hitl_callback()</i>"]
        Queue["asyncio.Queue<br/><i>per-request SSE pipe</i>"]
    end

    subgraph "Persistence"
        DB[("hitl_requests table<br/><i>PostgreSQL</i>")]
    end

    subgraph "Registry"
        ToolReg["ToolRegistry<br/><i>_requires_approval dict</i>"]
    end

    Stream -->|"creates"| Queue
    Stream -->|"creates"| CB
    Stream -->|"passes callback"| Orch
    Orch -->|"checks tools"| ToolReg
    Orch -->|"calls"| CB
    CB -->|"INSERT pending"| DB
    CB -->|"push SSE event"| Queue
    CB -->|"poll every 6s"| DB
    HitlEndpoint -->|"UPDATE decision"| DB
    Queue -->|"yield SSE"| Stream
    Stream -->|"SSE events"| FE

    style DB fill:#4CAF50,color:#fff
    style Queue fill:#FF9800,color:#fff
    style ToolReg fill:#2196F3,color:#fff
```

### Parallel Agents + HITL

When a stage has both HITL and non-HITL agents, they run in parallel via `asyncio.gather`. The non-HITL agent executes immediately while the HITL agent blocks on DB polling:

```mermaid
gantt
    title Stage 2: mail_agent (HITL) + web_agent (no HITL)
    dateFormat X
    axisFormat %s

    section web_agent
    Execute normally           :done, 0, 3

    section mail_agent
    Wait for approval (polling DB) :active, 0, 8
    Execute after approval         :done, 8, 11

    section Stage 3
    Composer (waits for both)      :12, 15
```

---

### `hitl_requests` Database Table

| Column | Type | Description |
|--------|------|-------------|
| `request_id` | `UUID PK` | Unique identifier for this approval request |
| `conversation_id` | `UUID FK → conversations` | Links to the conversation that triggered this |
| `agent_id` | `TEXT` | Agent instance ID (e.g. `mail_agent_0`) |
| `agent_name` | `TEXT` | Agent type (e.g. `mail_agent`) |
| `tool_names` | `TEXT[]` | Tools requiring approval (e.g. `{send_email, reply_to_message}`) |
| `task_description` | `TEXT` | What the agent was asked to do |
| `status` | `VARCHAR(16)` | `pending` → `approved` / `denied` / `timed_out` / `expired` |
| `user_instructions` | `TEXT` | Optional instructions from user on approval |
| `created_at` | `TIMESTAMPTZ` | When the request was created |
| `responded_at` | `TIMESTAMPTZ` | When the user responded (NULL if pending) |
| `expires_at` | `TIMESTAMPTZ` | Auto-deny deadline (`created_at + timeout`) |

**Status lifecycle:**

```mermaid
stateDiagram-v2
    [*] --> pending : INSERT (orchestrator)
    pending --> approved : User approves (POST /hitl/respond)
    pending --> denied : User denies (POST /hitl/respond)
    pending --> timed_out : expires_at reached (poll detects)
    pending --> expired : Server restart cleanup (startup job)

    approved --> [*]
    denied --> [*]
    timed_out --> [*]
    expired --> [*]
```

---

### SSE Event Types for HITL

| Event | When | Payload |
|-------|------|---------|
| `hitl_required` | Agent has HITL tools, approval needed | `{request_id, agent_id, agent_name, tool_names, task_description, timeout_seconds}` |
| `hitl_approved` | User approved the request | `{request_id, agent_id, user_instructions}` |
| `hitl_denied` | User denied the request | `{request_id, agent_id}` |
| `hitl_timeout` | Approval timed out | `{request_id, agent_id}` |

---

### Adding a New Agent with HITL Tools

Follow the standard ["Adding a New Agent"](#adding-a-new-agent-step-by-step) guide above, with these additions:

#### Step 1 — Mark tools with `requires_approval=True`

In `tools/<name>_tools.py`, add the flag to any tool that performs an irreversible action:

```python
from tools import tool

# Safe tool — no approval needed
@tool("deploy_agent")
async def check_deploy_status(deploy_id: str) -> Dict[str, Any]:
    """Read-only check — no HITL required."""
    ...

# Dangerous tool — requires user approval before the agent runs
@tool("deploy_agent", requires_approval=True)
async def trigger_deployment(service: str, version: str) -> Dict[str, Any]:
    """Deploys to production — user must approve first."""
    ...

# Another dangerous tool in the same agent
@tool("deploy_agent", requires_approval=True)
async def rollback_deployment(service: str) -> Dict[str, Any]:
    """Rolls back production — user must approve first."""
    ...
```

That's it. The `ToolRegistry` auto-discovers the flag at startup. When the Orchestrator sees that `deploy_agent`'s task includes `trigger_deployment` or `rollback_deployment`, it automatically triggers the HITL flow.

#### Step 2 — HITL instructions are auto-classified as **enhance** or **override** via `_effective_task()`

You do **not** need to manually check `hitl_context` in your agent. `BaseAgent` provides an async method `_effective_task(task_config)` that every agent should use when passing the task to prompts.

**How it works:**

When the user approves with additional instructions, `_effective_task()` calls a cheap LLM classifier (`gpt-4o-mini` by default) to determine the user's intent:

| Intent | Example | Behaviour |
|---|---|---|
| **enhance** | Original: "Search AI trends" → Instruction: "also include GPU growth" | Agent does **both** the original task AND the extra instruction |
| **override** | Original: "Search AI trends" → Instruction: "search about cricket instead" | Agent follows the instruction **instead of** the original task |

The classifier model is configurable via `HITL_CLASSIFIER_PROVIDER` and `HITL_CLASSIFIER_MODEL` in `.env`. Defaults: `openai` / `gpt-4o-mini`. If classification fails, it defaults to `enhance` (safe fallback — user gets both).

```python
# In agents/base_agent.py — already built in
async def _effective_task(self, task_config: AgentInput) -> str:
    # If no HITL instructions, returns task_config.task unchanged.
    # Otherwise:
    #   1. Calls _classify_hitl_intent() with a cheap LLM
    #   2. Returns a combined prompt (enhance) or replacement prompt (override)
```

**Usage in your agent — 2 lines:**

```python
# In agents/deploy_agent/agent.py
async def execute(self, task_config: AgentInput) -> AgentOutput:
    effective_task = await self._effective_task(task_config)  # ← line 1 (async!)

    prompt = self.prompts.deploy_prompt(
        task=effective_task,   # ← line 2: pass effective_task instead of task_config.task
        entities=task_config.entities,
        ...
    )

    # ... run tools, get result ...

    return AgentOutput(
        agent_id=task_config.agent_id,
        agent_name=self.agent_name,
        task_description=effective_task,  # ← IMPORTANT: use effective_task here too
        ...                               #    so the Composer knows the task was enhanced/overridden
    )
```

**What the LLM sees — enhance case:**

```
Complete BOTH the original task AND the additional user instructions below.

### Original Task
Search for AI trends in 2025

### User Enhanced Instructions
also include GPU market growth data
```

**What the LLM sees — override case:**

```
IMPORTANT — The user wants you to follow these override instructions
INSTEAD of the original task.

### User Override Instructions
search about cricket world cup instead

### Original Task (for reference only)
Search for AI trends in 2025
```

**What the LLM sees without HITL instructions:**

```
Deploy service-X version 2.3.1
```

> **Why `task_description=effective_task` in AgentOutput matters:**
> The Composer reads `task_description` from each agent's output. When it contains the HITL override marker, the Composer automatically detects it (via `_detect_hitl_overrides()`) and adds a notice to its own prompt: *"The user changed the task during approval. Present the data the agent actually returned, even if it differs from the Original User Query."*
> Without this, the Composer would see data that doesn't match the original query and say "I couldn't find that information."

#### Step 3 — Handle denial in prompts (optional)

If a downstream agent (e.g. `composer`) depends on a HITL agent that was denied, it receives `partial_data` with the denial reason in `dependency_outputs`. You can add prompt guidance:

```python
# In agents/composer/prompts.py  (or any downstream agent)
"""
If a dependency was denied by the user, explain what happened clearly:
- State which action was blocked
- Explain that it requires user approval
- Suggest the user try again via the streaming interface if they want to proceed
"""
```

#### No other changes needed

- **No orchestrator changes** — HITL pre-check is automatic for any tool with `requires_approval=True`
- **No streaming changes** — the callback + SSE events are generic
- **No route changes** — the `/hitl/respond` endpoint works for all agents
- **No DB changes** — the `hitl_requests` table is agent-agnostic
- **Adding/removing** — just toggle `requires_approval=True/False` on the `@tool` decorator

### Currently HITL-Protected Tools

| Tool | Agent | Why |
|------|-------|-----|
| `send_email` | `mail_agent` | Sends a real email via Gmail API |
| `reply_to_message` | `mail_agent` | Sends a real reply via Gmail API |
| `execute_code` | `code_agent` | Executes arbitrary Python code on the server |
| `create_repo` | `github_agent` | Creates a real repository on the user's GitHub account |
| `create_pull_request` | `github_agent` | Opens a real PR on a repository |
| `web_search` | `web_search_agent` | Performs a live web search (costs API credits) |
| `web_search_news` | `web_search_agent` | Performs a live news search (costs API credits) |
| `web_search_deep` | `web_search_agent` | Performs a deep web search (costs API credits) |

### Non-Streaming Endpoint Behavior

The `POST /query` endpoint does **not** support HITL dialogs (no SSE connection for approval UI). When a query triggers an agent with HITL tools on the non-streaming endpoint:

- The agent is **auto-skipped** with `error="hitl_skipped_non_streaming"`
- `partial_data` explains: *"This agent requires human approval. Use the streaming endpoint."*
- The Composer includes this in the final answer so the user knows why
- Other non-HITL agents in the plan still execute normally

### Production Guarantees

| Concern | Solution |
|---------|----------|
| **Server restart mid-approval** | All HITL state is in PostgreSQL. SSE connection drops → client re-submits. Orphaned `pending` rows are cleaned up on startup (`expire_stale_hitl_requests()`). |
| **Multiple workers** | Worker A handles SSE stream + DB polling. Worker B handles `/hitl/respond` POST. Both hit the same DB. No cross-worker coordination needed. |
| **Late response (after timeout)** | `/hitl/respond` returns `409 Conflict` if the request is already resolved. |
| **Timeout** | Configurable via `HITL_TIMEOUT_SECONDS` env var (default 120s). Treated as denial. |
| **Parallel agents in same stage** | Non-HITL agents execute immediately. HITL agent polls DB. `asyncio.gather` waits for all. |
| **No in-memory state** | `asyncio.Queue` is per-request (dies with the SSE connection). All real state is in the `hitl_requests` table. |

---

## Scheduled Jobs (Cron Agents)

Scheduled Jobs let users set up **cron-like automated agent pipelines** that run on a recurring schedule. Think of it as building multi-agent workflows that fire automatically — e.g., *"Every Monday at 9 AM, search for AI news and email me a summary."*

Each job is a **multi-step pipeline** where steps can depend on each other, enabling data flow between agents (step 2 reads step 1's output). After all agents complete, the **Composer Agent** synthesises a polished, citation-linked answer — the same quality as a live chat response. Jobs support natural language creation ("Parse & Preview"), manual step-by-step building, pause/resume, manual triggering, timezone-aware scheduling, and per-run history with full composed results.

### High-Level Architecture

```mermaid
flowchart TD
    subgraph UserLayer["User Layer"]
        UI["Frontend UI<br/><i>Natural Language / Manual Builder</i>"]
        API["FastAPI Router<br/><i>/api/v1/scheduled-jobs</i>"]
    end

    subgraph Storage["Storage Layer"]
        PG["PostgreSQL<br/><i>4 tables: jobs, steps,<br/>runs, step_results</i>"]
        Redis["Redis<br/><i>RedBeat schedule entries<br/>+ Pub/Sub status events</i>"]
    end

    subgraph Execution["Execution Layer"]
        Beat["Celery Beat<br/><i>RedBeat Scheduler<br/>reads dynamic cron entries</i>"]
        Worker["Celery Worker<br/><i>execute_scheduled_job task</i>"]
        Orch["Orchestrator<br/><i>execute_plan() — topological sort<br/>parallel where deps allow</i>"]
        Agents["Agent Instances<br/><i>RAG, Mail, Web Search,<br/>Code, GitHub, etc.</i>"]
        Composer["Composer Agent<br/><i>synthesise polished answer<br/>with citations</i>"]
    end

    UI -->|"POST /scheduled-jobs"| API
    API -->|"CRUD"| PG
    API -->|"sync_job_to_beat()<br/><i>TZ → UTC conversion</i>"| Redis

    Beat -->|"cron fires"| Worker
    Worker -->|"load job + steps"| PG
    Worker --> Orch
    Orch -->|"topological execution"| Agents
    Agents -->|"outputs"| Orch
    Orch -->|"agent results"| Composer
    Composer -->|"composed answer"| Worker
    Worker -->|"record results"| PG
    Worker -->|"Pub/Sub status +<br/>composed text"| Redis
    Redis -->|"SSE push"| UI

    style UI fill:#2196F3,color:#fff
    style API fill:#2196F3,color:#fff
    style PG fill:#4CAF50,color:#fff
    style Redis fill:#FF9800,color:#fff
    style Beat fill:#FF9800,color:#fff
    style Worker fill:#9C27B0,color:#fff
    style Orch fill:#9C27B0,color:#fff
    style Agents fill:#E91E63,color:#fff
    style Composer fill:#9C27B0,color:#fff
```

### End-to-End Flow — Job Creation to Execution

```mermaid
sequenceDiagram
    participant U as User (Frontend)
    participant API as FastAPI
    participant MA as MasterAgent
    participant DB as PostgreSQL
    participant RB as Redis / RedBeat
    participant Beat as Celery Beat
    participant W as Celery Worker
    participant O as Orchestrator
    participant A as Agents
    participant C as Composer Agent

    Note over U: Option A — Natural Language
    U->>API: POST /scheduled-jobs/from-prompt<br/>{ prompt: "Every Monday, search AI news<br/>and email me a summary",<br/>timezone: "Asia/Kolkata" }
    API->>MA: plan_scheduled_job(prompt)
    MA-->>API: { name, cron_expression,<br/>steps: [{web_search_agent, ...},<br/>{mail_agent, ...}] }
    API-->>U: { preview: { ... } }
    U->>U: Review preview, click "Create Job"

    Note over U: Option B — Manual Builder
    U->>API: POST /scheduled-jobs<br/>{ name, cron_expression,<br/>timezone, steps: [...] }

    API->>DB: INSERT scheduled_jobs + scheduled_job_steps
    API->>RB: sync_job_to_beat(job_id, cron, tz)<br/>_convert_cron_to_utc() → store UTC cron
    RB-->>RB: RedBeatSchedulerEntry saved
    API-->>U: { job_id, status: "active" }

    Note over Beat: Cron time arrives (UTC)
    Beat->>W: execute_scheduled_job.delay(job_id)
    W->>DB: load job + steps
    W->>DB: INSERT scheduled_job_runs (status: running)

    W->>O: execute_plan(ResolvedExecutionPlan)
    loop For each stage (topological order)
        O->>A: agent.execute(task_config)
        A-->>O: AgentOutput
        O->>O: Store in shared_state<br/>(next stage reads via dependency_outputs)
    end
    O-->>W: shared_state (all agent outputs)

    Note over W,C: Composer synthesises final answer
    W->>C: ComposerInput(agent_results, sources, memory)
    C-->>W: ComposerOutput(answer with citations)

    W->>DB: UPDATE step_results (+ composed_answer) + run status
    W->>DB: UPDATE job.last_run_at, next_run_at (timezone-aware)
    W->>RB: Pub/Sub → scheduled_job:{user_id}<br/>includes composed result_text
    RB-->>U: SSE push (run_complete event + text preview)
    U->>U: Toast notification (20s) + "View Full Results" button
    W->>W: _send_notification() (email / in_app)
```

### Job Execution Pipeline Detail

```mermaid
flowchart LR
    subgraph Input["Job Input"]
        Job["ScheduledJob<br/><i>cron: 0 9 * * 1</i>"]
        S1["Step 0<br/><b>web_search_agent</b><br/><i>'Search AI news'</i>"]
        S2["Step 1<br/><b>mail_agent</b><br/><i>'Email summary'</i><br/>depends_on: [0]"]
    end

    subgraph Convert["Plan Conversion"]
        RT["ResolvedAgentTask[]<br/><i>agent_ids assigned<br/>index deps → id deps</i>"]
        Plan["ResolvedExecutionPlan"]
    end

    subgraph Execute["Orchestrator — execute_plan()"]
        Topo["Topological Sort<br/><i>(Kahn's algorithm)</i>"]
        Stage1["Stage 1<br/>web_search_agent"]
        Stage2["Stage 2<br/>mail_agent<br/><i>reads web_search output<br/>via dependency_outputs</i>"]
    end

    subgraph Compose["Composer"]
        CI["ComposerInput<br/><i>agent_results, sources,<br/>memory</i>"]
        CA["Composer Agent<br/><i>LLM synthesises answer<br/>with [1], [2] citations</i>"]
    end

    subgraph Record["Results"]
        SR1["StepResult 0<br/>status: success"]
        SR2["StepResult 1<br/>status: success<br/><i>+ composed_answer</i>"]
        Run["Run<br/>status: success<br/>2/2 steps"]
    end

    Job --> S1 & S2
    S1 & S2 --> RT --> Plan
    Plan --> Topo
    Topo --> Stage1 --> Stage2
    Stage1 & Stage2 --> CI --> CA
    Stage1 --> SR1
    Stage2 --> SR2
    CA --> SR2
    SR1 & SR2 --> Run

    style Job fill:#2196F3,color:#fff
    style Stage1 fill:#FF9800,color:#fff
    style Stage2 fill:#FF9800,color:#fff
    style CA fill:#9C27B0,color:#fff
    style Run fill:#4CAF50,color:#fff
```

### Database Schema

```mermaid
erDiagram
    users ||--o{ scheduled_jobs : "owns"
    scheduled_jobs ||--o{ scheduled_job_steps : "has"
    scheduled_jobs ||--o{ scheduled_job_runs : "has"
    scheduled_job_runs ||--o{ scheduled_job_step_results : "has"
    scheduled_job_steps ||--o{ scheduled_job_step_results : "references"

    scheduled_jobs {
        uuid job_id PK
        uuid user_id FK
        varchar name
        text description
        varchar cron_expression
        varchar timezone
        varchar status "active | paused | deleted"
        varchar notification_mode "in_app | email | none"
        text notification_target
        int max_retries
        jsonb metadata
        timestamptz created_at
        timestamptz updated_at
        timestamptz last_run_at
        timestamptz next_run_at
    }

    scheduled_job_steps {
        uuid step_id PK
        uuid job_id FK
        int step_order
        varchar agent_name
        text task
        jsonb entities
        text_arr tools
        int_arr depends_on_steps
        int timeout
        int max_retries
        varchar priority
        jsonb config
        timestamptz created_at
    }

    scheduled_job_runs {
        uuid run_id PK
        uuid job_id FK
        varchar status "pending | running | success | partial_failure | failed"
        varchar trigger_type "scheduled | manual"
        timestamptz started_at
        timestamptz completed_at
        text error_summary
        int total_steps
        int completed_steps
        int failed_steps
        boolean notification_sent
        jsonb metadata
        timestamptz created_at
    }

    scheduled_job_step_results {
        uuid result_id PK
        uuid run_id FK
        uuid step_id FK
        int step_order
        varchar agent_name
        varchar status "pending | running | success | failed | skipped"
        jsonb agent_output
        text error_message
        timestamptz started_at
        timestamptz completed_at
        jsonb resource_usage
    }
```

### SQL Tables

```sql
CREATE TABLE IF NOT EXISTS scheduled_jobs (
    job_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name              VARCHAR(256) NOT NULL,
    description       TEXT NOT NULL DEFAULT '',
    cron_expression   VARCHAR(128) NOT NULL,
    timezone          VARCHAR(64)  NOT NULL DEFAULT 'UTC',
    status            VARCHAR(16)  NOT NULL DEFAULT 'active',
    notification_mode VARCHAR(32)  NOT NULL DEFAULT 'in_app',
    notification_target TEXT,
    max_retries       INT NOT NULL DEFAULT 2,
    metadata          JSONB DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_run_at       TIMESTAMPTZ,
    next_run_at       TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS scheduled_job_steps (
    step_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id            UUID NOT NULL REFERENCES scheduled_jobs(job_id) ON DELETE CASCADE,
    step_order        INT NOT NULL,
    agent_name        VARCHAR(64) NOT NULL,
    task              TEXT NOT NULL,
    entities          JSONB DEFAULT '{}',
    tools             TEXT[] DEFAULT '{}',
    depends_on_steps  INT[] DEFAULT '{}',
    timeout           INT NOT NULL DEFAULT 60,
    max_retries       INT NOT NULL DEFAULT 2,
    priority          VARCHAR(16) NOT NULL DEFAULT 'critical',
    config            JSONB DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scheduled_job_runs (
    run_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id            UUID NOT NULL REFERENCES scheduled_jobs(job_id) ON DELETE CASCADE,
    status            VARCHAR(16) NOT NULL DEFAULT 'pending',
    trigger_type      VARCHAR(16) NOT NULL DEFAULT 'scheduled',
    started_at        TIMESTAMPTZ,
    completed_at      TIMESTAMPTZ,
    error_summary     TEXT,
    total_steps       INT NOT NULL DEFAULT 0,
    completed_steps   INT NOT NULL DEFAULT 0,
    failed_steps      INT NOT NULL DEFAULT 0,
    notification_sent BOOLEAN NOT NULL DEFAULT FALSE,
    metadata          JSONB DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scheduled_job_step_results (
    result_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id            UUID NOT NULL REFERENCES scheduled_job_runs(run_id) ON DELETE CASCADE,
    step_id           UUID NOT NULL REFERENCES scheduled_job_steps(step_id) ON DELETE CASCADE,
    step_order        INT NOT NULL,
    agent_name        VARCHAR(64) NOT NULL,
    status            VARCHAR(16) NOT NULL DEFAULT 'pending',
    agent_output      JSONB,
    error_message     TEXT,
    started_at        TIMESTAMPTZ,
    completed_at      TIMESTAMPTZ,
    resource_usage    JSONB DEFAULT '{}'
);
```

### Job Status Lifecycle

```mermaid
stateDiagram-v2
    [*] --> active : Job created
    active --> paused : POST /pause
    paused --> active : POST /resume
    active --> deleted : DELETE
    paused --> deleted : DELETE

    state active {
        [*] --> Idle
        Idle --> Running : Cron fires / Manual trigger
        Running --> Idle : Run completes
    }
```

### Run Status Flow

```mermaid
stateDiagram-v2
    [*] --> pending : Run created
    pending --> running : Worker picks up
    running --> success : All steps pass
    running --> partial_failure : Some steps fail
    running --> failed : All steps fail / exception
```

### Cron Presets

| Preset | Cron Expression | Schedule |
|--------|----------------|----------|
| `every_hour` | `0 * * * *` | Top of every hour |
| `every_morning` | `0 9 * * *` | Daily at 9:00 AM |
| `every_evening` | `0 18 * * *` | Daily at 6:00 PM |
| `every_monday` | `0 9 * * 1` | Every Monday at 9:00 AM |
| `every_weekday` | `0 9 * * 1-5` | Weekdays at 9:00 AM |
| `twice_daily` | `0 9,18 * * *` | Daily at 9 AM and 6 PM |
| `weekly_friday` | `0 17 * * 5` | Every Friday at 5:00 PM |
| `monthly_first` | `0 9 1 * *` | 1st of each month at 9:00 AM |

Custom cron expressions are also supported (5-field: `minute hour day_of_month month day_of_week`).

### Natural Language Job Creation

The `POST /from-prompt` endpoint uses the **MasterAgent** with the full agent registry to parse natural language into a structured job definition:

```
User: "Every weekday morning, search for the latest AI research papers and email me a summary"

Parsed Result:
  name: "Daily AI Research Digest"
  cron_expression: "0 9 * * 1-5"
  steps:
    Step 0 — web_search_agent: "Search for the latest AI research papers published today"
    Step 1 — mail_agent: "Compose and send an email summarizing the AI research findings"
             depends_on: [0]
```

The LLM knows all available agents and their capabilities (from `agent_registry.yaml`), so it maps tasks to the correct agents and sets up dependency chains automatically.

### Notification Modes

| Mode | Behavior |
|------|----------|
| `in_app` | Redis Pub/Sub → SSE push to frontend. Shows as a toast notification in the UI. |
| `email` | Sends a summary email via the user's connected Gmail account (OAuth). Falls back silently if Gmail is not connected. |
| `none` | No notification. Results are available in run history. |

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Reuses `Orchestrator.execute_plan()`** | Steps are converted to `ResolvedAgentTask` — the same schema the live query pipeline uses. No duplicate orchestration logic. |
| **Full pipeline: Agents → Composer** | After all agents complete, the Composer Agent synthesises a polished, citation-linked answer — identical quality to live chat. The composed answer is stored in `agent_output.composed_answer` and sent via SSE for toast display. |
| **Timezone-aware scheduling** | Frontend sends browser timezone (e.g., `Asia/Kolkata`). `_convert_cron_to_utc()` converts cron hour/minute to UTC before saving to RedBeat. `next_run_at` is calculated using the job's timezone so the UI shows correct local times. |
| **RedBeat (Redis-backed dynamic scheduler)** | Add/remove/update cron entries at runtime without restarting Celery Beat. Each entry is a Redis key. `beat_max_loop_interval=10` ensures new entries are picked up within seconds. |
| **Auto-approve HITL** | Scheduled jobs run unattended, so HITL tools (email send, code execute) are auto-approved. The user pre-approves when creating the job. |
| **Soft-delete jobs** | `DELETE` sets `status = "deleted"` instead of hard-delete. Preserves run history for audit. |
| **`asyncio.run()` bridge** | Celery tasks are sync; the orchestrator is async. Same pattern as `document_tasks.py`. |
| **Step dependency via `depends_on_steps`** | Step indices map to agent IDs at runtime. Topological sort ensures correct execution order with parallel stages where possible. |
| **Separate run + step_result tables** | Each run is a snapshot. Step results track per-agent success/failure, output, timing, and resource usage for debugging. |
| **SSE + Toast with View Results** | `run_complete` event includes first 2000 chars of composed text. Toast stays 20s with a "View Full Results" button that opens a modal fetching `GET /{job_id}/runs/{run_id}` for the full composed answer + raw agent data. |

### File Map

| File | Purpose |
|------|---------|
| `database/schema.sql` | 4 new tables (`scheduled_jobs`, `scheduled_job_steps`, `scheduled_job_runs`, `scheduled_job_step_results`) |
| `database/models.py` | 4 SQLAlchemy ORM models (`ScheduledJob`, `ScheduledJobStep`, `ScheduledJobRun`, `ScheduledJobStepResult`) |
| `utils/schemas.py` | Pydantic schemas (`ScheduledJobCreate`, `ScheduledJobNLCreate`, `ScheduledJobUpdate`, `ScheduledJobStepCreate`, `CRON_PRESETS`) |
| `database/helpers.py` | CRUD helpers: `create_scheduled_job`, `get_scheduled_jobs_for_user`, `get_scheduled_job`, `update_scheduled_job`, `delete_scheduled_job`, `create_scheduled_job_run`, `get_scheduled_job_runs`, `get_scheduled_job_run_detail`, `load_scheduled_job_with_steps` |
| `tasks/scheduled_job_tasks.py` | Celery task `execute_scheduled_job` + async engine that loads job → builds `ResolvedExecutionPlan` → runs `Orchestrator.execute_plan()` → runs **Composer Agent** → records results → notifies |
| `tasks/schedule_sync.py` | RedBeat sync: `sync_job_to_beat()` with `_convert_cron_to_utc()` timezone conversion, `remove_job_from_beat()` deletes entries |
| `celery_app.py` | Beat scheduler config (`redbeat.RedBeatScheduler`) + task include |
| `api/scheduled_jobs.py` | FastAPI router — 12 endpoints for CRUD, pause/resume, trigger, run history |
| `core/master_agent.py` | `plan_scheduled_job()` — LLM-powered NL → structured job parser |
| `frontend/index.html` | Scheduled Jobs panel (icon rail, job cards, creation modal with NL + manual tabs, run history modal, **View Results** modal with composed answer, toast notifications with 20s duration + View Full Results button) |
| `core/composer_agent.py` | Composer Agent — called after orchestration to synthesise polished answer from agent outputs (same as live chat) |
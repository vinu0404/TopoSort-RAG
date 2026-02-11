# Multi-Agentic RAG System

A multi-agent Retrieval-Augmented Generation system built with FastAPI, Qdrant, PostgreSQL, and LLM providers (OpenAI / Anthropic). Agents are orchestrated via Kahn's topological sort, enabling parallel execution where dependencies allow and can handle any type of order execution of agents.Dynamic routing  of agents acording to query need.

### Dynamic routing of agents with TopoSort without using any frameworks like LangChain or LangGraph

### Frontend:
- http://127.0.0.1:8000/
![Login](images/frontend_login.png)
![Register](images/frontend_register.png)
![Chat](images/frontend-chat.png)


- http://localhost:8080/dashboard.html  
```
run frontend/dashboard.py then open dashboard.html
```
![Database Dashboard](images/dashboard.png)


### Backend:
- http://localhost:8000/docs#/
![Backend Screenshot](images/backend.png)

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

## Memory Extraction Pipeline

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

When a user uploads a document, it goes through this pipeline before it's searchable:

```mermaid
flowchart TD
    Upload["User uploads file<br/><i>PDF / DOCX / Excel / TXT</i>"]

    Upload --> Parser["Parser<br/><i>Extract raw text</i>"]

    Parser -->|"PDF"| PyMuPDF["PyMuPDF<br/>page.get_text()"]
    Parser -->|"DOCX"| PythonDocx["python-docx<br/>paragraphs"]
    Parser -->|"Excel/CSV"| Pandas["pandas<br/>df.to_string()"]
    Parser -->|"TXT/MD"| RawText["Read bytes<br/>UTF-8 decode"]

    PyMuPDF --> RawContent["Raw text content<br/>+ metadata"]
    PythonDocx --> RawContent
    Pandas --> RawContent
    RawText --> RawContent

    RawContent --> Describe["LLM generates<br/>document description<br/><i>2-3 sentences covering<br/>type, topics, scope</i>"]

    RawContent --> Chunker

    subgraph Chunker["Structure-Aware Chunker"]
        direction TB
        LLMParse["Regex based analyses structure<br/><i>identify sections, tables,<br/>headings, hierarchy</i>"]
        LLMParse --> SplitSections["Split into sections"]
        SplitSections --> TokenSplit["Token-based splitting<br/><i>1024 tokens per chunk<br/>respects paragraph boundaries</i>"]
        SplitSections --> TableChunk["Table chunks<br/><i>kept intact</i>"]
        TokenSplit --> UUIDs["Assign UUID<br/>to each chunk"]
        TableChunk --> UUIDs
        UUIDs --> Relationships["Build parent-child<br/>relationships<br/><i>first chunk = parent<br/>of its section</i>"]
    end

    Describe --> Store
    Relationships --> Embedder

    Embedder["Embedding Model<br/><i>text-embedding-3-small<br/>1536 dimensions</i>"]

    Embedder --> Store

    Store["Qdrant Vector Store"]

    subgraph Store["Store in Qdrant"]
        direction TB
        Collection["Create / get collection<br/><i>user_{id}_documents</i>"]
        Collection --> DocPoint["Upsert document-level point<br/><i>UUID, description embedding,<br/>filename, doc_type, total_chunks</i>"]
        DocPoint --> ChunkPoints["Upsert chunk points<br/><i>UUID, chunk embedding,<br/>text, metadata, parent_id</i>"]
        ChunkPoints --> PayloadIdx["Create payload indexes<br/><i>doc_type, date, section_title</i>"]
    end

    style Upload fill:#2196F3,color:#fff
    style Describe fill:#FF9800,color:#fff
    style LLMParse fill:#FF9800,color:#fff
    style Embedder fill:#9C27B0,color:#fff
```

---

## RAG Query Pipeline — How Retrieval Works

When the RAG Agent receives a query, this is the retrieval + synthesis flow:

```mermaid
flowchart TD
    Query["User query arrives<br/><i>via Master → Orchestrator</i>"]

    Query --> Entities["Extract entities<br/><i>date_range, doc_type,<br/>metric, names</i>"]
    Entities --> Filters["Build Qdrant filters<br/><i>metadata.date, metadata.doc_type,<br/>metadata.topic</i>"]

    Filters --> Hybrid

    subgraph Hybrid["Hybrid Search"]
        direction TB
        Dense["Dense Vector Search<br/><i>embed query →<br/>cosine similarity in Qdrant</i>"]
        Sparse["BM25 Sparse Search<br/><i>keyword matching</i>"]
        Dense --> RRF["Reciprocal Rank Fusion<br/><i>score = Σ 1/(k + rank)<br/>merge + deduplicate</i>"]
        Sparse --> RRF
    end

    RRF --> Top20["Top 20 candidate chunks"]

    Top20 --> Rerank

    subgraph Rerank["LLM Reranking"]
        direction TB
        RePrompt["Prompt LLM with query<br/>+ all 20 chunk previews"]
        RePrompt --> Score["LLM ranks by:<br/>• Direct relevance<br/>• Information quality<br/>• Context completeness"]
        Score --> Top8["Return top 8<br/>ranked_indices"]
    end

    Top8 --> Sources["Extract unique sources<br/><i>filename, page, section</i>"]
    Top8 --> Confidence["Calculate confidence<br/><i>avg chunk scores</i>"]

    Sources --> Output["AgentOutput<br/><i>chunks, sources, query<br/>confidence_score</i>"]
    Confidence --> Output

    Output -->|"passed to"| Composer["Composer Agent<br/><i>synthesise answer<br/>with [1], [2] citations</i>"]

    style Query fill:#2196F3,color:#fff
    style Dense fill:#4CAF50,color:#fff
    style Sparse fill:#8BC34A,color:#fff
    style RRF fill:#FF9800,color:#fff
    style RePrompt fill:#FF9800,color:#fff
    style Composer fill:#9C27B0,color:#fff
```

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
| **`users`** | Registered users | `user_id` (UUID PK), `email`, `display_name`, `created_at`,`password`,`username` |
| **`sessions`** | Login/chat sessions | `session_id` (UUID PK), `user_id` (FK), `started_at`, `is_active` |
| **`conversations`** | Conversation threads within a session | `conversation_id` (UUID PK), `session_id` (FK), `user_id` (FK), `title` |
| **`messages`** | Individual messages in a conversation | `message_id` (UUID PK), `conversation_id` (FK), `role` (`user`/`assistant`/`system`), `content`, `metadata` (JSONB) |
| **`agent_executions`** | Audit log of every agent run | `execution_id` (UUID PK), `conversation_id` (FK), `agent_name`, `status` (`pending`/`running`/`success`/`failed`), `input_payload` (JSONB), `output_payload` (JSONB), `error_message` |
| **`documents`** | Uploaded document metadata | `doc_id` (UUID PK), `user_id` (FK), `filename`, `doc_type`, `total_chunks`, `qdrant_collection` |
| **`conversation_summaries`** | Rolling summaries of every 3 conversation turns | `summary_id` (UUID PK), `conversation_id` (FK), `summary_text`, `turns_covered` |
| **`user_long_term_memory`** | Persistent user profile extracted by the Memory Extractor | `user_id` (UUID PK), `critical_facts` (JSONB), `preferences` (JSONB), `updated_at` |
| **`hitl_requests`** | HITL approval requests for agents with dangerous tools | `request_id` (UUID PK), `conversation_id` (FK), `agent_id`, `agent_name`, `tool_names` (TEXT[]), `status` (`pending`/`approved`/`denied`/`timed_out`/`expired`), `user_instructions`, `expires_at` |

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
| **`document_entry`** | `doc_id`, `filename`, `description`, `doc_type`, `uploaded_at`, `total_chunks` | Document-level vector for broad retrieval |
| **`chunk`** | `text`, `metadata` (`filename`, `page`, `section_title`, `doc_type`, `date`, …) | Individual text chunk for fine-grained search |

**Vector config:** `text-embedding-3-small` (1536 dimensions), cosine distance.

**Payload indexes:** `metadata.doc_type` (keyword), `metadata.date` (keyword), `metadata.section_title` (text).

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
| `metadata` | `Dict` | `{user_id, query_id, session_id}` |

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

## How to Add a New Agent

Adding a new agent is a **7-step process** touching 6 files (3 new, 3 existing). Below is a complete walkthrough using a fictional **`summary_agent`** as the example.

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1 │ config/agent_registry.yaml  — register capabilities      │
│  Step 2 │ config/settings.py          — add model/temp settings    │
│  Step 3 │ .env                        — set model env vars         │
│  Step 4 │ tools/summary_tools.py      — implement tool functions   │
│  Step 5 │ agents/summary_agent/       — prompts.py + agent.py      │
│  Step 6 │ api/routes.py & streaming.py — inject agent instance     │
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

```python
# agents/summary_agent/prompts.py

"""Summary agent prompts — production-quality, personalised."""

from __future__ import annotations
from typing import Any, Dict

from utils.prompt_utils import format_user_profile


class SummaryPrompts:

    @staticmethod
    def summarise_prompt(
        task: str,
        source_text: str,
        entities: Dict[str, Any],
        dependency_outputs: Dict[str, Any] | None = None,
        long_term_memory: Dict[str, Any] | None = None,
    ) -> str:
        dep_context = ""
        if dependency_outputs:
            dep_context = "\n### Context from other agents\n"
            for aid, data in dependency_outputs.items():
                dep_context += f"- **{aid}**: {str(data)[:500]}\n"

        entity_str = (
            ", ".join(f"{k}={v}" for k, v in entities.items())
            if entities else "none"
        )

        profile = format_user_profile(long_term_memory or {})

        return f"""You are a Summarisation Expert in a multi-agent RAG system.

### Task
{task}

### Entities
{entity_str}
{dep_context}
{profile}
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
            summarise = self.get_tool("summarise_text")

            # Build the source text from dependency outputs or the task itself
            source_text = ""
            if task_config.dependency_outputs:
                for dep_data in task_config.dependency_outputs.values():
                    if isinstance(dep_data, dict):
                        source_text += str(dep_data.get("chunks", dep_data)) + "\n"
                    else:
                        source_text += str(dep_data) + "\n"
            else:
                source_text = task_config.task

            # Ask LLM to summarise
            prompt = self.prompts.summarise_prompt(
                task=task_config.task,
                source_text=source_text,
                entities=task_config.entities,
                dependency_outputs=task_config.dependency_outputs,
                long_term_memory=task_config.long_term_memory,     # ← always pass memory
            )
            summary: str = await self.llm.generate(
                prompt=prompt,
                temperature=config.summary_temperature,
                model=config.summary_model,
            )

            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=task_config.task,
                task_done=True,
                result=summary,
                data={"summary": summary},
                confidence_score=0.9,
                resource_usage={
                    "time_taken_ms": int((time.perf_counter() - start) * 1000),
                },
                depends_on=list(task_config.dependency_outputs.keys()),
            )

        except Exception:
            raise   # let execute_with_retry handle retries
```

---

### Step 6 — Wire the Agent Instance

In both `api/routes.py` and `api/streaming.py`, add the agent to the `agent_instances` dict passed to the `Orchestrator`:

```python
from agents.summary_agent.agent import SummaryAgent

# When building agent_instances (before creating Orchestrator):
summary_llm = get_llm_provider(config.summary_model_provider, default_model=config.summary_model)
agent_instances["summary_agent"] = SummaryAgent(tool_registry, summary_llm)

orchestrator = Orchestrator(agent_instances=agent_instances)
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
    mock_llm.generate.return_value = "This is a concise summary."

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
| 5 | `agents/<name>/__init__.py` | Empty package file |
| 5 | `agents/<name>/prompts.py` | Prompt class using `format_user_profile()` from `utils/prompt_utils.py` |
| 5 | `agents/<name>/agent.py` | Subclass `BaseAgent`, implement `execute()` + `get_required_tools()` |
| 6 | `api/routes.py` + `api/streaming.py` | Import agent, create instance, add to `agent_instances` dict |
| 7 | `tests/test_<name>_agent.py` | Unit test for the new agent |

### Architecture Rules

- **`agent_name` must be consistent** across: `agent_registry.yaml` key, `BaseAgent.__init__("name")`, `@tool("name")`, and `agent_instances["name"]`.
- **Tools are auto-discovered** — any `tools/*_tools.py` file is scanned at startup. No manual registration needed.
- **User profile formatting** lives in `utils/prompt_utils.py` → `format_user_profile()`. Import it in your prompts — never duplicate it.
- **Long-term memory** is automatically loaded and injected by the Orchestrator into every `AgentInput.long_term_memory`. Pass it to your prompt methods.
- **Retry/timeout** is handled by `BaseAgent.execute_with_retry()`. Your `execute()` should raise on failure, not catch-and-suppress.
- **Dependencies** are declared by the Master Agent in the execution plan (`depends_on`). The topological sort ensures your agent only runs after its dependencies complete.

---

## Adding HITL (Human-in-the-Loop) to an Agent

Some tools perform **irreversible actions** — sending emails, executing code, making API calls. The HITL system lets you gate these tools behind a **user approval step** before the agent runs. All state is persisted in PostgreSQL — no in-memory coordination, survives server restarts, works across multiple workers.

### How HITL Works — Full Flow

```mermaid
sequenceDiagram
    participant User as 👤 User (Frontend)
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

    loop Poll DB every 1.5s
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
    participant User as 👤 User
    participant SSE as SSE Stream
    participant Orch as Orchestrator
    participant DB as PostgreSQL

    Orch->>DB: INSERT hitl_requests (pending)
    SSE-->>User: event: hitl_required

    User->>DB: POST /hitl/respond<br/>{decision: "denied"}
    Note over DB: status = 'denied'

    Orch->>DB: Poll → status='denied'
    SSE-->>User: event: hitl_denied

    Note over Orch: Create synthetic AgentOutput:<br/>task_done=False<br/>error="denied_by_user"<br/>partial_data={reason, tools}

    Note over Orch: Downstream agents see<br/>dependency_outputs[mail_agent_0] =<br/>{reason: "denied_by_user"}<br/>→ Composer explains what happened
```

### What Happens on Timeout

```mermaid
sequenceDiagram
    participant SSE as SSE Stream
    participant Orch as Orchestrator
    participant DB as PostgreSQL

    Orch->>DB: INSERT hitl_requests (pending, expires_at=+120s)
    SSE-->>SSE: event: hitl_required

    loop 120 seconds of polling
        Orch->>DB: SELECT status → 'pending'
    end

    Note over DB: expires_at reached

    Orch->>DB: UPDATE status='timed_out'
    SSE-->>SSE: event: hitl_timeout

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
    CB -->|"poll every 1.5s"| DB
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
| `hitl_approved` | User approved the request | `{request_id, agent_id}` |
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
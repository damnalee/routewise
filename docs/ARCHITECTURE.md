# RouteWise architecture

Diagrams render on GitHub. Source code lives in `src/routewise/`.

## 1. Components

```mermaid
flowchart TB
    subgraph HTTP["HTTP layer, app.py"]
        API["FastAPI endpoints<br/>auth, request id, error mapping"]
    end
    subgraph CORE["Pipeline, gateway.py"]
        GW["Gateway.answer"]
    end
    subgraph CACHE["Caching"]
        EMB["Embedder<br/>sentence-transformers, CPU, locked"]
        SC["SemanticCache<br/>TTL, LRU, namespaces"]
        IDX["VectorIndex<br/>exact cosine search"]
        QG["query_guard<br/>numbers, negation, word order"]
    end
    subgraph ROUTE["Routing"]
        CX["HeuristicScorer"]
        RT["Router<br/>threshold plus cost pressure"]
    end
    subgraph LLMS["Providers"]
        GQ["GroqClient<br/>key pool, retries"]
        OL["OllamaClient"]
    end
    subgraph OBS["Observability"]
        TR["CostTracker<br/>SQLite"]
        PM["Prometheus metrics"]
        DASH["Streamlit dashboard"]
    end
    API --> GW
    GW --> EMB
    GW --> SC
    SC --> IDX
    SC --> QG
    GW --> RT
    RT --> CX
    GW --> GQ
    GW -.-> OL
    GW --> TR
    GW --> PM
    TR --> DASH
    TR -- cost pressure --> RT
```

## 2. One request, end to end

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant G as Gateway
    participant E as Embedder
    participant K as SemanticCache
    participant R as Router
    participant L as LLM provider
    participant T as CostTracker
    C->>G: POST /v1/query
    G->>E: embed query
    E-->>G: vector
    G->>K: lookup vector, namespace, query
    alt cache hit
        K-->>G: cached answer
        G->>T: record cache_hit
        G-->>C: answer, source cache, cost 0
    else cache miss
        G->>T: read cost pressure
        T-->>G: value from 0 to 1
        G->>R: route query, context size, pressure
        R-->>G: tier and model
        G->>L: generate with chosen model
        alt large model fails with a transient error
            L--xG: error
            G->>L: generate with small model
        end
        L-->>G: text and token counts
        G->>K: store answer unless degraded
        G->>T: record outcome, tokens, cost, latency
        G-->>C: answer, source, cost, saved
    end
```

## 3. Cache lookup

```mermaid
flowchart TD
    A["Query vector and namespace"] --> B{"Namespace has entries?"}
    B -- no --> MISS["Miss"]
    B -- yes --> C["Top 5 candidates with cosine >= threshold"]
    C --> D{"Any candidate left?"}
    D -- no --> MISS
    D -- yes --> E{"Expired?"}
    E -- yes --> F["Evict entry"] --> D
    E -- no --> G{"Lexical guard passes?<br/>same numbers, same negations,<br/>same word order"}
    G -- no --> H["Count a rejection"] --> D
    G -- yes --> HIT["Hit: return cached answer"]
```

## 4. Routing decision

```mermaid
flowchart TD
    Q["Query and context size"] --> S["Complexity score 0 to 1<br/>length, code, reasoning, math,<br/>multi-part, context, simple lookup"]
    Q --> P["Cost pressure<br/>spend in window / budget"]
    P --> TH["Effective threshold<br/>base + 0.20 x pressure"]
    S --> D{"force_tier set?"}
    TH --> D
    D -- yes --> FT["Use the forced tier"]
    D -- no --> X{"score >= threshold?"}
    X -- yes --> LG["Large model"]
    X -- no --> SM["Small model"]
```

## 5. Failure handling

```mermaid
flowchart TD
    A["Large model call"] --> B{"Result"}
    B -- success --> OK["Answer, cached"]
    B -- transient error --> C{"Tier was forced?"}
    B -- other error --> ERR["Return HTTP error"]
    C -- yes --> ERR
    C -- no --> D["Call small model"]
    D --> E{"Result"}
    E -- success --> DEG["Answer marked degraded, not cached"]
    E -- error --> ERR
```

Provider errors map to clean HTTP responses and never leak provider details:

| Provider error | HTTP status | Code |
|---|---|---|
| Rate limited, all keys cooling | 429 with `Retry-After: 5` | `upstream_rate_limited` |
| Timeout | 504 | `upstream_timeout` |
| Unavailable or 5xx | 503 | `upstream_unavailable` |
| Auth, bad response, model not found | 502 | `upstream_error` |

Groq API key life cycle, per key:

```mermaid
stateDiagram-v2
    [*] --> Ready
    Ready --> Cooling: HTTP 429 with Retry-After
    Cooling --> Ready: cooldown elapsed
    Ready --> Disabled: HTTP 401 or 403
    Disabled --> [*]
```

## 6. Data model

One SQLite table, written once per request.

```mermaid
erDiagram
    REQUESTS {
        integer id PK
        real ts
        text namespace
        text query_preview
        text outcome "cache_hit, small, large, error"
        text model
        real score
        integer tokens_in
        integer tokens_out
        real cost_usd
        real baseline_usd
        real latency_s
        text error
    }
```

## How savings are computed

- `baseline_usd`: what the same request would have cost on the large model.
- `cost_usd`: what it actually cost. Zero for cache hits and errors.
- `saved = baseline - cost`. A cache hit counts the full large-model cost as saved, using estimated tokens (flagged `tokens_estimated`).
- Prices are approximate reference values from config. Verify them at console.groq.com.

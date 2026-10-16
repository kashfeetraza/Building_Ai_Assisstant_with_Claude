---
name: fullstack-ai-assistant
description: Build a production-ready full-stack AI chat assistant (ChatGPT/Claude-style) with a FastAPI backend (Python 3.13, uv, ruff, pytest) using LangChain for AI orchestration, local Ollama models first, and cloud AI fallbacks (Anthropic, OpenAI, Gemini, Grok, Meta Llama). Includes a Next.js 16 / React 19 / TypeScript 7 / Tailwind 4 / shadcn/ui frontend with streaming chat, a local-vs-cloud model dropdown, history, and light/dark themes. Use whenever the user asks to create, scaffold, extend, rebrand, deploy, or debug an AI chatbot / AI assistant web app, a local-LLM chat UI, a LangChain + Ollama + cloud fallback backend, or "the full stack AI app" built with this system, including adding providers, auth, database history, or file uploads to it.
---

# Full-stack AI Assistant — LangChain Edition

A production-oriented, tested full-stack AI assistant template with a LangChain-powered AI layer.

Keep the existing project architecture, frontend, API contract, UI, design system, scripts, testing standards, deployment workflow, and delivery requirements intact. The custom AI model invocation layer is replaced by LangChain integrations while the application's existing behaviour is preserved.

## Core technology stack

- Backend: FastAPI, Python 3.13, uv, Ruff, pytest.
- AI framework: LangChain, LangChain Core, provider-specific LangChain integrations.
- Local AI: Ollama through `langchain-ollama` and `ChatOllama`.
- Cloud AI: OpenAI, Anthropic, Google Gemini, xAI Grok, and compatible Meta Llama endpoints.
- Frontend: Next.js 16, React 19, TypeScript 7, Tailwind CSS 4, shadcn/ui (Radix).
- Communication: HTTP APIs and Server-Sent Events (SSE).
- Deployment: Docker Compose, with optional Ollama service/profile.

Do not downgrade requested major versions. Refresh stable dependency versions when necessary and verify compatibility before upgrading.

## 1. Core behaviour

The application must retain these behaviours after the LangChain migration.

### Model selection

The model dropdown offers:

- `Auto`
- Every available Ollama chat model discovered from the running local Ollama server.
- Models exposed by each configured cloud provider whose credentials and provider configuration are valid.

Filter out embedding-only models from the chat model list.

The UI must distinguish local and cloud models and display the selected model clearly.

### Auto selection

`Auto` prefers a usable local Ollama chat model first, followed by configured cloud providers in `CLOUD_PRIORITY` order.

The order must be configurable.

If no local model is available, try configured cloud providers. If no usable provider exists, return a short, actionable error without leaking credentials or internal stack traces.

### Fallback behaviour

If the selected provider fails before the first response token is sent, try the next eligible provider according to the fallback policy.

Return fallback metadata through the existing response protocol:

- `meta.fallback`
- `meta.notice`

The frontend must notify the user when a different provider or model is used.

Do not silently change the user's explicitly selected model unless the existing fallback policy permits it.

Once response content has been emitted, do not restart the answer using another model. Doing so could duplicate or contradict already delivered content.

Fallback eligibility, retry limits, timeouts, and provider ordering must be explicit and testable.

### Streaming

AI responses stream incrementally through LangChain's asynchronous streaming interface where supported.

Preserve the existing SSE event format and frontend parser. Do not change the public API contract just to introduce LangChain.

Keep streaming, fallback metadata, error events, and completion events consistent with the existing frontend.

## 2. Settle the brief

Use sensible defaults. Ask at most one short question, and only if an important choice genuinely prevents implementation.

| Input | Default |
|---|---|
| App display name | `Assistant` |
| Placeholder user name | `Rizwan` |
| Cloud providers | All five provider categories, enabled when configured |
| AI framework | LangChain |
| Local inference | Ollama |
| Additional scope | None; see `references/extending.md` |

Use the established project identity and any user-provided name. Do not invent additional personal names in UI or documentation.

If the user requests authentication, database history, uploads, RAG, or other features, consult `references/extending.md` before implementing them.

## 3. Refresh versions and dependencies

The original template was pinned and verified in September 2026. When the current date is later, check current stable versions and package compatibility before changing dependencies.

Preserve the requested major versions of Next.js, React, TypeScript, Tailwind CSS, Python, and FastAPI wherever compatible.

Use `uv` for Python dependency management. After changing `api/pyproject.toml`, regenerate `api/uv.lock` with `uv lock` (the Docker build uses `uv sync --frozen`, which trusts the lockfile).

Core LangChain packages:

- `langchain`
- `langchain-core`
- `langchain-ollama`

Add provider packages only when their integrations are required:

- `langchain-openai`
- `langchain-anthropic`
- `langchain-google-genai`

For Grok and Meta Llama, choose the appropriate supported LangChain integration or OpenAI-compatible endpoint. Verify the endpoint's API compatibility and authentication requirements instead of assuming every provider supports every OpenAI API feature.

Use the official LangChain integrations documentation to verify package names, model classes, configuration, and streaming support.

References:

- LangChain integrations: https://docs.langchain.com/oss/python/integrations/providers
- Ollama integration: https://docs.langchain.com/oss/python/integrations/providers/ollama/
- LangChain Python API reference: https://reference.langchain.com/python/integrations/overview

Do not install every integration unnecessarily. Keep the dependency set limited to supported features and configured providers.

## 4. Scaffold the project

Use the existing scaffold script and bundled template. Do not recreate the project by hand or from memory.

```bash
python <skill-dir>/scripts/scaffold.py --dest ./<folder> \
  --app-name "Acme Copilot" \
  --user-name "Rizwan" \
  --description "Acme's internal AI assistant"
```

The scaffolder must continue to:

- Copy `assets/template/`.
- Rename packages to `<slug>-api` and `<slug>-web`.
- Update package metadata and lockfiles.
- Set the UI name and placeholder user in `web/lib/config.ts`.
- Update the API title and README heading.
- Create `api/.env` and `web/.env.local` from example files.
- Refuse to overwrite an existing folder without `--force`.

If the bundled template ever lacks the LangChain implementation, update the template and scaffolding logic consistently. Do not merely document LangChain while leaving the generated backend on the old custom provider invocation architecture.

## 5. LangChain AI architecture

The AI logic uses LangChain's chat model interfaces and provider integrations.

Keep FastAPI as the HTTP application framework. LangChain is responsible for model integration, prompt construction, invocation, and streaming; it does not replace FastAPI.

### Backend structure

```text
api/
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── deps.py
│   ├── core/
│   │   └── config.py
│   ├── providers/
│   │   ├── base.py            Provider interface, LangChainProvider, prompt, safe errors
│   │   ├── ollama.py          ChatOllama + local model discovery
│   │   ├── anthropic.py       ChatAnthropic + model listing
│   │   ├── openai_compat.py   ChatOpenAI for OpenAI, Grok, Meta Llama + model listing
│   │   ├── google.py          ChatGoogleGenerativeAI for Gemini
│   │   └── registry.py        build_providers(), discovery cache, candidate order
│   ├── services/
│   │   └── chat.py            streaming, fallback, error events
│   └── routes/
│       ├── chat.py
│       ├── models.py
│       └── health.py
└── tests/
```

Keep this structure. Do not create redundant files just to match another layout; add a file only for a genuinely new responsibility.

### LangChain model interface

Use LangChain chat models rather than directly invoking each provider's SDK from chat routes.

Integrations in use:

- Ollama: `ChatOllama` from `langchain_ollama`.
- OpenAI: `ChatOpenAI` from `langchain_openai`.
- Anthropic: `ChatAnthropic` from `langchain_anthropic`.
- Gemini: `ChatGoogleGenerativeAI` from `langchain_google_genai`.
- Grok and Meta Llama: `ChatOpenAI` pointed at the provider's OpenAI-compatible endpoint, with provider-specific configuration.

Keep provider-specific SDK types inside their adapter modules. Routes and chat services work with a consistent internal model/provider interface.

Do not assume every provider supports identical parameters, structured output, tool calling, streaming metadata, or retry behaviour.

### Model factory

Each adapter implements `_build_model(model, temperature)` (declared on `LangChainProvider` in `providers/base.py`), and `build_providers()` in `providers/registry.py` is the central factory that:

1. Receives the selected model identifier (via the registry's candidate list).
2. Resolves the provider and actual model name.
3. Validates provider configuration and credentials.
4. Creates the appropriate LangChain chat model.
5. Applies supported generation settings, timeouts, and bounded retries.
6. Returns the model through the consistent `Provider.stream_chat` interface.
7. Reports safe, actionable configuration or provider errors.

Example of local model initialization:

```python
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="qwen3.5:2b",
    temperature=0.7,
)
```

Use the configured model name rather than hardcoding this example in production code.

Read settings from the application configuration. Do not access environment variables directly throughout routes and services.

### Prompt construction

`providers/base.py` builds the prompt with `ChatPromptTemplate` and a `MessagesPlaceholder`, and converts app messages to LangChain `SystemMessage` / `HumanMessage` / `AIMessage` objects. Message objects (not template strings) are passed in, so user text containing braces is never treated as a template.

Keep system instructions separate from user messages. `ChatService` adds the configured system prompt; all system messages are merged into one leading system message.

Preserve conversation roles and message ordering. Validate incoming message history and apply appropriate size limits.

Do not treat user-provided content as trusted system instructions.

Do not add a vector database, retriever, agent, or tool-calling workflow by default. Those are separate features and must only be added when requested.

### Model discovery

Keep model discovery separate from model invocation.

For Ollama:

- Query the running Ollama server for installed models.
- Filter out embedding-only models.
- Expose available chat models in the dropdown.
- Handle an unavailable Ollama server gracefully.

For cloud providers:

- Expose configured providers and models that can be verified through supported provider APIs or explicit configuration.
- Do not assume that every API key permits model-list discovery.
- Allow explicitly configured model IDs when live discovery is unavailable.
- Do not make external model-list calls on every chat request; use an appropriate cache.

LangChain standardizes model invocation, but it does not automatically provide a universal model catalogue for all providers. Keep discovery logic explicit.

## 6. Provider abstraction and fallback

The application still needs its own provider registry and fallback policy, even though model invocation uses LangChain.

Use a small internal abstraction that exposes only the application-level capabilities required by the chat service, such as:

- Provider and model identification.
- Model construction.
- Async streaming.
- Safe error classification.
- Capability metadata where needed.

Keep LangChain and provider SDK-specific types inside the adapter layer.

Do not wrap LangChain in unnecessary abstractions that duplicate its own model interface. Retain the existing registry only for application-specific discovery, routing, configuration, and fallback.

### Fallback algorithm

The chat service should:

1. Resolve the requested model or `Auto`.
2. Construct the ordered candidate list.
3. Attempt each eligible candidate within configured timeouts.
4. Emit fallback metadata if a different candidate is selected.
5. Start streaming the first successful candidate.
6. Stop trying alternatives after the first user-visible content token is emitted.
7. Report safe errors if all eligible candidates fail.

Use a bounded retry policy (`LLM_MAX_RETRIES`). Do not repeatedly retry invalid credentials, unavailable model IDs, or other clearly non-retryable failures without a reason.

Distinguish provider connection failures, authentication failures, rate limits, invalid requests, and model-not-found errors where the integration permits it (`describe_failure()` in `providers/base.py`).

Never expose API keys, raw stack traces, or sensitive internal configuration to the frontend.

## 7. FastAPI and SSE contract

Keep the public HTTP API and SSE protocol unchanged unless the user explicitly requests an API redesign.

Routes should remain thin. Business logic belongs in services and the model registry.

The chat route should:

- Validate the request.
- Read settings through application dependencies.
- Invoke the LangChain-based chat service.
- Stream events using the established SSE schema.
- Send fallback metadata and user-safe errors.
- Close the stream cleanly when complete or cancelled.

Use async streaming where the selected LangChain integration supports it.

Handle client disconnects, cancellation, timeouts, and exceptions without leaking resources.

Do not perform blocking work on the event loop unnecessarily.

Keep settings injectable through `request.app.state.settings` and dependencies. Do not read global settings directly inside route handlers.

## 8. Frontend requirements — preserve unchanged

Keep the existing frontend design and behaviour.

Technology:

- Next.js 16.
- React 19.
- TypeScript 7.
- Tailwind CSS 4.
- shadcn/ui and Radix primitives.

Preserve:

- ChatGPT/Claude-style chat interface.
- Sidebar and conversation navigation.
- Model dropdown with local/cloud labels.
- Auto model selection.
- Streaming message display.
- Markdown rendering.
- Conversation history and existing persistence behaviour.
- Light and dark themes.
- Responsive mobile layout.
- Accessible keyboard interactions.
- Existing spacing and neutral design tokens.

No gradients or decorative colour. Use the existing neutral token palette and design language.

Keep `web/lib/types.ts` in sync with `api/app/schemas.py`.

Do not expose provider API keys to the browser. Provider credentials belong on the backend.

Keep the existing SSE parser in `web/lib/api.ts` compatible with the server's event format.

Use the current frontend hooks and components instead of introducing a second state-management system without a real need.

## 9. Configuration and environment variables

Keep configuration centralized in `api/app/core/config.py`.

Support the existing application configuration and appropriate LangChain provider settings, including:

- Ollama base URL and default model.
- Provider API keys.
- Provider-specific model IDs.
- Cloud provider priority.
- Request timeouts and bounded retries (`REQUEST_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`).
- Generation settings.
- Model discovery cache settings.
- CORS and frontend API configuration.

Use example files for environment variables. Never commit real credentials.

Cloud providers must remain optional. The application should start and expose useful status information even if no cloud API keys are configured.

A missing optional provider key must not prevent the local Ollama path from working.

Do not log secrets or include them in exception messages returned to users.

## 10. Testing and quality requirements

Every new backend behaviour must have pytest coverage using fake or mocked LangChain model interfaces. Tests must not require a real cloud account or make network requests.

Retain and update existing tests for:

- Chat request validation.
- Model discovery.
- Provider registry.
- Local-first model selection.
- Configured cloud priority.
- Streaming events.
- Fallback before the first response token.
- No fallback after visible response content.
- Provider authentication and connection errors.
- Missing configuration.
- Cancellation and timeouts.
- Safe user-facing errors.
- Injected settings.
- API response compatibility.

Use deterministic fakes: `FakeChatModel` (a fake LangChain chat model) for the adapter layer and `FakeProvider` for the chat service and routes. Do not call actual Ollama, OpenAI, Anthropic, Gemini, Grok, or Meta endpoints during unit tests.

Use integration tests only when explicitly configured for a live environment.

Run the required quality checks:

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

For the frontend:

```bash
npm ci
npm run typecheck
npm run build
```

Fix every failure before delivery. Do not claim tests passed unless they were actually executed successfully.

## 11. References for customization

Read only the relevant reference files.

| Task | Reference |
|---|---|
| Understand architecture and AI flow | `references/architecture.md` |
| Change or consume HTTP/SSE API | `references/api-contract.md` |
| Add or tune LangChain provider integrations | `references/providers.md` |
| UI changes, rebranding, accessibility | `references/frontend.md` |
| Auth, database history, attachments, RAG, agents, reasoning, usage | `references/extending.md` |
| Docker, environment variables, reverse proxy, deployment | `references/deployment.md` |
| Troubleshooting failures | `references/troubleshooting.md` |

## 12. Scripts

Keep the existing scripts and their intended behaviour.

| Script | Purpose |
|---|---|
| `scripts/scaffold.py` | Generate a project from the bundled template |
| `scripts/verify.sh` | Run backend and frontend quality checks |
| `scripts/doctor.py` | Check required tools, Ollama models, API health, and web proxy |

The verification script supports:

```bash
bash <skill-dir>/scripts/verify.sh ./<folder>
bash <skill-dir>/scripts/verify.sh ./<folder> --api
```

The doctor script verifies tools such as `uv`, Node.js >= 20.9, Ollama, installed local models, the running API, and the frontend proxy.

## 13. Troubleshooting

Consult `references/troubleshooting.md` whenever something fails.

Common issues to cover:

- Ollama server is not running.
- The requested Ollama model is not installed.
- An embedding-only model was selected for chat.
- A cloud provider's API key is missing or invalid.
- A provider's model discovery endpoint is unavailable.
- A LangChain integration does not support a requested capability.
- SSE streaming is interrupted.
- A provider fails before the first token.
- Frontend and backend API schemas are inconsistent.
- Localhost IPv4/IPv6 binding causes connection errors.
- Python package resolution or Node dependency installation fails.
- Docker service URLs are incorrectly configured.

Do not invent successful workarounds. Reproduce the error, use the relevant logs, make a targeted fix, and run the appropriate checks again.

If Ollama cannot be installed in the test environment, use a controlled stand-in server or fake model for the relevant tests, and clearly report that live Ollama inference was not verified.

## 14. Live smoke test

Start the API:

```bash
cd api
uv sync
uv run fastapi dev app/main.py
```

Start the frontend in a separate terminal:

```bash
cd web
npm install
npm run dev
```

Pull a model if needed:

```bash
ollama pull llama3.2
```

Use the existing doctor script:

```bash
python <skill-dir>/scripts/doctor.py \
  --api http://localhost:8000 \
  --web http://localhost:3000
```

Verify the actual endpoints and configuration before reporting success.

When a headless browser is available and UI changes have been made, capture and inspect light mode, dark mode, and a 390 px mobile viewport.

## 15. Deployment

Preserve the existing Docker Compose setup and deployment workflow.

Support:

- FastAPI backend.
- Next.js frontend.
- Optional Ollama service/profile.
- Environment-specific API URLs.
- CORS configuration.
- Production frontend output.
- Secure environment-variable handling.

Ensure the backend uses the correct Ollama service URL when running in Docker. A container's `localhost` refers to that container, not automatically to the host or another service.

Do not commit real `.env` or `.env.local` files.

Consult `references/deployment.md` for platform-specific configuration and go-live steps.

## 16. Delivery requirements

Before delivering:

1. Exclude `node_modules`, `.next`, `.venv`, caches, and real `.env` files.
2. Run the applicable verification scripts.
3. Fix all reported failures.
4. Produce a ZIP of the clean project folder.
5. Summarize what was built or changed.
6. Report the actual test results, including test counts where available.
7. Clearly identify anything not verified, such as live cloud API keys or live Ollama inference.
8. Include the three primary run commands.

```bash
ollama pull llama3.2
cd api && uv sync && uv run fastapi dev app/main.py
cd web && npm install && npm run dev
```

Do not claim that the project is production-ready, fully tested, or successfully packaged unless those checks and operations actually succeeded.

## 17. Key implementation files

| File | Responsibility |
|---|---|
| `api/app/main.py` | Application factory, lifespan, CORS |
| `api/app/core/config.py` | Settings and environment variables |
| `api/app/providers/base.py` | `Provider` interface, `LangChainProvider`, prompt, message conversion, safe errors |
| `api/app/providers/ollama.py` | LangChain `ChatOllama` adapter and local model discovery |
| `api/app/providers/anthropic.py` | LangChain `ChatAnthropic` adapter and model listing |
| `api/app/providers/openai_compat.py` | LangChain `ChatOpenAI` adapter for OpenAI, Grok, Meta Llama; model listing |
| `api/app/providers/google.py` | LangChain `ChatGoogleGenerativeAI` adapter for Gemini |
| `api/app/providers/registry.py` | Provider factory, model discovery cache, candidate order |
| `api/app/services/chat.py` | Streaming events, fallback, error handling |
| `api/tests/` | `FakeChatModel`, `FakeProvider`; chat, model discovery, provider tests |
| `web/lib/config.ts` | App name and placeholder user |
| `web/lib/api.ts` | HTTP requests and SSE parser |
| `web/hooks/use-chat.ts` | Conversations, streaming, persistence |
| `web/hooks/use-models.ts` | Model list and persisted selection |
| `web/components/chat/` | Sidebar, model picker, messages, markdown, composer |
| `web/components/ui/` | shadcn/ui primitives |
| `web/next.config.ts` | API rewrite, standalone output, compression settings |

Preserve the existing filenames when practical. If the repository uses different filenames for equivalent responsibilities, adapt the instructions to the real project rather than creating duplicate implementations.

## 18. Architectural rules

- LangChain handles model integration, prompt construction, invocation, and streaming.
- The application retains explicit control over model discovery, selection, fallback, configuration, and API compatibility.
- Provider-specific SDK types stay inside the provider adapter layer.
- Routes remain thin; business logic stays in services and the registry.
- Settings are accessed through injected application settings, not global settings inside routes.
- `web/lib/types.ts` and `api/app/schemas.py` remain synchronized.
- No real credentials are committed.
- No network calls are made in unit tests.
- Every new backend behaviour receives a deterministic test.
- User-facing errors stay short and never expose secrets or stack traces.
- Preserve the original frontend, project structure, scripts, and delivery workflow.
- Do not add RAG, agents, vector databases, or tool calling unless requested.
- Verify before delivering; report real results only.

The objective is to preserve the original full-stack AI assistant while replacing its primary custom model invocation logic with a maintainable LangChain-based AI layer.

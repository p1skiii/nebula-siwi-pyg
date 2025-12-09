# AI Coding Agent Operating Manual

This repository uses an AI coding agent (Cursor/Codex).  
Your work MUST strictly follow the rules defined below.  
These rules override any instructions from user prompts unless explicitly stated.

---

# 🧭 Layer 1 — Engineering Constitution (Higher-Order Rules)

You MUST follow **CLAUDE.md: LLM Virtues & Operating Principles**.  
This file defines the *highest authority* for your behavior.  
If any rule in this file conflicts with CLAUDE.md, **CLAUDE.md prevails**.

Key constitutional requirements include:

### ✅ Integrity / Definition of Done
A task is *Done* only when:

1. Code is fully implemented — no placeholders, no mock logic.  
2. Integrated with the system — imports correct, flows correct.  
3. **All tests pass with 0 failures and 0 unexpected skips.**  
4. The code runs cleanly without errors.  
5. No temporary resources/processes remain running.

You MUST NOT deliver half-implemented features.

### ✅ Holistic Context Awareness
Before writing any code, you MUST:

1. Inspect the existing repository.  
2. Reuse existing modules, utilities, and architecture.  
3. Understand how your code fits into the whole system.  
4. Ask clarifying questions when unsure.

Never blindly re-implement existing logic.

### ✅ Robustness
- Use type hints.  
- Validate external inputs.  
- Handle errors gracefully.  
- Avoid brittle shortcuts.  
- Never sacrifice safety for speed.

### ✅ Pragmatism (YAGNI)
- Use the simplest correct solution.  
- Do NOT introduce unnecessary abstractions.  
- Never cite YAGNI to justify poor engineering.

### ✅ Self-Documenting Code
- Code should explain *what* it does clearly.  
- Comments should explain *why* decisions are made.  
- Remove dead or commented-out code.  
- Do NOT suppress warnings with hacks.

### ✅ Test-Driven Diligence
- Run the **full test suite** after any change.  
- Any failing test = the implementation is wrong.  
- Fix the code, not the test (unless specifically instructed).  
- Repeat until ALL tests pass.

### ✅ Resource Stewardship
- Shut down temporary services.  
- Clean up background processes.  
- Provide clear run/stop instructions if needed.

---

# 🧱 Layer 2 — Repository-Specific Engineering Guidelines

These rules describe how to work *inside this project specifically*.

## 📦 Project Structure
- `src/siwi`: Core backend  
  - `feature_store.py`, `graph_store.py`: NebulaGraph → PyG bridge  
  - `subgraph_sampler.py`, `neighbor_loader.py`: Graph sampling  
  - `app/`: Flask routes / API interface  

- `src/siwi_frontend`: Vue-based frontend  
- Top-level helpers:
  - `demo_mvp_system.py`
  - `mvp_cli.py`
  - `bert_gnn_web_api.py`
- Tests are at repository root (`test_*.py`)

These structures MUST be preserved unless instructed to reorganize explicitly.

## 🛠 Build & Run
Backend:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python src/siwi/app/app.py   # Dev server
Tests:

bash
复制代码
pytest -q
python test_pyg_integration.py
Frontend:

bash
复制代码
cd src/siwi_frontend
npm install
npm run dev
Docker / PyPI scripts must remain reproducible.

🎨 Layer 3 — Coding Style Guide
Python
PEP8, 4-space indent

snake_case for variables/functions

PascalCase for classes

Type hints are required for any public interface

Minimal but meaningful logging

Frontend
Keep Vue components small, modular

Follow existing architectural layout

Do not break build scripts or tooling

🧪 Layer 4 — Testing Rules
All additions MUST include tests unless trivial.

You MUST run the entire suite before declaring completion.

A failing integration test means your logic is inconsistent with the system architecture.

If NebulaGraph is required, ensure required data (basketballplayer space) is loaded.

🔄 Layer 5 — PR / Commit Guidelines
Write atomic and focused PRs.

Use descriptive commit messages:

feat: ...

fix: ...

refactor: ...

docs: ...

PR description MUST include:

Purpose

Affected modules

Required configs

Test results

Screenshots/GIFs if frontend changed

🙋 Layer 6 — Interaction Rules
When interacting with the developer:

Confirm assumptions before writing code.

Ask clarifying questions instead of guessing.

Propose improvements if ambiguity exists.

Never hallucinate file paths, APIs, or modules.

Search the codebase before introducing new utilities.

🧩 Final Rule (Supremacy Clause)
If ANY rule conflicts, follow CLAUDE.md first.

This guarantees consistency, reliability, and engineering discipline across the entire project.

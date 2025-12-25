# Sustainability Data Extraction System (AA Impact Inc. Case Study)

A production-style system to automatically extract ESRS/CSRD-aligned sustainability indicators from large bank reports (400–1,100+ pages). The system uses a multi-phase pipeline, local open‑source LLMs via Ollama, and a lightweight SQLite store to deliver structured, auditable outputs at zero API cost.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
  - [Ingestion & Parsing](#ingestion--parsing)
  - [Retrieval Layer (SimpleRAG)](#retrieval-layer-simplerag)
  - [LLM Extraction & Storage](#llm-extraction--storage)
- [Key Design Decisions](#key-design-decisions)
  - [Local Open‑Source Models via Ollama](#local-open-source-models-via-ollama)
  - [SimpleRAG Instead of Vector RAG](#simplerag-instead-of-vector-rag)
  - [“Precision Auditor” LLM Prompting](#precision-auditor-llm-prompting)
  - [Robust JSON Cleaning and Validation](#robust-json-cleaning-and-validation)
- [Features](#features)
  - [Indicators Covered](#indicators-covered)
  - [Multi‑Phase Pipeline](#multi-phase-pipeline)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Install Dependencies](#install-dependencies)
  - [Setup Ollama and Model](#setup-ollama-and-model)
- [Running the System](#running-the-system)
  - [Single‑Bank Test Run](#single-bank-test-run)
  - [Full Extraction (All Banks)](#full-extraction-all-banks)
  - [Inspecting Results](#inspecting-results)
- [Error Handling & Limitations](#error-handling--limitations)
  - [Local Model Limitations](#local-model-limitations)
  - [Accuracy vs Cost Trade‑Off](#accuracy-vs-cost-trade-off)
- [How This Meets the Case Study Criteria](#how-this-meets-the-case-study-criteria)
- [Possible Extensions](#possible-extensions)
- [License / Attribution](#license--attribution)

---

## Architecture Overview

The system is organized around three main layers that separate concerns and make the pipeline maintainable and testable.

### Ingestion & Parsing

- Reads bank sustainability / annual reports in PDF format from `data/reports/`.
- Converts pages to text and identifies candidate sections using anchor keywords (e.g., "ESRS index", "GRI content index", "Sustainability index").

### Retrieval Layer (SimpleRAG)

- Uses an in‑memory keyword-based retriever (SimpleRAG) instead of a vector DB.
- For each indicator, scores pages by keyword hits and returns top-ranked page contexts (±1 page) to the LLM.

### LLM Extraction & Storage

- A unified `LLMProcessor` orchestrates calls to a local Ollama model (e.g., `llama3.1:8b` or `mistral:7b`).
- The LLM receives a careful "precision auditor" prompt for each indicator and returns a strict JSON structure.
- A JSON cleaning layer normalizes numbers, converts `"null"` strings to `null`, and guards against malformed outputs.
- Extractions are persisted in SQLite (`data/sustainability_data.db`) and exported to CSV for analysis.

This design allows parsing, retrieval, LLM extraction, and persistence to evolve independently.

---

## Key Design Decisions

### Local Open‑Source Models via Ollama

- Runs local LLMs through an Ollama server on `localhost:11434` to avoid cloud API costs.
- No external LLM or embedding APIs are required — end‑to‑end extraction cost is effectively zero.
- The LLM backend is abstracted in `LLMProcessor`, making it easy to switch to cloud models if desired.

### SimpleRAG Instead of Vector RAG

- Pages are stored in memory as `{company: {page_num: text}}`.
- For each indicator, SimpleRAG scores pages by keyword occurrence and returns the top N contexts.
- This avoids vector DB complexity while delivering strong recall for indicators with clear textual anchors.
- A vector‑RAG pathway exists but is feature-flagged off for this case study.

### “Precision Auditor” LLM Prompting

- Prompts are framed as precision auditing tasks rather than open QA.
- Enforces matching exact row labels (e.g., "Total Scope 2 GHG Emissions") and selecting the correct year column (e.g., 2024).
- Strict JSON format enforced with fields such as:
  - `value` — a pure number (e.g., `1074786.0`)
  - `unit` — expected unit (e.g., `tCO2e`, `%`, `MWh`)
  - `source_quote` — exact row text used
  - `confidence_score` — semantics tied to row/label correctness
- This reduces systematic errors like copying the same number across indicators or pulling the wrong year.

### Robust JSON Cleaning and Validation

- Defensive cleaning for "almost JSON" common from local LLMs:
  - Strip markdown/code fences and extract the first JSON object.
  - Normalize `value` (remove commas, `%`, currency symbols) and convert to float when possible.
  - Interpret `"null"` or empty strings as `None`.
  - Normalize `data_quality` values (`"hi"`, `"med"`, `"lo"`) to `high` / `medium` / `low`.
  - Fill missing required fields with safe defaults.
- If parsing still fails, a regex reconstruction attempts to recover `value`, `unit`, `page_number`, and `source_quote`, marking the extraction as medium-quality.
- Duplicate-value safeguard: if the same numeric value appears across distinct indicators (e.g., Scope 1/2/3 all equal), confidence is down-weighted and flagged as likely copy error.

---

## Features

### Indicators Covered

- Configured to handle ~20 indicators across Environmental, Social, and Governance categories (e.g., Scope 1–3 GHG, GHG intensity, total energy, % female employees, gender pay gap, training hours, board female representation, board meetings, corruption incidents, supplier metrics).
- Each indicator includes:
  - Canonical name and ESRS reference
  - Unit specification and validation ranges
  - A list of search keywords for SimpleRAG
  - Configured extraction method (table or narrative)

### Multi‑Phase Pipeline

For each bank and report year the pipeline runs three phases plus finalization:

**Phase A – Anchor Page Discovery**

- Scans pages for anchor keywords (ESRS/GRI/index pages) and logs relevant page ranges.

**Phase B – Table‑Based Extraction**

- For table indicators, identifies relevant pages and feeds markdown tables + metadata (indicator name, expected unit, ESRS ref, page number) to the LLM with the precision prompt.
- Applies JSON cleaning, validation, and duplicate checking before saving.

**Phase C – Narrative‑Based Extraction**

- For narrative indicators (e.g., board composition, corruption incidents, net-zero target year), uses SimpleRAG to retrieve top narrative contexts and calls the LLM with a narrative prompt.

**Finalization**

- Stores successful extractions in SQLite and writes a CSV summary with `confidence` and `data_quality` labels.
- Prints high-level stats: total extractions, success rate, per-bank breakdown, confidence distribution, pages processed, and runtime.

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- Git
- Ollama installed and running locally
- 8–16 GB RAM recommended for 7B–8B models

### Install Dependencies

```bash
git clone <your-repo-url>
cd Sustainability_extractor

python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

Dependencies include PDF parsing libs, Pydantic, LangChain wrappers (OpenAI/Ollama), and SQLite (stdlib).

### Setup Ollama and Model

Install Ollama (see official instructions) and start the server:

```bash
ollama serve
```

Pull a local model, for example:

```bash
ollama pull llama3.1:8b
# or
ollama pull mistral:7b
```

Create a `.env` file (optional but recommended):

```text
PRIMARY_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
TEMPERATURE=0
CONFIDENCE_THRESHOLD=0.5
```

Place bank reports in `data/reports/` (e.g., `AIB_2024.pdf`, `BBVA_2024.pdf`).

---

## Running the System

### Single‑Bank Test Run

Validate the end‑to‑end pipeline on one bank first:

```bash
python main.py --bank AIB --model llama3.1:8b --no-rag
```

This will:

- Scan the AIB report for anchor pages
- Run table extraction for configured indicators
- Run narrative extraction using SimpleRAG
- Write results to `data/sustainability_data.db` and `data/output/sustainability_extractions.csv`

### Full Extraction (All Banks)

Once single‑bank looks good, run:

```bash
python main.py --model llama3.1:8b --no-rag
```

This processes all configured banks sequentially and updates the DB/CSV.

### Inspecting Results

**CSV:**

```bash
cat data/output/sustainability_extractions.csv
```

Or open the CSV in Excel / Sheets and filter by `company`, `indicator`, `confidence`, or `data_quality`.

**SQLite:**

```bash
python -c "import sqlite3; conn = sqlite3.connect('data/sustainability_data.db'); print(conn.execute('SELECT COUNT(*) FROM extractions').fetchone()); conn.close()"
```

---

## Error Handling & Limitations

### 6.1 Local Model Limitations

- Local 7B–8B models may:
  - Reuse the same number across indicators (mitigated with duplicate detection / confidence downgrades)
  - Produce malformed JSON (mitigated by cleaning and reconstruction)
- The system marks suspicious extractions with lowered confidence and `data_quality = "low"`.

### 6.2 Accuracy vs Cost Trade‑Off

- The `LLMProcessor` can point to a cloud model (e.g., GPT‑4‑class) to increase accuracy at higher marginal cost.
- For the case study, configuration favors local zero‑cost models with controls (confidence thresholds, validation ranges, duplicate detection) to keep outputs usable despite model noise.

---

## How This Meets the Case Study Criteria

- **Extraction Accuracy** — indicator‑specific prompts, year awareness, unit validation, and duplicate detection reduce systematic errors.
- **Technical Approach** — modular architecture, LLM backend abstraction, robust JSON parsing strategies, and pragmatic SimpleRAG usage.
- **Scalability & Cost** — zero marginal API cost per run; local hardware does the heavy lifting.
- **Innovation** — hybrid/local LLM architecture under real constraints with explicit trade‑off documentation.

---

## Possible Extensions

- Re‑enable and harden a vector‑based RAG for more ambiguous narrative indicators.
- Add a UI / dashboard to explore extractions with PDF page previews.
- Implement per‑indicator post‑processing (e.g., intensity calculations from raw values).
- Introduce human‑in‑the‑loop validation and feedback logging to iteratively tighten prompts and validation rules.

---

## License / Attribution

Copy this into `README.md` and adapt details (repo URL, exact indicators, and metrics from runs) to match your implementation and results.

---

*Generated README content for the AA Impact Inc. case study.*


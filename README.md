# Multi-Agent Data Analyst

An autonomous, multi-agent data analysis and reporting system built with **LangGraph**, **pandas**, **Plotly**, **Streamlit**, and the **Gemini API**.

## Pipeline Flow

```text
User uploads CSV
       │
       ▼
[ Cleaner Agent ] ──> Handles missing values, validates schema, prepares clean data
       │
       ▼
[ Analyst Agent ] ──> Writes and runs pandas code to calculate statistical insights
       │
       ▼
[ Visualizer Agent ] ──> Generates interactive Plotly charts and visual summaries
       │
       ▼
[ Reviewer Agent ] ──> Independently rechecks the numbers & logic
       │
       ├─── Discrepancy found (max 3 retries) ──> [ Routes back to Analyst ]
       │
       └─── Validated ──> Final Insights Report with Charts
```

---

## Project Structure

```text
multi-data-analyst-agent/
│
├── agents/                  # Multi-agent definitions and prompt logic
│   ├── cleaner.py           # Cleans and standardizes raw CSV data
│   ├── analyst.py           # Generates and executes pandas analysis code
│   ├── visualizer.py        # Generates interactive Plotly charts
│   └── reviewer.py          # Verifies numbers & routes feedback loop
│
├── tools/                   # Helper tools & execution environments
│   ├── profiler.py          # DataFrame inspection and summary statistics
│   └── sandbox.py           # Safe Python code execution environment
│
├── components/              # Reusable UI widgets and layout modules
├── eval/                    # Benchmarks and evaluation test suites
├── data/                    # Sample and test datasets
├── assets/                  # Media, architecture diagrams, and images
│
├── graph.py                 # LangGraph StateGraph orchestration and compilation
├── state.py                 # Shared typed state schema across agent nodes
├── llm.py                   # Gemini LLM initialization and helper function
├── run.py                   # Headless CLI entrypoint for local execution
├── app.py                   # Streamlit web application interface
│
├── requirements.txt         # Project dependencies
├── .env.example             # Template for environment variables
├── .gitignore               # Ignored files (secrets, venv, pycache)
└── README.md                # Project documentation
```

---

## Quickstart Guide

### 1. Prerequisites
- **Python 3.11**
- A **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/)

### 2. Setup Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### On macOS / Linux:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the `.env.example` file to `.env`:
```powershell
# Windows PowerShell:
Copy-Item .env.example .env

# macOS / Linux:
cp .env.example .env
```
Open `.env` and replace `your_key_here` with your actual Gemini API key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 5. Verify Gemini Connection
Run a quick test to confirm that `call_llm` in `llm.py` connects to Gemini:
```bash
python llm.py
```
Or via inline Python:
```bash
python -c "from llm import call_llm; print(call_llm('Hello, say test successful!'))"
```

===========================================================
PROJECT: AI Vendor Management System (RAG Module)
DEVELOPER: Yang Wei
DATE: 2026-05-07
ENVIRONMENT: macOS M2 (Apple Silicon)
===========================================================

1. VERSION SELECTION LOGIC:
   - Target Version: Python 3.11.x
   - Reason: Ensuring 100% compatibility with Docling, PyTorch (MPS), 
     and Milvus Lite on Apple Silicon architecture.

2. ENVIRONMENT SETUP:
   $ brew install python@3.11
   $ cd /Users/grace/Desktop/PIP2026/python/my_rag_project
   $ /opt/homebrew/bin/python3.11 -m venv .venv
   $ source .venv/bin/activate

3. DEPENDENCY INSTALLATION:
   $ pip install --upgrade pip
   $ pip install -r requirements.txt

4. ACCELERATION NOTES (M2):
   - Docling leverages MPS (Metal Performance Shaders) for layout analysis.
   - Using Milvus Lite for serverless local vector storage.

5. VERIFICATION:
   $ python --version  # Should be 3.11.x
   $ python parse_pdf.py  # Test parsing sample.pdf
===========================================================
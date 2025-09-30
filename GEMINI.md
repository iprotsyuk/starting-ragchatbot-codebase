# GEMINI.md

## Project Overview

This project is a full-stack Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about course materials. It uses a Python backend with FastAPI and a vanilla JavaScript frontend.

**Key Technologies:**

*   **Backend:** Python, FastAPI, Uvicorn
*   **Frontend:** HTML, CSS, JavaScript
*   **Vector Store:** ChromaDB
*   **AI Model:** Google Gemini
*   **Package Manager:** uv

**Architecture:**

The application is composed of three main parts:

1.  **Frontend:** A simple web interface that allows users to interact with the chatbot. It sends user queries to the backend and displays the results.
2.  **Backend:** A FastAPI server that exposes a `/api/query` endpoint to handle user queries and a `/api/courses` endpoint to provide course statistics. It uses a `RAGSystem` to orchestrate the process of retrieving relevant information and generating responses.
3.  **RAG System:** The core of the application, which includes:
    *   A `VectorStore` (using ChromaDB) to store and search for course content.
    *   An `AIGenerator` that interacts with the Gemini API to generate responses.
    *   A `ToolManager` and `CourseSearchTool` to enable the AI model to search for course content.
    *   A `DocumentProcessor` to process and chunk the course documents.

## Building and Running

### Prerequisites

*   Python 3.13 or higher
*   `uv` (Python package manager)
*   A Gemini API key

### Installation

1.  **Install `uv`:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Install Python dependencies:**
    ```bash
    uv sync
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your Gemini API key:
    ```
    GEMINI_API_KEY=your_gemini_api_key_here
    ```

### Running the Application

*   **Quick Start (Recommended):**
    ```bash
    chmod +x run.sh
    ./run.sh
    ```

*   **Manual Start:**
    ```bash
    cd backend
    uv run uvicorn app:app --reload --port 8000
    ```

The application will be available at `http://localhost:8000`.

## Development Conventions

*   **Backend:** The backend is a modular FastAPI application. The core logic is encapsulated in the `RAGSystem` class, which is initialized in `app.py`. The application is configured to reload on code changes.
*   **Frontend:** The frontend is a simple single-page application with no build process. The JavaScript code is in `frontend/script.js` and communicates with the backend via a `/api` proxy.
*   **Dependencies:** Python dependencies are managed with `uv` and are listed in `pyproject.toml`.

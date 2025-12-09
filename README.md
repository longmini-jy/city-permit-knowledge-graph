# City Permitting Knowledge Graph

A structured, machine-readable knowledge graph system that ingests, normalizes, versions, and serves municipal permitting rules and regulations.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env` and set your `OPENAI_API_KEY`.

### 4. Run the API

```bash
uvicorn src.app.main:app --reload
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   └── app/
│       ├── __init__.py
│       ├── main.py           # FastAPI application
│       ├── config.py         # Configuration management
│       └── api/
│           └── __init__.py
├── tests/
│   └── __init__.py
├── requirements.txt
├── .env.example
├── .env
└── README.md
```

## Development

### Running Tests

```bash
pytest
```

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

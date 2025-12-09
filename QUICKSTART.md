# Quick Start Guide

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- OpenAI API key

## Setup Instructions

### 1. Virtual Environment (Already Created)

The virtual environment has been created at `venv/`. To activate it:

```bash
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 2. Dependencies (Already Installed)

All dependencies have been installed. To verify:

```bash
venv/bin/pip list
```

### 3. Configure OpenAI API Key

Edit the `.env` file and replace the placeholder with your actual OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

### 4. Run the API

Start the development server:

```bash
venv/bin/uvicorn src.app.main:app --reload
```

Or use the Python module directly:

```bash
venv/bin/python -m src.app.main
```

The API will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### 5. Test the API

Run the test suite:

```bash
venv/bin/pytest
```

Or test with curl:

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   └── app/
│       ├── __init__.py          # Package initialization
│       ├── main.py              # FastAPI application
│       ├── config.py            # Configuration management
│       └── api/                 # API routes (future)
│           └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_main.py         # API tests
├── venv/                    # Virtual environment
├── .env                     # Environment variables (DO NOT COMMIT)
├── .env.example             # Environment template
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
├── pytest.ini               # Pytest configuration
├── setup.sh                 # Setup script
└── README.md                # Full documentation
```

## Next Steps

1. **Set your OpenAI API key** in `.env`
2. **Start the server** with `uvicorn src.app.main:app --reload`
3. **Visit the docs** at http://localhost:8000/docs
4. **Begin implementing** the next task in the implementation plan

## Troubleshooting

### Import Errors

If you see import errors, make sure the virtual environment is activated:

```bash
source venv/bin/activate
```

### Port Already in Use

If port 8000 is already in use, specify a different port:

```bash
uvicorn app.main:app --reload --port 8001
```

### OpenAI API Key Not Set

The health check endpoint will show `openai_configured: false` if the API key is not set. Edit `.env` to add your key.

## Development Workflow

1. Activate virtual environment: `source venv/bin/activate`
2. Make code changes
3. Run tests: `pytest`
4. Check API: Visit http://localhost:8000/docs
5. Commit changes (excluding `.env`)

## Available Commands

```bash
# Run API server
venv/bin/uvicorn src.app.main:app --reload

# Run tests
venv/bin/pytest

# Run tests with coverage
venv/bin/pytest --cov=src

# Run specific test
venv/bin/pytest tests/test_main.py::test_root_endpoint -v

# Check code style (install first: pip install black flake8)
venv/bin/black src/ tests/
venv/bin/flake8 src/ tests/
```

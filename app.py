# app.py — entry point for HuggingFace Spaces
import uvicorn
from main import app  # Import FastAPI 'app' from your main.py

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

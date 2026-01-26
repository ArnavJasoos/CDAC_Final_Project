from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/clean-text")
async def clean_text(request: TextRequest):
    return {
        "sentiment": {
            "positive": 0.7,
            "negative": 0.2,
            "neutral": 0.1
        },
        "events": [
            {
                "date": "2026-01-22",
                "event": "Test event",
                "confidence": 0.95
            }
        ],
        "entities": {
            "PERSON": ["John", "Alice"],
            "LOCATION": ["New York", "London"],
            "ORG": ["Google", "Microsoft"]
        }
    }

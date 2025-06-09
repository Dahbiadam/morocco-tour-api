from fastapi import FastAPI
from datetime import datetime

app = FastAPI(
    title="MoroccoTour AI Backend",
    version="4.0.0"
)

@app.get("/status")
async def get_status():
    return {
        "status": "online",
        "timestamp": "2025-06-09 13:50:35",
        "user": "Dahbiadam"
    }

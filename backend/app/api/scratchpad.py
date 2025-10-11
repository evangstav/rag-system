from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List

router = APIRouter()

# Simple in-memory storage for MVP (use database in production)
scratchpad_store: Dict[str, dict] = {}


class Todo(BaseModel):
    id: str
    text: str
    done: bool


class ScratchpadData(BaseModel):
    todos: List[Todo] = []
    notes: str = ""
    journal: str = ""


@router.get("/")
async def get_scratchpad():
    """Get current scratchpad data"""
    return scratchpad_store.get("default", ScratchpadData().model_dump())


@router.post("/")
async def save_scratchpad(data: ScratchpadData):
    """Save scratchpad data"""
    scratchpad_store["default"] = data.model_dump()
    return {"status": "saved"}

"""
ConvoHubAI - LiveKit Video API Routes
100% Free & Open Source Video Integration
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import time
import jwt

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models.user import User


router = APIRouter(prefix="/video", tags=["Video"])


# ============================================
# CONFIGURATION (from settings)
# ============================================

LIVEKIT_API_KEY = settings.livekit_api_key
LIVEKIT_API_SECRET = settings.livekit_api_secret
LIVEKIT_URL = settings.livekit_url


# ============================================
# SCHEMAS
# ============================================

class CreateRoomRequest(BaseModel):
    name: str
    empty_timeout: int = 300
    max_participants: int = 2
    metadata: Optional[str] = None


class TokenRequest(BaseModel):
    room_name: str
    participant_name: str
    is_agent: bool = False


class RoomResponse(BaseModel):
    name: str
    sid: Optional[str] = None
    empty_timeout: int
    max_participants: int
    creation_time: datetime
    metadata: Optional[str] = None


class TokenResponse(BaseModel):
    token: str
    room_name: str
    participant_name: str


# ============================================
# HELPER FUNCTIONS
# ============================================

def generate_livekit_token(
    room_name: str,
    participant_name: str,
    is_agent: bool = False,
    ttl: int = 3600
) -> str:
    """
    Generate a LiveKit access token.
    """
    now = int(time.time())
    
    video_grant = {
        "room": room_name,
        "roomJoin": True,
        "canPublish": True,
        "canSubscribe": True,
        "canPublishData": True,
    }
    
    if is_agent:
        video_grant["roomAdmin"] = True
        video_grant["roomRecord"] = True
    
    payload = {
        "iss": LIVEKIT_API_KEY,
        "sub": participant_name,
        "nbf": now,
        "exp": now + ttl,
        "iat": now,
        "name": participant_name,
        "video": video_grant,
        "metadata": "",
    }
    
    token = jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")
    return token


# In-memory room storage
active_rooms = {}


# ============================================
# API ROUTES
# ============================================

@router.post("/token", response_model=TokenResponse)
async def create_token(
    data: TokenRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Generate an access token for a participant to join a video room.
    """
    participant_name = data.participant_name or current_user.full_name or current_user.email
    
    token = generate_livekit_token(
        room_name=data.room_name,
        participant_name=participant_name,
        is_agent=data.is_agent
    )
    
    return TokenResponse(
        token=token,
        room_name=data.room_name,
        participant_name=participant_name
    )


@router.post("/rooms")
async def create_room(
    data: CreateRoomRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new video room.
    For LiveKit Cloud, rooms are created automatically when first participant joins.
    We just generate a token and return room info.
    """
    room_name = data.name
    
    # Generate token for the user
    participant_name = current_user.full_name or current_user.email or "User"
    token = generate_livekit_token(
        room_name=room_name,
        participant_name=participant_name,
        is_agent=False
    )
    
    # Store room info locally
    room_info = {
        "name": room_name,
        "sid": f"RM_{int(time.time())}",
        "empty_timeout": data.empty_timeout,
        "max_participants": data.max_participants,
        "creation_time": datetime.utcnow().isoformat(),
        "metadata": data.metadata,
        "created_by": str(current_user.id),
        "token": token,
        "livekit_url": LIVEKIT_URL
    }
    
    active_rooms[room_name] = room_info
    
    return room_info


@router.get("/rooms")
async def list_rooms(
    current_user: User = Depends(get_current_user)
):
    """
    List all active video rooms.
    """
    user_rooms = [
        room for room in active_rooms.values()
        if room.get("created_by") == str(current_user.id)
    ]
    
    return {"rooms": user_rooms}


@router.get("/rooms/{room_name}")
async def get_room(
    room_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get details of a specific room.
    """
    if room_name not in active_rooms:
        raise HTTPException(status_code=404, detail="Room not found")
    
    return active_rooms[room_name]


@router.delete("/rooms/{room_name}")
async def delete_room(
    room_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete a video room.
    """
    if room_name not in active_rooms:
        raise HTTPException(status_code=404, detail="Room not found")
    
    del active_rooms[room_name]
    
    return {"message": "Room deleted successfully"}


@router.get("/config")
async def get_config(
    current_user: User = Depends(get_current_user)
):
    """
    Get LiveKit configuration for the frontend.
    """
    return {
        "url": LIVEKIT_URL,
        "enabled": bool(LIVEKIT_API_KEY and LIVEKIT_API_SECRET)
    }
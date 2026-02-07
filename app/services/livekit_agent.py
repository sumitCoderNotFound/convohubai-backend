"""
ConvoHubAI - LiveKit AI Agent Service
Real-time Voice & Video AI Agents using LiveKit
100% Open Source - No per-minute costs when self-hosted
"""

import asyncio
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass
import aiohttp

from app.core.config import settings


@dataclass
class AgentConfig:
    """Configuration for an AI agent"""
    agent_id: str
    name: str
    system_prompt: str
    welcome_message: str
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    voice_provider: str = "elevenlabs"  # or "deepgram", "openai"
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Default ElevenLabs voice
    language: str = "en"
    temperature: float = 0.7
    channels: list = None
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = ["chat"]


class LiveKitAgentService:
    """
    Service to manage LiveKit AI Agents
    
    LiveKit Agents can:
    - Join video/audio rooms
    - Listen to users (Speech-to-Text)
    - Process with LLM (GPT-4, Claude, etc.)
    - Respond with voice (Text-to-Speech)
    - See video (Vision models)
    """
    
    def __init__(self):
        self.livekit_url = settings.livekit_url
        self.api_key = settings.livekit_api_key
        self.api_secret = settings.livekit_api_secret
        self.active_agents: Dict[str, Any] = {}
    
    async def create_agent_token(
        self, 
        room_name: str, 
        agent_config: AgentConfig
    ) -> str:
        """
        Create a token for the AI agent to join a room
        """
        import time
        import jwt
        
        now = int(time.time())
        
        # Agent has special permissions
        video_grant = {
            "room": room_name,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
            "roomAdmin": True,
            "agent": True,  # Mark as agent
        }
        
        payload = {
            "iss": self.api_key,
            "sub": f"agent-{agent_config.agent_id}",
            "nbf": now,
            "exp": now + 86400,  # 24 hours
            "iat": now,
            "name": agent_config.name,
            "video": video_grant,
            "metadata": json.dumps({
                "type": "ai_agent",
                "agent_id": agent_config.agent_id,
                "llm_provider": agent_config.llm_provider,
                "llm_model": agent_config.llm_model,
            }),
        }
        
        token = jwt.encode(payload, self.api_secret, algorithm="HS256")
        return token
    
    async def dispatch_agent_to_room(
        self,
        room_name: str,
        agent_config: AgentConfig
    ) -> Dict[str, Any]:
        """
        Dispatch an AI agent to join a LiveKit room
        
        In production, this would:
        1. Start a Python agent worker (using livekit-agents SDK)
        2. Or call a serverless function
        3. Or send to a queue for processing
        """
        
        # Generate token for the agent
        agent_token = await self.create_agent_token(room_name, agent_config)
        
        # Store active agent info
        self.active_agents[room_name] = {
            "agent_id": agent_config.agent_id,
            "agent_name": agent_config.name,
            "room_name": room_name,
            "status": "joining",
            "token": agent_token,
            "config": {
                "system_prompt": agent_config.system_prompt,
                "welcome_message": agent_config.welcome_message,
                "llm_provider": agent_config.llm_provider,
                "llm_model": agent_config.llm_model,
                "voice_provider": agent_config.voice_provider,
                "voice_id": agent_config.voice_id,
                "temperature": agent_config.temperature,
            }
        }
        
        return {
            "success": True,
            "room_name": room_name,
            "agent_id": agent_config.agent_id,
            "agent_token": agent_token,
            "livekit_url": self.livekit_url,
            "message": f"Agent '{agent_config.name}' dispatched to room"
        }
    
    async def remove_agent_from_room(self, room_name: str) -> bool:
        """Remove an agent from a room"""
        if room_name in self.active_agents:
            del self.active_agents[room_name]
            return True
        return False
    
    async def get_agent_status(self, room_name: str) -> Optional[Dict]:
        """Get status of an agent in a room"""
        return self.active_agents.get(room_name)
    
    async def list_active_agents(self) -> list:
        """List all active agents"""
        return list(self.active_agents.values())


# Singleton instance
livekit_agent_service = LiveKitAgentService()


# ============================================
# AGENT WORKER CODE (for reference)
# This would run as a separate Python process
# ============================================
"""
To run an actual LiveKit AI agent, you need to:

1. Install the SDK:
   pip install livekit-agents livekit-plugins-openai livekit-plugins-silero

2. Create an agent worker (agent_worker.py):

```python
from livekit import agents
from livekit.agents import llm, stt, tts, voice_assistant
from livekit.plugins import openai, silero, elevenlabs

async def entrypoint(ctx: agents.JobContext):
    # Get agent config from room metadata
    room = ctx.room
    
    # Initialize components
    vad = silero.VAD.load()  # Voice Activity Detection
    stt = openai.STT()       # Speech-to-Text
    llm = openai.LLM(model="gpt-4")  # Language Model
    tts = elevenlabs.TTS()   # Text-to-Speech
    
    # Create voice assistant
    assistant = voice_assistant.VoiceAssistant(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        chat_ctx=llm.ChatContext().append(
            role="system",
            text="You are a helpful AI assistant..."
        ),
    )
    
    # Start the assistant
    assistant.start(ctx.room)
    
    # Send welcome message
    await assistant.say("Hello! How can I help you today?")
    
    # Keep running
    await assistant.join()

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
```

3. Run the worker:
   python agent_worker.py dev --url wss://your-livekit-server
"""
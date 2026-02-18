"""
ConvoHubAI - Dynamic LiveKit AI Agent Worker
COST OPTIMISED: Target 1-1.5p per minute

Cost Breakdown:
- Groq LLM: FREE (0.08p)
- Groq Whisper STT: FREE (0.08p)  
- Deepgram Aura TTS: 0.15p
- Telephony (Telnyx): 0.3p
- LiveKit: 0.4p
- Server: 0.1p
TOTAL: ~1.1p per minute
"""

import asyncio
import logging
import httpx
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from typing import List, Dict

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess, 
    WorkerOptions,
    cli,
    RoomInputOptions,
)
from livekit.plugins import silero, groq, deepgram

# NOTE: OpenAI import removed to prevent accidental usage (expensive!)
# If you need OpenAI, uncomment: from livekit.plugins import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-agent")

# Backend API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ===========================================
# COST CONFIGURATION
# ===========================================
# Set these to control costs. Default = CHEAPEST options

DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "groq")  # FREE!
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "llama-3.3-70b-versatile")  # FREE!
DEFAULT_STT_PROVIDER = os.getenv("DEFAULT_STT_PROVIDER", "groq")  # FREE! (Whisper)
DEFAULT_TTS_PROVIDER = os.getenv("DEFAULT_TTS_PROVIDER", "deepgram")  # Cheapest quality TTS

# Cost tracking (for analytics)
COST_PER_MINUTE = {
    "groq_llm": 0.0008,      # ~0.08p - essentially FREE
    "groq_stt": 0.0008,      # ~0.08p - FREE Whisper
    "deepgram_stt": 0.0080,  # ~0.8p
    "deepgram_tts": 0.0015,  # ~0.15p (Aura)
    "openai_tts": 0.0120,    # ~1.2p - EXPENSIVE, avoid!
    "total_target": 0.011,   # ~1.1p target
}


class TranscriptManager:
    """
    Manages conversation transcripts and saves them to the backend.
    Also tracks cost per call.
    """
    
    def __init__(self, agent_id: str, room_name: str, participant_identity: str):
        self.agent_id = agent_id
        self.room_name = room_name
        self.participant_identity = participant_identity
        self.messages: List[Dict] = []
        self.call_start_time = datetime.utcnow()
        self.call_id = None
    
    def add_message(self, role: str, content: str):
        """Add a message to the transcript."""
        self.messages.append({
            "role": role,  # "user" or "assistant"
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        logger.info(f"📝 [{role}]: {content[:100]}...")
    
    def calculate_cost(self, duration_seconds: int) -> dict:
        """Calculate estimated cost for this call."""
        duration_minutes = duration_seconds / 60
        estimated_cost_gbp = duration_minutes * COST_PER_MINUTE["total_target"]
        
        return {
            "duration_minutes": round(duration_minutes, 2),
            "estimated_cost_gbp": round(estimated_cost_gbp, 4),
            "cost_per_minute_gbp": COST_PER_MINUTE["total_target"],
        }
    
    async def save_transcript(self):
        """Save the transcript to the backend API."""
        if not self.messages:
            logger.info("No messages to save")
            return
        
        call_end_time = datetime.utcnow()
        duration_seconds = (call_end_time - self.call_start_time).total_seconds()
        cost_info = self.calculate_cost(int(duration_seconds))
        
        transcript_data = {
            "agent_id": self.agent_id,
            "room_name": self.room_name,
            "participant_identity": self.participant_identity,
            "call_start": self.call_start_time.isoformat(),
            "call_end": call_end_time.isoformat(),
            "duration_seconds": int(duration_seconds),
            "messages": self.messages,
            "message_count": len(self.messages),
            "channel": "video",
            "cost_info": cost_info,  # Include cost tracking
        }
        
        logger.info(f"💰 Call cost: £{cost_info['estimated_cost_gbp']:.4f} ({cost_info['duration_minutes']:.2f} mins)")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{API_URL}/api/v1/calls/transcript",
                    json=transcript_data,
                    timeout=10.0
                )
                
                if response.status_code in [200, 201]:
                    result = response.json()
                    self.call_id = result.get("call_id")
                    logger.info(f"✅ Transcript saved! Call ID: {self.call_id}")
                    logger.info(f"📊 Duration: {int(duration_seconds)}s, Messages: {len(self.messages)}")
                else:
                    logger.warning(f"Failed to save transcript: {response.status_code}")
                    await self._save_local_backup(transcript_data)
                    
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
            await self._save_local_backup(transcript_data)
    
    async def _save_local_backup(self, transcript_data: dict):
        """Save transcript locally as backup if API fails."""
        try:
            backup_dir = "transcripts"
            os.makedirs(backup_dir, exist_ok=True)
            
            filename = f"{backup_dir}/transcript_{self.room_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(transcript_data, f, indent=2)
            
            logger.info(f"💾 Transcript backed up locally: {filename}")
        except Exception as e:
            logger.error(f"Failed to save local backup: {e}")


class DynamicAgent(Agent):
    """
    Dynamic AI Agent that uses configuration from the backend.
    """
    
    def __init__(self, system_prompt: str, agent_name: str = "AI Assistant"):
        super().__init__(
            instructions=system_prompt
        )
        self.agent_name = agent_name


async def fetch_agent_config(agent_id: str) -> dict:
    """
    Fetch agent configuration from the backend API.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_URL}/api/v1/agents/{agent_id}",
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to fetch agent config: {response.status_code}")
                return None
    except Exception as e:
        logger.error(f"Error fetching agent config: {e}")
        return None


def get_llm_from_config(provider: str = None, model: str = None):
    """
    Get the appropriate LLM based on provider configuration.
    DEFAULT: Groq (FREE!)
    
    Cost comparison:
    - Groq Llama 3.3 70B: FREE (~0.08p/min)
    - OpenAI GPT-4o-mini: ~2p/min (25x more expensive!)
    - OpenAI GPT-4o: ~10p/min (125x more expensive!)
    """
    provider = (provider or DEFAULT_LLM_PROVIDER).lower()
    
    if provider == "groq":
        model_name = model or DEFAULT_LLM_MODEL
        logger.info(f"🧠 Using Groq LLM: {model_name} (FREE!)")
        return groq.LLM(model=model_name)
    
    elif provider == "openai":
        # WARNING: OpenAI is expensive! Only use if specifically requested
        logger.warning("⚠️ Using OpenAI LLM - this is EXPENSIVE! Consider Groq instead.")
        try:
            from livekit.plugins import openai as openai_plugin
            model_name = model or "gpt-4o-mini"
            return openai_plugin.LLM(model=model_name)
        except ImportError:
            logger.error("OpenAI plugin not available, falling back to Groq")
            return groq.LLM(model=DEFAULT_LLM_MODEL)
    
    else:
        logger.info(f"Unknown provider '{provider}', using FREE Groq")
        return groq.LLM(model=DEFAULT_LLM_MODEL)


def get_stt_from_config(provider: str = None):
    """
    Get Speech-to-Text based on configuration.
    DEFAULT: Groq Whisper (FREE!)
    
    Cost comparison:
    - Groq Whisper: FREE (~0.08p/min)
    - Deepgram Nova-2: ~0.8p/min (10x more expensive!)
    """
    provider = (provider or DEFAULT_STT_PROVIDER).lower()
    
    if provider == "groq":
        logger.info("🎤 Using Groq Whisper STT (FREE!)")
        return groq.STT(model="whisper-large-v3")
    
    elif provider == "deepgram":
        logger.info("🎤 Using Deepgram STT (paid)")
        return deepgram.STT()
    
    else:
        # Default to FREE Groq Whisper
        logger.info("🎤 Defaulting to Groq Whisper STT (FREE!)")
        return groq.STT(model="whisper-large-v3")


def get_tts_from_config(provider: str = None):
    """
    Get Text-to-Speech based on configuration.
    DEFAULT: Deepgram Aura (cheapest quality TTS!)
    
    Cost comparison:
    - Deepgram Aura: ~0.15p/min (CHEAPEST!)
    - OpenAI TTS-1: ~1.2p/min (8x more expensive!)
    - ElevenLabs: ~2.4p/min (16x more expensive!)
    """
    provider = (provider or DEFAULT_TTS_PROVIDER).lower()
    
    if provider == "deepgram":
        logger.info("🔊 Using Deepgram Aura TTS (cheapest!)")
        return deepgram.TTS(model="aura-asteria-en")
    
    elif provider == "openai":
        # WARNING: OpenAI TTS is expensive!
        logger.warning("⚠️ Using OpenAI TTS - this is EXPENSIVE! Consider Deepgram Aura instead.")
        try:
            from livekit.plugins import openai as openai_plugin
            return openai_plugin.TTS(model="tts-1", voice="alloy")
        except ImportError:
            logger.error("OpenAI plugin not available, falling back to Deepgram")
            return deepgram.TTS(model="aura-asteria-en")
    
    else:
        # Default to cheapest option
        logger.info("🔊 Defaulting to Deepgram Aura TTS (cheapest!)")
        return deepgram.TTS(model="aura-asteria-en")


def get_default_system_prompt(agent_name: str = "AI Assistant") -> str:
    """Returns a default system prompt if none is configured."""
    return f"""You are {agent_name}, a helpful and professional AI assistant.

## YOUR CONVERSATION FLOW:

### Step 1: GREETING & NAME
- Greet warmly and ask for their name
- "Hello! I'm {agent_name}. How can I help you today? May I know your name?"

### Step 2: COLLECT EMAIL (if needed)
- After getting their name, ask for email if relevant
- "Thanks [Name]! What's the best email to reach you?"

### Step 3: UNDERSTAND THEIR NEEDS
- Ask what they need help with
- Listen carefully and provide helpful responses

### Step 4: PROVIDE ASSISTANCE
- Answer questions clearly and concisely
- Offer relevant information and next steps

### Step 5: CONFIRM & CLOSE
- Summarize what was discussed
- Ask if there's anything else you can help with

## IMPORTANT RULES:
1. Keep responses SHORT (1-2 sentences) - this is voice!
2. Be warm, friendly, and professional
3. Listen actively and respond to what they say
4. If you don't understand, ask for clarification

Remember: You're having a voice conversation, so be natural and conversational!"""


def prewarm(proc: JobProcess):
    """Preload models for faster startup"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """
    Main entry point for the AI agent.
    COST OPTIMISED: Uses FREE Groq for LLM & STT, cheap Deepgram for TTS.
    """
    logger.info(f"🤖 Agent joining room: {ctx.room.name}")
    logger.info(f"💰 Cost target: ~1.1p per minute")
    
    # Connect to the room
    await ctx.connect()
    
    # Extract agent_id from room name
    room_name = ctx.room.name
    agent_id = None
    agent_config = None
    
    # Try to extract agent_id from room name
    # Format: "agent-{uuid}-{timestamp}"
    if room_name.startswith("agent-"):
        parts = room_name.split("-")
        if len(parts) >= 6:
            try:
                agent_id = "-".join(parts[1:6])
                logger.info(f"📋 Extracted agent_id: {agent_id}")
            except:
                agent_id = parts[1]
    
    # Fetch agent config from API
    if agent_id:
        agent_config = await fetch_agent_config(agent_id)
        if agent_config:
            logger.info(f"✅ Loaded agent config: {agent_config.get('name', 'Unknown')}")
    
    # Extract configuration or use defaults
    if agent_config:
        agent_name = agent_config.get("name", "AI Assistant")
        system_prompt = agent_config.get("system_prompt") or get_default_system_prompt(agent_name)
        welcome_message = agent_config.get("welcome_message") or f"Hello! I'm {agent_name}. How can I help you today?"
        
        # LLM config - can be overridden per agent, but defaults to FREE Groq
        llm_provider = agent_config.get("llm_provider") or DEFAULT_LLM_PROVIDER
        llm_model = agent_config.get("llm_model") or DEFAULT_LLM_MODEL
        
        # STT/TTS config - can be overridden, but defaults to cheapest
        stt_provider = agent_config.get("stt_provider") or DEFAULT_STT_PROVIDER
        tts_provider = agent_config.get("tts_provider") or DEFAULT_TTS_PROVIDER
    else:
        agent_name = "AI Assistant"
        system_prompt = get_default_system_prompt(agent_name)
        welcome_message = "Hello! I'm your AI Assistant. How can I help you today?"
        llm_provider = DEFAULT_LLM_PROVIDER
        llm_model = DEFAULT_LLM_MODEL
        stt_provider = DEFAULT_STT_PROVIDER
        tts_provider = DEFAULT_TTS_PROVIDER
    
    logger.info(f"🎯 Agent: {agent_name}")
    logger.info(f"🧠 LLM: {llm_provider} / {llm_model}")
    logger.info(f"🎤 STT: {stt_provider}")
    logger.info(f"🔊 TTS: {tts_provider}")
    
    # Wait for a participant  
    participant = await ctx.wait_for_participant()
    logger.info(f"👤 User joined: {participant.identity}")
    
    # Initialize transcript manager
    transcript = TranscriptManager(
        agent_id=agent_id or "unknown",
        room_name=room_name,
        participant_identity=participant.identity
    )
    
    # Get LLM (FREE Groq by default!)
    llm = get_llm_from_config(llm_provider, llm_model)
    
    # Create the agent session with COST-OPTIMISED providers
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=get_stt_from_config(stt_provider),   # FREE Groq Whisper!
        llm=llm,                                  # FREE Groq LLM!
        tts=get_tts_from_config(tts_provider),   # Cheap Deepgram Aura!
    )
    
    # Create dynamic agent
    agent = DynamicAgent(
        system_prompt=system_prompt,
        agent_name=agent_name
    )
    
    # Track conversation for transcript
    @session.on("user_speech_committed")
    def on_user_speech(msg):
        """Called when user finishes speaking."""
        if hasattr(msg, 'content') and msg.content:
            transcript.add_message("user", msg.content)
    
    @session.on("agent_speech_committed") 
    def on_agent_speech(msg):
        """Called when agent finishes speaking."""
        if hasattr(msg, 'content') and msg.content:
            transcript.add_message("assistant", msg.content)
    
    # Start the session
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(),
    )
    
    logger.info(f"✅ {agent_name} started!")
    
    # Send and record welcome message
    transcript.add_message("assistant", welcome_message)
    await session.say(welcome_message)
    
    # Wait for the session to end (participant disconnects)
    try:
        disconnect_event = asyncio.Event()
        
        @ctx.room.on("participant_disconnected")
        def on_disconnect(p):
            if p.identity == participant.identity:
                logger.info(f"👋 User disconnected: {p.identity}")
                disconnect_event.set()
        
        @ctx.room.on("disconnected")
        def on_room_disconnect():
            logger.info("🔌 Room disconnected")
            disconnect_event.set()
        
        await disconnect_event.wait()
        
    except asyncio.CancelledError:
        logger.info("Session cancelled")
    finally:
        # Save transcript when call ends
        logger.info("📝 Saving transcript...")
        await transcript.save_transcript()
        logger.info("✅ Call ended, transcript saved!")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     ConvoHubAI - COST OPTIMISED AI Agent                      ║
║     Target: 1-1.5p per minute                                 ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  💰 Cost Breakdown (per minute):                              ║
║  • Groq LLM (Llama 3.3 70B):     FREE  (~0.08p)              ║
║  • Groq Whisper STT:             FREE  (~0.08p)              ║
║  • Deepgram Aura TTS:            ~0.15p                       ║
║  • Telephony (Telnyx):           ~0.30p                       ║
║  • LiveKit WebRTC:               ~0.40p                       ║
║  • Server:                       ~0.10p                       ║
║  ─────────────────────────────────────────                    ║
║  TOTAL:                          ~1.1p per minute             ║
║                                                               ║
║  📊 Comparison:                                               ║
║  • Retell AI real cost: 15-25p/min (we're 15x cheaper!)      ║
║  • Vapi real cost: 10-26p/min (we're 10x cheaper!)           ║
║                                                               ║
║  Run: python agent_worker.py dev                              ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check for required API keys
    groq_key = os.getenv("GROQ_API_KEY")
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print("🔑 API Keys Status:")
    print(f"   GROQ_API_KEY:     {'✅ Found (FREE!)' if groq_key else '❌ Not found - Get FREE at https://console.groq.com'}")
    print(f"   DEEPGRAM_API_KEY: {'✅ Found' if deepgram_key else '❌ Not found - Get at https://deepgram.com'}")
    print(f"   OPENAI_API_KEY:   {'⚠️ Found (expensive, not used by default)' if openai_key else '✅ Not found (good - we use FREE alternatives!)'}")
    print()
    
    print("⚙️  Default Providers (COST OPTIMISED):")
    print(f"   LLM: {DEFAULT_LLM_PROVIDER} / {DEFAULT_LLM_MODEL}")
    print(f"   STT: {DEFAULT_STT_PROVIDER} (Whisper - FREE!)")
    print(f"   TTS: {DEFAULT_TTS_PROVIDER} (Aura - cheapest!)")
    print()
    
    if not groq_key:
        print("❌ GROQ_API_KEY is required for FREE LLM and STT!")
        print("   Get your FREE key at: https://console.groq.com")
        print()
        exit(1)
    
    if not deepgram_key:
        print("❌ DEEPGRAM_API_KEY is required for TTS!")
        print("   Get your key at: https://deepgram.com")
        print()
        exit(1)
    
    print("✅ Starting COST-OPTIMISED agent worker...")
    print("💰 Target cost: ~1.1p per minute")
    print()
    
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
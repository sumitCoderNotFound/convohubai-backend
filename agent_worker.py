"""
ConvoHubAI - LiveKit AI Agent Worker (v1.3.12)
Structured Conversation Flow - Always collects Name, Email, Booking Details

100% FREE: Groq LLM + Deepgram STT/TTS
"""

import asyncio
import logging
from dotenv import load_dotenv
import os

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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-agent")


class HotelBookingAssistant(Agent):
    """
    Hotel Front Desk AI Assistant with Structured Conversation Flow
    
    ALWAYS follows this flow:
    1. Greet and ask for NAME
    2. Ask for EMAIL
    3. Ask for CHECK-IN date
    4. Ask for CHECK-OUT date
    5. Ask for ROOM TYPE preference
    6. Ask for NUMBER OF GUESTS
    7. CONFIRM all details
    8. Thank them and provide confirmation
    """
    
    def __init__(self):
        super().__init__(
            instructions="""You are a professional Hotel Front Desk Assistant. You MUST follow a strict conversation flow to collect booking information.

## YOUR CONVERSATION FLOW (FOLLOW THIS EXACTLY):

### Step 1: GREETING
- Greet the guest warmly
- Immediately ask for their FULL NAME
- Example: "Welcome to Grand Hotel! I'd be happy to help you with a reservation. May I have your full name please?"

### Step 2: COLLECT EMAIL
- After getting their name, ALWAYS ask for their EMAIL
- Say: "Thank you [Name]. And what's the best email address to send your confirmation to?"

### Step 3: COLLECT CHECK-IN DATE  
- Ask: "Great! What date would you like to check in?"

### Step 4: COLLECT CHECK-OUT DATE
- Ask: "And what date will you be checking out?"

### Step 5: ROOM PREFERENCE
- Offer room options and ask preference:
- "We have three room types available:
  - Standard Room at $99 per night
  - Deluxe Room at $149 per night  
  - Suite at $249 per night
  Which would you prefer?"

### Step 6: NUMBER OF GUESTS
- Ask: "How many guests will be staying?"

### Step 7: CONFIRM BOOKING
- Repeat ALL details back:
- "Let me confirm your booking:
  - Name: [their name]
  - Email: [their email]
  - Check-in: [date]
  - Check-out: [date]
  - Room: [room type]
  - Guests: [number]
  - Total: [calculated price]
  Is this correct?"

### Step 8: COMPLETION
- If confirmed: "Perfect! Your reservation is confirmed. A confirmation email will be sent to [email]. Is there anything else I can help you with?"
- If changes needed: Help them modify and re-confirm

## IMPORTANT RULES:
1. ALWAYS ask for name first if you don't have it
2. ALWAYS ask for email after name - this is REQUIRED
3. Keep responses SHORT (1-2 sentences max) - this is voice!
4. Be warm, friendly, and professional
5. If they ask other questions, answer briefly then return to the booking flow
6. NEVER skip asking for email - we need it to send confirmation

## HOTEL INFORMATION:
- Check-in: 3:00 PM
- Check-out: 11:00 AM  
- Amenities: Pool, Gym, Restaurant, Free WiFi, Free Parking
- Room Types: Standard ($99), Deluxe ($149), Suite ($249)

Remember: SHORT responses, ALWAYS collect name and email!"""
        )


def prewarm(proc: JobProcess):
    """Preload models for faster startup"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """
    Main entry point for the AI agent.
    """
    logger.info(f"🤖 Agent joining room: {ctx.room.name}")
    
    # Connect to the room
    await ctx.connect()
    
    # Wait for a participant  
    participant = await ctx.wait_for_participant()
    logger.info(f"👤 User joined: {participant.identity}")
    
    # Create the agent session - 100% FREE!
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),  # FREE tier
        llm=groq.LLM(model="llama-3.3-70b-versatile"),  # FREE - current model
        tts=deepgram.TTS(model="aura-asteria-en"),  # FREE tier
    )
    
    # Start the session with our Hotel Booking Assistant
    await session.start(
        room=ctx.room,
        agent=HotelBookingAssistant(),
        room_input_options=RoomInputOptions(),
    )
    
    logger.info("✅ Hotel Booking Assistant started!")
    
    # Send welcome message - immediately ask for name
    await session.say(
        "Welcome to Grand Hotel! I'd be happy to help you make a reservation. May I have your full name please?"
    )


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     ConvoHubAI - Hotel Booking Assistant (100% FREE!)         ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Conversation Flow:                                           ║
║  1. Ask for NAME                                              ║
║  2. Ask for EMAIL                                             ║
║  3. Ask for CHECK-IN date                                     ║
║  4. Ask for CHECK-OUT date                                    ║
║  5. Ask for ROOM TYPE                                         ║
║  6. Ask for NUMBER OF GUESTS                                  ║
║  7. CONFIRM booking                                           ║
║                                                               ║
║  Run: python agent_worker.py dev                              ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check for required API keys
    groq_key = os.getenv("GROQ_API_KEY")
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    
    if not groq_key:
        print("⚠️  GROQ_API_KEY not found!")
        print("   Get FREE key at: https://console.groq.com\n")
        exit(1)
    
    if not deepgram_key:
        print("⚠️  DEEPGRAM_API_KEY not found!")
        print("   Get FREE key at: https://deepgram.com\n")
        exit(1)
    
    print("✅ API keys found! Starting agent...\n")
    
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
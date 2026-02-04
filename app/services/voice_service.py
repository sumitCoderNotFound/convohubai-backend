"""
ConvoHubAI - Voice Service
Handles voice calls using Twilio, Deepgram (STT), and OpenAI (TTS)
"""
import os
import json
import base64
import asyncio
import httpx
from typing import Optional, Dict, Any
from datetime import datetime


class VoiceService:
    """Service for handling voice calls."""
    
    def __init__(self):
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
    # ============================================
    # TWILIO - Phone Number Management
    # ============================================
    
    async def list_available_numbers(self, country: str = "US", area_code: str = None) -> list:
        """List available phone numbers to purchase."""
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/AvailablePhoneNumbers/{country}/Local.json"
        
        params = {"VoiceEnabled": "true", "SmsEnabled": "true"}
        if area_code:
            params["AreaCode"] = area_code
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                params=params,
                auth=(self.twilio_account_sid, self.twilio_auth_token)
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("available_phone_numbers", [])
            else:
                raise Exception(f"Failed to list numbers: {response.text}")
    
    async def buy_phone_number(self, phone_number: str, webhook_url: str) -> dict:
        """Purchase a phone number and configure webhook."""
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/IncomingPhoneNumbers.json"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                data={
                    "PhoneNumber": phone_number,
                    "VoiceUrl": webhook_url,
                    "VoiceMethod": "POST",
                    "StatusCallback": f"{webhook_url}/status",
                    "StatusCallbackMethod": "POST",
                },
                auth=(self.twilio_account_sid, self.twilio_auth_token)
            )
            
            if response.status_code == 201:
                return response.json()
            else:
                raise Exception(f"Failed to buy number: {response.text}")
    
    async def list_owned_numbers(self) -> list:
        """List all owned phone numbers."""
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/IncomingPhoneNumbers.json"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                auth=(self.twilio_account_sid, self.twilio_auth_token)
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("incoming_phone_numbers", [])
            else:
                raise Exception(f"Failed to list owned numbers: {response.text}")
    
    async def update_phone_number_webhook(self, phone_sid: str, webhook_url: str) -> dict:
        """Update webhook URL for a phone number."""
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/IncomingPhoneNumbers/{phone_sid}.json"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                data={
                    "VoiceUrl": webhook_url,
                    "VoiceMethod": "POST",
                },
                auth=(self.twilio_account_sid, self.twilio_auth_token)
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to update webhook: {response.text}")
    
    async def release_phone_number(self, phone_sid: str) -> bool:
        """Release (delete) a phone number."""
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/IncomingPhoneNumbers/{phone_sid}.json"
        
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                url,
                auth=(self.twilio_account_sid, self.twilio_auth_token)
            )
            
            return response.status_code == 204
    
    # ============================================
    # TWILIO - Outbound Calls
    # ============================================
    
    async def make_call(self, to_number: str, from_number: str, webhook_url: str) -> dict:
        """Initiate an outbound call."""
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/Calls.json"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                data={
                    "To": to_number,
                    "From": from_number,
                    "Url": webhook_url,
                    "Method": "POST",
                    "StatusCallback": f"{webhook_url}/status",
                    "StatusCallbackMethod": "POST",
                    "StatusCallbackEvent": ["initiated", "ringing", "answered", "completed"],
                    "Record": "true",
                },
                auth=(self.twilio_account_sid, self.twilio_auth_token)
            )
            
            if response.status_code == 201:
                return response.json()
            else:
                raise Exception(f"Failed to make call: {response.text}")
    
    async def end_call(self, call_sid: str) -> dict:
        """End an active call."""
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/Calls/{call_sid}.json"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                data={"Status": "completed"},
                auth=(self.twilio_account_sid, self.twilio_auth_token)
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to end call: {response.text}")
    
    async def get_call(self, call_sid: str) -> dict:
        """Get call details."""
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/Calls/{call_sid}.json"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                auth=(self.twilio_account_sid, self.twilio_auth_token)
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to get call: {response.text}")
    
    # ============================================
    # DEEPGRAM - Speech to Text
    # ============================================
    
    async def transcribe_audio(self, audio_data: bytes, mime_type: str = "audio/wav") -> str:
        """Transcribe audio using Deepgram."""
        url = "https://api.deepgram.com/v1/listen"
        
        params = {
            "model": "nova-2",
            "smart_format": "true",
            "language": "en",
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                params=params,
                content=audio_data,
                headers={
                    "Authorization": f"Token {self.deepgram_api_key}",
                    "Content-Type": mime_type,
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                transcript = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
                return transcript
            else:
                raise Exception(f"Failed to transcribe: {response.text}")
    
    def get_deepgram_websocket_url(self) -> str:
        """Get Deepgram WebSocket URL for real-time streaming."""
        return f"wss://api.deepgram.com/v1/listen?model=nova-2&encoding=mulaw&sample_rate=8000&channels=1"
    
    def get_deepgram_headers(self) -> dict:
        """Get headers for Deepgram WebSocket connection."""
        return {"Authorization": f"Token {self.deepgram_api_key}"}
    
    # ============================================
    # OPENAI - Text to Speech
    # ============================================
    
    async def text_to_speech(self, text: str, voice: str = "alloy") -> bytes:
        """Convert text to speech using OpenAI TTS."""
        url = "https://api.openai.com/v1/audio/speech"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={
                    "model": "tts-1",
                    "input": text,
                    "voice": voice,  # alloy, echo, fable, onyx, nova, shimmer
                    "response_format": "mp3",
                },
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.content
            else:
                raise Exception(f"Failed to generate speech: {response.text}")
    
    async def text_to_speech_stream(self, text: str, voice: str = "alloy"):
        """Stream text to speech for lower latency."""
        url = "https://api.openai.com/v1/audio/speech"
        
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                json={
                    "model": "tts-1",
                    "input": text,
                    "voice": voice,
                    "response_format": "mp3",
                },
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
    
    # ============================================
    # TWIML Response Generators
    # ============================================
    
    def generate_welcome_twiml(self, welcome_message: str, gather_url: str) -> str:
        """Generate TwiML for initial call greeting with input gathering."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{welcome_message}</Say>
    <Gather input="speech" action="{gather_url}" method="POST" speechTimeout="auto" speechModel="phone_call">
        <Say voice="Polly.Joanna">I'm listening.</Say>
    </Gather>
    <Say voice="Polly.Joanna">I didn't catch that. Goodbye!</Say>
</Response>"""
    
    def generate_response_twiml(self, response_text: str, gather_url: str, end_call: bool = False) -> str:
        """Generate TwiML for AI response."""
        if end_call:
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{response_text}</Say>
    <Hangup/>
</Response>"""
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{response_text}</Say>
    <Gather input="speech" action="{gather_url}" method="POST" speechTimeout="auto" speechModel="phone_call">
    </Gather>
    <Say voice="Polly.Joanna">Are you still there?</Say>
    <Gather input="speech" action="{gather_url}" method="POST" speechTimeout="auto" speechModel="phone_call">
    </Gather>
    <Say voice="Polly.Joanna">Goodbye!</Say>
</Response>"""
    
    def generate_hold_twiml(self, message: str = "Please hold while I process your request.") -> str:
        """Generate TwiML for hold/processing state."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{message}</Say>
    <Pause length="2"/>
</Response>"""


# Create singleton instance
voice_service = VoiceService()
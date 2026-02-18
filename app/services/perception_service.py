"""
ConvoHubAI - Perception & Sentiment Analysis Service
Analyzes caller emotions, sentiment, and engagement during calls
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json
import httpx
from pydantic import BaseModel


class Emotion(str, Enum):
    HAPPY = "happy"
    EXCITED = "excited"
    NEUTRAL = "neutral"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    SAD = "sad"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class EngagementLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MessageAnalysis(BaseModel):
    """Analysis of a single message"""
    text: str
    role: str  # user or assistant
    timestamp: datetime
    sentiment: Sentiment
    emotion: Optional[Emotion] = None
    confidence: float = 0.0
    keywords: List[str] = []
    intent: Optional[str] = None


class ConversationAnalytics(BaseModel):
    """Overall conversation analytics"""
    conversation_id: str
    duration_seconds: int
    message_count: int
    user_message_count: int
    ai_message_count: int
    
    # Sentiment breakdown
    overall_sentiment: Sentiment
    sentiment_scores: Dict[str, float]  # positive, neutral, negative percentages
    
    # Emotion tracking
    dominant_emotion: Emotion
    emotion_timeline: List[Dict[str, Any]]
    
    # Engagement
    engagement_level: EngagementLevel
    avg_response_time_ms: int
    avg_user_word_count: float
    
    # Intent & Outcomes
    detected_intents: List[str]
    topics_discussed: List[str]
    
    # Lead quality (for sales)
    lead_score: int  # 0-100
    lead_indicators: List[str]
    
    # Action items
    follow_up_required: bool
    action_items: List[str]


class PerceptionAnalysisService:
    """
    Service for analyzing caller perception, sentiment, and emotions
    """
    
    def __init__(self):
        self.positive_keywords = [
            "great", "love", "amazing", "perfect", "excellent", "wonderful",
            "fantastic", "awesome", "brilliant", "excited", "happy", "yes",
            "definitely", "absolutely", "thanks", "helpful", "interested"
        ]
        
        self.negative_keywords = [
            "bad", "terrible", "awful", "hate", "angry", "frustrated",
            "confused", "don't understand", "problem", "issue", "wrong",
            "not working", "disappointed", "upset", "annoying", "waste"
        ]
        
        self.buying_intent_keywords = [
            "price", "cost", "how much", "book", "reserve", "sign up",
            "register", "apply", "start", "when can", "available",
            "discount", "offer", "deal", "ready", "let's do it"
        ]
        
        self.confusion_keywords = [
            "what", "how", "don't understand", "confused", "unclear",
            "explain", "what do you mean", "sorry", "repeat", "again"
        ]
    
    def analyze_sentiment(self, text: str) -> tuple[Sentiment, float]:
        """
        Simple rule-based sentiment analysis
        For production, use OpenAI, AWS Comprehend, or similar
        """
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return Sentiment.NEUTRAL, 0.5
        
        positive_ratio = positive_count / total
        
        if positive_ratio > 0.6:
            return Sentiment.POSITIVE, positive_ratio
        elif positive_ratio < 0.4:
            return Sentiment.NEGATIVE, 1 - positive_ratio
        else:
            return Sentiment.NEUTRAL, 0.5
    
    def detect_emotion(self, text: str, sentiment: Sentiment) -> tuple[Emotion, float]:
        """
        Detect emotion from text
        For production, use Hume AI or similar for voice emotion detection
        """
        text_lower = text.lower()
        
        # Check for specific emotion indicators
        if any(word in text_lower for word in ["excited", "amazing", "love", "fantastic"]):
            return Emotion.EXCITED, 0.8
        
        if any(word in text_lower for word in ["happy", "great", "wonderful", "thanks"]):
            return Emotion.HAPPY, 0.7
        
        if any(word in text_lower for word in self.confusion_keywords):
            return Emotion.CONFUSED, 0.7
        
        if any(word in text_lower for word in ["frustrated", "annoying", "problem"]):
            return Emotion.FRUSTRATED, 0.7
        
        if any(word in text_lower for word in ["angry", "terrible", "awful", "hate"]):
            return Emotion.ANGRY, 0.8
        
        # Default based on sentiment
        if sentiment == Sentiment.POSITIVE:
            return Emotion.HAPPY, 0.5
        elif sentiment == Sentiment.NEGATIVE:
            return Emotion.FRUSTRATED, 0.5
        else:
            return Emotion.NEUTRAL, 0.5
    
    def detect_intent(self, text: str) -> List[str]:
        """Detect user intents from text"""
        text_lower = text.lower()
        intents = []
        
        if any(word in text_lower for word in self.buying_intent_keywords):
            intents.append("purchase_intent")
        
        if any(word in text_lower for word in ["help", "support", "issue", "problem"]):
            intents.append("support_request")
        
        if any(word in text_lower for word in ["info", "information", "tell me", "what is", "how does"]):
            intents.append("information_seeking")
        
        if any(word in text_lower for word in ["compare", "difference", "versus", "vs", "better"]):
            intents.append("comparison")
        
        if any(word in text_lower for word in ["cancel", "refund", "return", "stop"]):
            intents.append("cancellation")
        
        if any(word in text_lower for word in ["complaint", "disappointed", "not happy"]):
            intents.append("complaint")
        
        return intents if intents else ["general_inquiry"]
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple extraction - for production use NLP libraries
        stop_words = {"i", "me", "my", "we", "our", "you", "your", "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "and", "but", "if", "or", "because", "as", "until", "while", "it", "this", "that", "these", "those", "am", "what", "which", "who", "whom"}
        
        words = text.lower().split()
        keywords = [word.strip(".,!?;:") for word in words if word.lower() not in stop_words and len(word) > 2]
        
        return list(set(keywords))[:10]  # Return top 10 unique keywords
    
    def analyze_message(self, text: str, role: str) -> MessageAnalysis:
        """Analyze a single message"""
        sentiment, sentiment_confidence = self.analyze_sentiment(text)
        emotion, emotion_confidence = self.detect_emotion(text, sentiment)
        intents = self.detect_intent(text)
        keywords = self.extract_keywords(text)
        
        return MessageAnalysis(
            text=text,
            role=role,
            timestamp=datetime.utcnow(),
            sentiment=sentiment,
            emotion=emotion,
            confidence=max(sentiment_confidence, emotion_confidence),
            keywords=keywords,
            intent=intents[0] if intents else None
        )
    
    def calculate_lead_score(
        self,
        messages: List[MessageAnalysis],
        sentiment_scores: Dict[str, float],
        detected_intents: List[str]
    ) -> tuple[int, List[str]]:
        """
        Calculate lead quality score (0-100)
        Higher = more likely to convert
        """
        score = 50  # Start at neutral
        indicators = []
        
        # Sentiment bonus/penalty
        if sentiment_scores.get("positive", 0) > 0.5:
            score += 15
            indicators.append("Positive sentiment throughout")
        elif sentiment_scores.get("negative", 0) > 0.4:
            score -= 20
            indicators.append("Negative sentiment detected")
        
        # Intent bonuses
        if "purchase_intent" in detected_intents:
            score += 25
            indicators.append("Showed purchase/booking intent")
        
        if "information_seeking" in detected_intents:
            score += 10
            indicators.append("Actively seeking information")
        
        if "complaint" in detected_intents:
            score -= 15
            indicators.append("Has complaints")
        
        # Engagement bonus
        user_messages = [m for m in messages if m.role == "user"]
        if len(user_messages) > 5:
            score += 10
            indicators.append("High engagement (many messages)")
        
        # Word count bonus
        avg_words = sum(len(m.text.split()) for m in user_messages) / max(len(user_messages), 1)
        if avg_words > 10:
            score += 10
            indicators.append("Detailed responses")
        
        # Contact info shared bonus
        all_text = " ".join(m.text for m in user_messages)
        if "@" in all_text:
            score += 15
            indicators.append("Shared email")
        if any(c.isdigit() for c in all_text) and len([c for c in all_text if c.isdigit()]) >= 10:
            score += 10
            indicators.append("Shared phone number")
        
        return max(0, min(100, score)), indicators
    
    def analyze_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        duration_seconds: int = 0
    ) -> ConversationAnalytics:
        """
        Analyze an entire conversation
        """
        # Analyze each message
        analyzed_messages = []
        emotion_timeline = []
        all_intents = []
        all_topics = []
        
        for msg in messages:
            analysis = self.analyze_message(msg.get("content", ""), msg.get("role", "user"))
            analyzed_messages.append(analysis)
            
            emotion_timeline.append({
                "timestamp": analysis.timestamp.isoformat(),
                "emotion": analysis.emotion.value if analysis.emotion else "neutral",
                "sentiment": analysis.sentiment.value,
                "confidence": analysis.confidence
            })
            
            if analysis.intent:
                all_intents.append(analysis.intent)
            all_topics.extend(analysis.keywords)
        
        # Calculate sentiment breakdown
        sentiments = [m.sentiment for m in analyzed_messages if m.role == "user"]
        total_user_msgs = max(len(sentiments), 1)
        sentiment_scores = {
            "positive": sum(1 for s in sentiments if s == Sentiment.POSITIVE) / total_user_msgs,
            "neutral": sum(1 for s in sentiments if s == Sentiment.NEUTRAL) / total_user_msgs,
            "negative": sum(1 for s in sentiments if s == Sentiment.NEGATIVE) / total_user_msgs,
        }
        
        # Determine overall sentiment
        if sentiment_scores["positive"] > sentiment_scores["negative"]:
            overall_sentiment = Sentiment.POSITIVE
        elif sentiment_scores["negative"] > sentiment_scores["positive"]:
            overall_sentiment = Sentiment.NEGATIVE
        else:
            overall_sentiment = Sentiment.NEUTRAL
        
        # Determine dominant emotion
        user_emotions = [m.emotion for m in analyzed_messages if m.role == "user" and m.emotion]
        if user_emotions:
            emotion_counts = {}
            for e in user_emotions:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        else:
            dominant_emotion = Emotion.NEUTRAL
        
        # Calculate engagement
        user_msgs = [m for m in analyzed_messages if m.role == "user"]
        avg_word_count = sum(len(m.text.split()) for m in user_msgs) / max(len(user_msgs), 1)
        
        if avg_word_count > 15 and len(user_msgs) > 5:
            engagement = EngagementLevel.HIGH
        elif avg_word_count > 8 or len(user_msgs) > 3:
            engagement = EngagementLevel.MEDIUM
        else:
            engagement = EngagementLevel.LOW
        
        # Get unique intents and topics
        detected_intents = list(set(all_intents))
        topics_discussed = list(set(all_topics))[:20]
        
        # Calculate lead score
        lead_score, lead_indicators = self.calculate_lead_score(
            analyzed_messages, sentiment_scores, detected_intents
        )
        
        # Determine if follow-up needed
        follow_up_required = (
            "purchase_intent" in detected_intents or
            "complaint" in detected_intents or
            lead_score >= 70
        )
        
        # Extract action items
        action_items = []
        if "purchase_intent" in detected_intents:
            action_items.append("Follow up on purchase/booking interest")
        if "complaint" in detected_intents:
            action_items.append("Address customer complaint")
        if sentiment_scores["negative"] > 0.3:
            action_items.append("Review conversation for improvement opportunities")
        if lead_score >= 80:
            action_items.append("High priority lead - contact within 24 hours")
        
        return ConversationAnalytics(
            conversation_id=conversation_id,
            duration_seconds=duration_seconds,
            message_count=len(messages),
            user_message_count=len(user_msgs),
            ai_message_count=len(messages) - len(user_msgs),
            overall_sentiment=overall_sentiment,
            sentiment_scores=sentiment_scores,
            dominant_emotion=dominant_emotion,
            emotion_timeline=emotion_timeline,
            engagement_level=engagement,
            avg_response_time_ms=0,  # Would need timestamps
            avg_user_word_count=avg_word_count,
            detected_intents=detected_intents,
            topics_discussed=topics_discussed,
            lead_score=lead_score,
            lead_indicators=lead_indicators,
            follow_up_required=follow_up_required,
            action_items=action_items
        )


# ==========================================
# ADVANCED: Hume AI Integration (Voice Emotion)
# ==========================================

class HumeAIService:
    """
    Integration with Hume AI for real-time voice emotion detection
    https://www.hume.ai/
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.hume.ai/v0"
    
    async def analyze_audio_emotion(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Analyze emotions from audio data
        Returns emotions like: joy, sadness, anger, fear, surprise, disgust, contempt
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/batch/jobs",
                headers={"X-Hume-Api-Key": self.api_key},
                files={"file": ("audio.wav", audio_data, "audio/wav")},
                data={"models": json.dumps({"prosody": {}})}  # Voice prosody analysis
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.text}


# ==========================================
# ADVANCED: Symbl.ai Integration (Real-time)
# ==========================================

class SymblAIService:
    """
    Integration with Symbl.ai for real-time conversation intelligence
    https://symbl.ai/
    """
    
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = None
    
    async def get_access_token(self):
        """Get OAuth token"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.symbl.ai/oauth2/token:generate",
                json={
                    "type": "application",
                    "appId": self.app_id,
                    "appSecret": self.app_secret
                }
            )
            data = response.json()
            self.access_token = data.get("accessToken")
            return self.access_token
    
    async def analyze_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get conversation analytics from Symbl.ai
        Includes: topics, action items, follow-ups, sentiment, questions
        """
        if not self.access_token:
            await self.get_access_token()
        
        async with httpx.AsyncClient() as client:
            # Get topics
            topics = await client.get(
                f"https://api.symbl.ai/v1/conversations/{conversation_id}/topics",
                headers={"Authorization": f"Bearer {self.access_token}"}
            )
            
            # Get action items
            actions = await client.get(
                f"https://api.symbl.ai/v1/conversations/{conversation_id}/action-items",
                headers={"Authorization": f"Bearer {self.access_token}"}
            )
            
            # Get sentiment
            messages = await client.get(
                f"https://api.symbl.ai/v1/conversations/{conversation_id}/messages?sentiment=true",
                headers={"Authorization": f"Bearer {self.access_token}"}
            )
            
            return {
                "topics": topics.json() if topics.status_code == 200 else [],
                "action_items": actions.json() if actions.status_code == 200 else [],
                "messages_with_sentiment": messages.json() if messages.status_code == 200 else []
            }


# Singleton instance
perception_service = PerceptionAnalysisService()
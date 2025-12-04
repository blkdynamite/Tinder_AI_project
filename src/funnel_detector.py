"""
Cross-Platform Funnel Detector

This module detects when scammers attempt to move victims off Tinder
to external platforms like Snapchat, Instagram, WhatsApp, or Telegram.
This is a common scam tactic to avoid platform monitoring and reporting.
"""

import re
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossPlatformFunnelDetector:
    """
    Detects attempts to move conversations off-platform to external messaging apps.
    
    Scammers often try to move victims to platforms like Snapchat, Instagram,
    WhatsApp, or Telegram to avoid platform monitoring and make it harder to
    report suspicious behavior.
    """
    
    def __init__(self):
        """
        Initialize the Cross-Platform Funnel Detector.
        
        No parameters required. Sets up internal patterns and configurations.
        """
        # Platform keywords (case-insensitive matching)
        self.platform_keywords = {
            'snapchat': ['snapchat', 'snap', 'sc', 'snap code', 'add me on snap'],
            'instagram': ['instagram', 'ig', 'insta', 'dm me', 'follow me'],
            'whatsapp': ['whatsapp', 'wa', 'whats app', 'what\'s app'],
            'telegram': ['telegram', 'tg', 'tele'],
            'kik': ['kik'],
            'viber': ['viber'],
            'signal': ['signal']
        }
        
        # Velocity indicators - phrases suggesting quick platform switch
        self.velocity_patterns = [
            r'\b(?:can\s+we\s+talk|let\'s\s+talk|let\'s\s+move|let\'s\s+switch|text\s+me\s+on|message\s+me\s+on|hit\s+me\s+up\s+on)\b',
            r'\b(?:move\s+to|switch\s+to|go\s+to|add\s+me|find\s+me\s+on)\b',
            r'\b(?:easier\s+to\s+talk|better\s+on|more\s+active\s+on)\b'
        ]
    
    def detect_snapchat_request(self, message: str, sender_profile: dict) -> dict:
        """
        Detect if a message is attempting to move the conversation off-platform.
        
        Analyzes both the message content and sender profile characteristics
        to determine the likelihood of a funnel attempt.
        
        Args:
            message: The text message to analyze
            sender_profile: Dictionary containing profile information with keys:
                - photo_count (int): Number of photos in profile
                - account_age_days (int): Age of account in days
                - verified (bool): Whether account is verified
        
        Returns:
            Dictionary containing:
                - platform_request_score (float): 0-1 score for platform request detection
                - account_risk_score (float): 0-1 score for account suspiciousness
                - message_velocity_score (float): 0-1 score for urgency/velocity
                - confidence_score (float): 0-1 combined weighted score
                - platform_detected (str or None): Detected platform name or None
        """
        try:
            if not message or not isinstance(message, str):
                logger.warning("Invalid message input for detection")
                return self._empty_result()
            
            message_lower = message.lower()
            
            # 1. Platform Request Detection
            platform_request_score, platform_detected = self._detect_platform_request(message_lower)
            
            # 2. Account Risk Score
            account_risk_score = self._calculate_account_risk(sender_profile)
            
            # 3. Message Velocity Score
            message_velocity_score = self._detect_message_velocity(message_lower)
            
            # 4. Combined Confidence Score
            confidence_score = (
                platform_request_score * 0.5 +
                account_risk_score * 0.25 +
                message_velocity_score * 0.25
            )
            
            return {
                'platform_request_score': round(platform_request_score, 3),
                'account_risk_score': round(account_risk_score, 3),
                'message_velocity_score': round(message_velocity_score, 3),
                'confidence_score': round(confidence_score, 3),
                'platform_detected': platform_detected
            }
            
        except Exception as e:
            logger.error(f"Error in detect_snapchat_request: {e}")
            return self._empty_result()
    
    def _detect_platform_request(self, message_lower: str) -> tuple:
        """
        Detect if message requests moving to an external platform.
        
        Args:
            message_lower: Lowercase message text
        
        Returns:
            Tuple of (score: float, platform: str or None)
        """
        for platform, keywords in self.platform_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return (0.85, platform)
        
        return (0.0, None)
    
    def _calculate_account_risk(self, sender_profile: dict) -> float:
        """
        Calculate risk score based on account characteristics.
        
        Args:
            sender_profile: Dictionary with profile information
        
        Returns:
            Risk score between 0.0 and 1.0
        """
        try:
            score = 0.0
            
            # Photo count risk
            photo_count = sender_profile.get('photo_count', 0)
            if photo_count == 1:
                score += 0.5
            
            # Account age risk
            account_age_days = sender_profile.get('account_age_days', 365)
            if account_age_days < 7:
                score += 0.6
            
            # Verification risk
            verified = sender_profile.get('verified', False)
            if not verified:
                score += 0.3
            
            # Cap at 1.0
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating account risk: {e}")
            return 0.5  # Default to medium risk on error
    
    def _detect_message_velocity(self, message_lower: str) -> float:
        """
        Detect if message suggests urgent/quick platform switch.
        
        Args:
            message_lower: Lowercase message text
        
        Returns:
            Velocity score between 0.0 and 1.0
        """
        try:
            for pattern in self.velocity_patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    return 0.7
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error detecting message velocity: {e}")
            return 0.0
    
    def get_action(self, confidence_score: float) -> dict:
        """
        Determine recommended action based on confidence score.
        
        Args:
            confidence_score: Combined confidence score from detection (0-1)
        
        Returns:
            Dictionary containing:
                - action (str): 'WARN', 'FLAG', or 'ALLOW'
                - confidence (float): The confidence score
                - reasoning (str): Brief explanation of the decision
        """
        try:
            if not isinstance(confidence_score, (int, float)):
                logger.warning(f"Invalid confidence_score type: {type(confidence_score)}")
                confidence_score = 0.0
            
            confidence_score = float(confidence_score)
            
            if confidence_score > 0.85:
                return {
                    'action': 'WARN',
                    'confidence': round(confidence_score, 3),
                    'reasoning': 'High risk funnel attempt detected'
                }
            elif confidence_score > 0.70:
                return {
                    'action': 'FLAG',
                    'confidence': round(confidence_score, 3),
                    'reasoning': 'Medium risk, possible funnel attempt'
                }
            else:
                return {
                    'action': 'ALLOW',
                    'confidence': round(confidence_score, 3),
                    'reasoning': 'No immediate risk detected'
                }
                
        except Exception as e:
            logger.error(f"Error in get_action: {e}")
            return {
                'action': 'ALLOW',
                'confidence': 0.0,
                'reasoning': 'Error in action determination'
            }
    
    def _empty_result(self) -> dict:
        """
        Return empty/default result structure.
        
        Returns:
            Dictionary with default/zero values
        """
        return {
            'platform_request_score': 0.0,
            'account_risk_score': 0.0,
            'message_velocity_score': 0.0,
            'confidence_score': 0.0,
            'platform_detected': None
        }


if __name__ == "__main__":
    # Test the detector
    detector = CrossPlatformFunnelDetector()
    
    # Test case 1: High risk
    test_message = "Hey! Can we talk on snapchat? It's easier there"
    test_profile = {
        'photo_count': 1,
        'account_age_days': 3,
        'verified': False
    }
    
    result = detector.detect_snapchat_request(test_message, test_profile)
    action = detector.get_action(result['confidence_score'])
    
    print("Test Case 1 - High Risk:")
    print(f"Result: {result}")
    print(f"Action: {action}")
    print()
    
    # Test case 2: Low risk
    test_message2 = "Hi, how are you?"
    test_profile2 = {
        'photo_count': 5,
        'account_age_days': 200,
        'verified': True
    }
    
    result2 = detector.detect_snapchat_request(test_message2, test_profile2)
    action2 = detector.get_action(result2['confidence_score'])
    
    print("Test Case 2 - Low Risk:")
    print(f"Result: {result2}")
    print(f"Action: {action2}")


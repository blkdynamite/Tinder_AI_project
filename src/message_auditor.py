import re
import numpy as np
from datetime import datetime, timedelta
from transformers import pipeline
import spacy
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageAuditor:
    def __init__(self):
        """Initialize the message auditor with NLP models"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True
        )

        # Risk patterns for different types of scams
        self.risk_patterns = {
            'financial_scam': {
                'direct_money_requests': [
                    r'\b(?:send|wire|transfer)\s+(?:me\s+)?(?:money|cash|\$|bitcoin|crypto)\b',
                    r'\b(?:can\s+you\s+)?(?:lend|loan|give)\s+me\s+(?:some\s+)?money\b',
                    r'\b(?:i\s+need|please\s+send)\s+(?:\$?\d+|\$?\d+(?:\.\d{2})?)\b'
                ],
                'investment_opportunities': [
                    r'\b(?:investment|invest|opportunity|business|deal)\b',
                    r'\b(?:bitcoin|crypto|stock|trading|profit)\b',
                    r'\b(?:guaranteed\s+return|high\s+yield|quick\s+money)\b'
                ],
                'emergency_stories': [
                    r'\b(?:emergency|accident|hospital|sick|ill|dying)\b',
                    r'\b(?:help|need|please|urgent)\b',
                    r'\b(?:family|mother|father|child|wife|husband)\b'
                ]
            },
            'romance_scam': {
                'love_bombing': [
                    r'\b(?:love|adore|perfect|meant\s+to\s+be|soulmate)\b',
                    r'\b(?:forever|always|together|future)\b',
                    r'\b(?:beautiful|amazing|special|unique)\b'
                ],
                'isolation_tactics': [
                    r'\b(?:secret|private|between\s+us\s+only)\b',
                    r'\b(?:don\'t\s+tell|keep\s+it\s+quiet)\b',
                    r'\b(?:only\s+you\s+understand|you\'re\s+different)\b'
                ]
            },
            'catfishing': {
                'inconsistency_flags': [
                    r'\b(?:actually|really|honestly|truthfully)\b.*\b(?:i\'m|i\s+am)\b',
                    r'\b(?:sorry|apologize|mistake|wrong)\b',
                    r'\b(?:change|different|other)\b.*\b(?:photo|picture)\b'
                ]
            }
        }

        # Escalation indicators
        self.escalation_indicators = {
            'personal_info_requests': [
                r'\b(?:what\'s|what\s+is)\s+your\s+(?:address|phone|email|social)\b',
                r'\b(?:send\s+me|give\s+me)\s+(?:photo|picture|address)\b',
                r'\b(?:where\s+do\s+you\s+live|what\'s\s+your\s+number)\b'
            ],
            'pressure_tactics': [
                r'\b(?:hurry|quick|fast|soon|now|urgent)\b',
                r'\b(?:don\'t\s+have\s+much\s+time|running\s+out\s+of\s+time)\b',
                r'\b(?:before\s+it\'s\s+too\s+late|limited\s+time)\b'
            ],
            'secrecy_requests': [
                r'\b(?:secret|private|confidential|don\'t\s+tell)\b',
                r'\b(?:between\s+us\s+only|just\s+you\s+and\s+me)\b'
            ]
        }

    def analyze_conversation(self, conversation):
        """
        Comprehensive analysis of a conversation thread
        Returns detailed risk assessment and behavioral insights
        """

        messages = conversation.get('messages', [])
        if not messages:
            return self._empty_conversation_analysis()

        analysis = {
            'conversation_id': conversation.get('id'),
            'profile_id': conversation.get('profile_id'),
            'overall_risk_score': 0.0,
            'is_scam_likelihood': 0.0,
            'sentiment_analysis': {},
            'pattern_analysis': {},
            'behavioral_analysis': {},
            'temporal_analysis': {},
            'recommendations': []
        }

        # Analyze message patterns and content
        pattern_analysis = self._analyze_message_patterns(messages)
        analysis['pattern_analysis'] = pattern_analysis

        # Analyze sentiment progression
        sentiment_analysis = self._analyze_sentiment_progression(messages)
        analysis['sentiment_analysis'] = sentiment_analysis

        # Analyze behavioral patterns
        behavioral_analysis = self._analyze_behavioral_patterns(messages)
        analysis['behavioral_analysis'] = behavioral_analysis

        # Analyze temporal patterns
        temporal_analysis = self._analyze_temporal_patterns(messages)
        analysis['temporal_analysis'] = temporal_analysis

        # Calculate overall risk scores
        analysis['overall_risk_score'] = self._calculate_overall_risk(analysis)
        analysis['is_scam_likelihood'] = self._calculate_scam_likelihood(analysis)

        # Generate recommendations
        analysis['recommendations'] = self._generate_conversation_recommendations(analysis)

        return analysis

    def _analyze_message_patterns(self, messages):
        """Analyze messages for risk patterns and suspicious content"""

        analysis = {
            'total_messages': len(messages),
            'their_messages': 0,
            'your_messages': 0,
            'risk_keywords_found': [],
            'pattern_matches': {},
            'content_risk_score': 0.0
        }

        their_messages = []
        your_messages = []

        for msg in messages:
            sender = msg.get('sender', '')
            # Support both formats: 'them'/'me' and profile IDs
            internal_sender = msg.get('_internal_sender', sender)
            if internal_sender == 'them' or (sender != 'me' and sender != 'user_001' and sender):
                # It's from them
                analysis['their_messages'] += 1
                text = msg.get('text', msg.get('content', ''))
                their_messages.append(text)
            else:
                # It's from me
                analysis['your_messages'] += 1
                text = msg.get('text', msg.get('content', ''))
                your_messages.append(text)

        # Analyze their messages for risk patterns
        for category, subcategories in self.risk_patterns.items():
            analysis['pattern_matches'][category] = {}

            for subcategory, patterns in subcategories.items():
                matches = []
                for pattern in patterns:
                    for msg_text in their_messages:
                        found_matches = re.findall(pattern, msg_text.lower(), re.IGNORECASE)
                        if found_matches:
                            matches.extend(found_matches)

                if matches:
                    analysis['pattern_matches'][category][subcategory] = {
                        'count': len(matches),
                        'examples': list(set(matches[:3]))  # Show up to 3 unique examples
                    }
                    analysis['risk_keywords_found'].append({
                        'category': category,
                        'subcategory': subcategory,
                        'matches': matches
                    })

        # Analyze escalation patterns
        escalation_matches = []
        for category, patterns in self.escalation_indicators.items():
            for pattern in patterns:
                for msg_text in their_messages:
                    found_matches = re.findall(pattern, msg_text.lower(), re.IGNORECASE)
                    if found_matches:
                        escalation_matches.extend(found_matches)

        analysis['escalation_indicators'] = {
            'count': len(escalation_matches),
            'examples': list(set(escalation_matches[:5]))
        }

        # Calculate content risk score
        pattern_score = min(len(analysis['risk_keywords_found']) * 0.15, 0.6)
        escalation_score = min(analysis['escalation_indicators']['count'] * 0.1, 0.4)
        analysis['content_risk_score'] = pattern_score + escalation_score

        return analysis

    def _analyze_sentiment_progression(self, messages):
        """Analyze how sentiment changes throughout the conversation"""

        analysis = {
            'sentiment_scores': [],
            'overall_sentiment': 0.5,
            'sentiment_volatility': 0.0,
            'sentiment_trend': 'stable',
            'manipulative_patterns': []
        }

        sentiments = []

        for msg in messages:
            text = msg.get('text', msg.get('content', ''))

            try:
                # Get sentiment scores
                sentiment_results = self.sentiment_analyzer(text[:512])

                # Convert to numerical score (0=negative, 0.5=neutral, 1=positive)
                sentiment_score = 0
                for result in sentiment_results[0]:
                    if result['label'] == 'LABEL_2':  # Positive
                        sentiment_score += result['score'] * 1.0
                    elif result['label'] == 'LABEL_0':  # Negative
                        sentiment_score += result['score'] * 0.0
                    else:  # Neutral
                        sentiment_score += result['score'] * 0.5

                sentiments.append(sentiment_score)

            except Exception as e:
                logger.warning(f"Sentiment analysis failed for message: {e}")
                sentiments.append(0.5)  # Default to neutral

        analysis['sentiment_scores'] = sentiments
        analysis['overall_sentiment'] = np.mean(sentiments) if sentiments else 0.5

        if len(sentiments) > 1:
            # Calculate volatility (standard deviation)
            analysis['sentiment_volatility'] = np.std(sentiments)

            # Analyze trend
            early_sentiment = np.mean(sentiments[:len(sentiments)//3])
            late_sentiment = np.mean(sentiments[-len(sentiments)//3:])

            if late_sentiment > early_sentiment + 0.2:
                analysis['sentiment_trend'] = 'improving'
            elif late_sentiment < early_sentiment - 0.2:
                analysis['sentiment_trend'] = 'declining'
            else:
                analysis['sentiment_trend'] = 'stable'

            # Check for manipulative patterns (very positive after negative)
            for i in range(1, len(sentiments)):
                if sentiments[i-1] < 0.3 and sentiments[i] > 0.8:
                    analysis['manipulative_patterns'].append(f'Sudden positivity shift at message {i+1}')

        return analysis

    def _analyze_behavioral_patterns(self, messages):
        """Analyze behavioral patterns in the conversation"""

        analysis = {
            'message_frequency': {},
            'response_times': [],
            'conversation_balance': 0.5,  # Ratio of their messages to total
            'consistency_score': 0.8,
            'engagement_patterns': []
        }

        if not messages:
            return analysis

        # Message frequency analysis
        timestamps = []
        for msg in messages:
            try:
                ts = datetime.fromisoformat(msg.get('timestamp', '').replace('Z', '+00:00'))
                timestamps.append(ts)
            except:
                continue

        if len(timestamps) > 1:
            timestamps.sort()
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600  # hours
                         for i in range(len(timestamps)-1)]
            analysis['response_times'] = time_diffs

            # Analyze response patterns
            avg_response_time = np.mean(time_diffs)
            analysis['message_frequency'] = {
                'average_response_hours': avg_response_time,
                'fastest_response_hours': min(time_diffs),
                'slowest_response_hours': max(time_diffs)
            }

        # Conversation balance
        their_count = sum(1 for msg in messages 
                         if msg.get('_internal_sender') == 'them' or 
                         (msg.get('sender') != 'me' and msg.get('sender') != 'user_001' and msg.get('sender')))
        total_count = len(messages)
        analysis['conversation_balance'] = their_count / total_count if total_count > 0 else 0.5

        # Engagement patterns
        if analysis['conversation_balance'] > 0.8:
            analysis['engagement_patterns'].append('One-sided conversation (they send most messages)')
        elif analysis['conversation_balance'] < 0.3:
            analysis['engagement_patterns'].append('Low engagement from them')

        # Check for scripted responses (similar message lengths)
        lengths = [len(msg.get('text', msg.get('content', ''))) 
                  for msg in messages 
                  if msg.get('_internal_sender') == 'them' or 
                  (msg.get('sender') != 'me' and msg.get('sender') != 'user_001' and msg.get('sender'))]
        if lengths and len(lengths) > 3:
            length_std = np.std(lengths)
            length_mean = np.mean(lengths)
            if length_std / length_mean < 0.3:  # Low variance in message lengths
                analysis['engagement_patterns'].append('Unnatural message length consistency')

        return analysis

    def _analyze_temporal_patterns(self, messages):
        """Analyze timing patterns in the conversation"""

        analysis = {
            'conversation_duration_days': 0,
            'active_hours': [],
            'weekend_preference': False,
            'urgency_patterns': []
        }

        timestamps = []
        for msg in messages:
            try:
                ts = datetime.fromisoformat(msg.get('timestamp', '').replace('Z', '+00:00'))
                timestamps.append(ts)
            except:
                continue

        if len(timestamps) >= 2:
            timestamps.sort()
            duration = (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600)
            analysis['conversation_duration_days'] = duration

            # Analyze active hours
            hours = [ts.hour for ts in timestamps]
            analysis['active_hours'] = list(set(hours))  # Unique hours

            # Check for unusual timing (very late/early hours)
            unusual_hours = [h for h in hours if h < 6 or h > 22]
            if unusual_hours:
                analysis['urgency_patterns'].append('Messages sent at unusual hours')

            # Weekend preference
            weekend_messages = sum(1 for ts in timestamps if ts.weekday() >= 5)
            if weekend_messages / len(timestamps) > 0.7:
                analysis['weekend_preference'] = True

        return analysis

    def _calculate_overall_risk(self, analysis):
        """Calculate overall risk score from all analysis components"""

        weights = {
            'content_risk': 0.4,
            'sentiment_volatility': 0.2,
            'behavioral_imbalance': 0.2,
            'temporal_suspicion': 0.2
        }

        # Content risk
        content_risk = analysis.get('pattern_analysis', {}).get('content_risk_score', 0)

        # Sentiment risk (high volatility = higher risk)
        sentiment_volatility = analysis.get('sentiment_analysis', {}).get('sentiment_volatility', 0)
        sentiment_risk = min(sentiment_volatility * 2, 1.0)

        # Behavioral risk (imbalanced conversations are riskier)
        balance = analysis.get('behavioral_analysis', {}).get('conversation_balance', 0.5)
        behavioral_risk = abs(balance - 0.5) * 1.5  # Deviation from 50/50 balance

        # Temporal risk
        temporal_risk = 0
        temporal = analysis.get('temporal_analysis', {})
        if temporal.get('urgency_patterns'):
            temporal_risk += 0.3
        if temporal.get('weekend_preference'):
            temporal_risk += 0.1

        overall_score = (
            content_risk * weights['content_risk'] +
            sentiment_risk * weights['sentiment_volatility'] +
            behavioral_risk * weights['behavioral_imbalance'] +
            temporal_risk * weights['temporal_suspicion']
        )

        return min(overall_score, 1.0)

    def _calculate_scam_likelihood(self, analysis):
        """Calculate specific scam likelihood based on patterns"""

        scam_indicators = 0
        total_indicators = 0

        # Financial scam indicators
        pattern_analysis = analysis.get('pattern_analysis', {})
        financial_patterns = pattern_analysis.get('pattern_matches', {}).get('financial_scam', {})
        scam_indicators += len(financial_patterns)
        total_indicators += 3  # Expected financial scam patterns

        # Escalation indicators
        escalation_count = pattern_analysis.get('escalation_indicators', {}).get('count', 0)
        scam_indicators += min(escalation_count, 3)  # Cap at 3
        total_indicators += 3

        # Behavioral red flags
        behavioral = analysis.get('behavioral_analysis', {})
        if behavioral.get('conversation_balance', 0.5) > 0.8:
            scam_indicators += 1
        if len(behavioral.get('engagement_patterns', [])) > 0:
            scam_indicators += 1
        total_indicators += 2

        return scam_indicators / total_indicators if total_indicators > 0 else 0

    def _generate_conversation_recommendations(self, analysis):
        """Generate recommendations based on conversation analysis"""

        recommendations = []
        risk_score = analysis.get('overall_risk_score', 0)
        scam_likelihood = analysis.get('is_scam_likelihood', 0)

        if risk_score > 0.7 or scam_likelihood > 0.6:
            recommendations.extend([
                "High risk conversation - exercise extreme caution",
                "Do not share financial information or send money",
                "Consider reporting this conversation",
                "Avoid sharing personal contact details"
            ])
        elif risk_score > 0.4 or scam_likelihood > 0.3:
            recommendations.extend([
                "Moderate risk - proceed with caution",
                "Verify their identity through other means",
                "Avoid sharing sensitive personal information",
                "Consider meeting in public places only"
            ])
        else:
            recommendations.extend([
                "Conversation appears normal",
                "Continue exercising standard online safety precautions"
            ])

        # Specific recommendations based on patterns
        pattern_analysis = analysis.get('pattern_analysis', {})
        if pattern_analysis.get('escalation_indicators', {}).get('count', 0) > 2:
            recommendations.append("They are escalating quickly - this is a red flag")

        sentiment = analysis.get('sentiment_analysis', {})
        if sentiment.get('sentiment_volatility', 0) > 0.3:
            recommendations.append("Erratic emotional patterns detected")

        return recommendations

    def _empty_conversation_analysis(self):
        """Return default analysis for empty conversations"""

        return {
            'overall_risk_score': 0.0,
            'is_scam_likelihood': 0.0,
            'sentiment_analysis': {'overall_sentiment': 0.5},
            'pattern_analysis': {'total_messages': 0},
            'behavioral_analysis': {},
            'temporal_analysis': {},
            'recommendations': ['No messages to analyze']
        }

    def audit_conversation(self, conversation):
        """
        Wrapper method for app.py compatibility
        Returns formatted audit results matching expected interface
        """
        analysis = self.analyze_conversation(conversation)
        
        # Format for app.py interface
        audit_result = {
            'risk_score': analysis.get('overall_risk_score', 0),
            'analysis': '. '.join(analysis.get('insights', [])) if analysis.get('insights') else 'No analysis available',
            'flagged_messages': []
        }
        
        # Extract flagged messages from pattern analysis
        pattern_analysis = analysis.get('pattern_analysis', {})
        risk_keywords = pattern_analysis.get('risk_keywords_found', [])
        
        messages = conversation.get('messages', [])
        for msg in messages:
            sender = msg.get('sender', '')
            internal_sender = msg.get('_internal_sender', sender)
            if internal_sender == 'them' or (sender != 'me' and sender != 'user_001' and sender):
                text = msg.get('text', msg.get('content', ''))
                # Check if message contains risk keywords
                for keyword_group in risk_keywords:
                    for match in keyword_group.get('matches', []):
                        if match.lower() in text.lower():
                            audit_result['flagged_messages'].append({
                                'message': text,
                                'reason': f"{keyword_group.get('category', 'Unknown')} - {keyword_group.get('subcategory', 'pattern')}"
                            })
                            break

        # Add escalation indicators as flagged messages
        escalation = pattern_analysis.get('escalation_indicators', {})
        if escalation.get('count', 0) > 0:
            for msg in messages:
                sender = msg.get('sender', '')
                internal_sender = msg.get('_internal_sender', sender)
                if internal_sender == 'them' or (sender != 'me' and sender != 'user_001' and sender):
                    text = msg.get('text', msg.get('content', ''))
                    for example in escalation.get('examples', []):
                        if example.lower() in text.lower():
                            audit_result['flagged_messages'].append({
                                'message': text,
                                'reason': 'Escalation indicator detected'
                            })
                            break
        
        return audit_result

    def batch_analyze_conversations(self, conversations):
        """Analyze multiple conversations efficiently"""

        results = []
        for conversation in conversations:
            try:
                analysis = self.analyze_conversation(conversation)
                results.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze conversation {conversation.get('id')}: {e}")
                results.append(self._empty_conversation_analysis())

        return results

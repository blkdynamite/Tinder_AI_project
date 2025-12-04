import cv2
import numpy as np
import requests
from PIL import Image
import io
import spacy
from transformers import pipeline
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfileScanner:
    def __init__(self):
        """Initialize the profile scanner with ML models"""
        try:
            # Load spaCy model for NLP analysis
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

        # Risk keywords and patterns
        self.risk_patterns = {
            'financial_scam': [
                r'\b(?:money|cash|bitcoin|crypto|wire|transfer|send|loan|investment)\b',
                r'\b(?:lottery|prize|winner|inheritance|refund|grant)\b',
                r'\b(?:emergency|hospital|accident|help needed)\b'
            ],
            'romance_scam': [
                r'\b(?:love at first sight|meant to be|perfect match)\b',
                r'\b(?:military|overseas|deployed|tour)\b',
                r'\b(?:widow|divorced|single parent)\b'
            ],
            'catfishing': [
                r'\b(?:model|actor|celebrity|rich|famous)\b',
                r'\b(?:too good to be true|dream come true)\b'
            ]
        }

        self.suspicious_locations = [
            'nigeria', 'ghana', 'kenya', 'ivory coast', 'senegal',
            'london', 'toronto', 'vancouver', 'miami'
        ]

    def analyze_profile(self, profile):
        """
        Comprehensive analysis of a Tinder profile
        Returns detailed risk assessment and insights
        """

        analysis = {
            'profile_id': profile.get('id'),
            'overall_risk_score': 0.0,
            'risk_factors': [],
            'insights': [],
            'recommendations': []
        }

        # Analyze text content (bio, name, etc.)
        text_analysis = self._analyze_text_content(profile)
        analysis.update(text_analysis)

        # Analyze profile photos
        photo_analysis = self._analyze_photos(profile)
        analysis['photo_analysis'] = photo_analysis

        # Analyze demographic data
        demo_analysis = self._analyze_demographics(profile)
        analysis.update(demo_analysis)

        # Calculate overall risk score
        analysis['overall_risk_score'] = self._calculate_overall_risk(analysis)

        # Generate insights and recommendations
        analysis['insights'] = self._generate_insights(analysis)
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _analyze_text_content(self, profile):
        """Analyze text content for risk indicators"""

        text_content = ""
        if profile.get('bio'):
            text_content += profile['bio'] + " "
        if profile.get('occupation'):
            text_content += profile['occupation'] + " "
        if profile.get('name'):
            text_content += profile['name'] + " "

        analysis = {
            'text_risk_score': 0.0,
            'sentiment_score': 0.5,
            'keyword_matches': [],
            'linguistic_patterns': []
        }

        if not text_content.strip():
            return analysis

        text_lower = text_content.lower()

        # Check for risk keywords
        for category, patterns in self.risk_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    analysis['keyword_matches'].extend([{
                        'category': category,
                        'pattern': pattern,
                        'matches': matches
                    }])

        # Sentiment analysis
        try:
            sentiment_results = self.sentiment_analyzer(text_content[:512])  # Limit input length
            # Calculate weighted sentiment score
            sentiment_score = 0
            for result in sentiment_results[0]:
                if result['label'] == 'LABEL_2':  # Positive
                    sentiment_score += result['score'] * 0.8
                elif result['label'] == 'LABEL_0':  # Negative
                    sentiment_score += result['score'] * 0.2
                else:  # Neutral
                    sentiment_score += result['score'] * 0.5
            analysis['sentiment_score'] = sentiment_score
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            analysis['sentiment_score'] = 0.5

        # spaCy analysis for linguistic patterns
        doc = self.nlp(text_content)

        # Check for excessive capitalization (scam indicator)
        caps_ratio = sum(1 for token in doc if token.text.isupper() and len(token.text) > 1) / len(doc)
        if caps_ratio > 0.1:
            analysis['linguistic_patterns'].append('excessive_capitalization')

        # Check for repeated punctuation
        repeated_punct = re.findall(r'[!?]{2,}', text_content)
        if repeated_punct:
            analysis['linguistic_patterns'].append('emotional_manipulation')

        # Calculate text risk score
        keyword_risk = min(len(analysis['keyword_matches']) * 0.2, 0.8)
        pattern_risk = len(analysis['linguistic_patterns']) * 0.1
        analysis['text_risk_score'] = min(keyword_risk + pattern_risk, 1.0)

        return analysis

    def _analyze_photos(self, profile):
        """Analyze profile photos for authenticity indicators"""

        analysis = {
            'photo_count': len(profile.get('photos', [])),
            'photo_quality_score': 0.5,
            'authenticity_indicators': [],
            'risk_factors': []
        }

        photos = profile.get('photos', [])
        if not photos:
            analysis['risk_factors'].append('no_photos')
            return analysis

        # Basic photo analysis
        if len(photos) < 3:
            analysis['risk_factors'].append('few_photos')
            analysis['photo_quality_score'] -= 0.2

        if len(photos) > 10:
            analysis['risk_factors'].append('excessive_photos')
            analysis['authenticity_indicators'].append('professional_modeling')

        # Simulate advanced image analysis (in real implementation, would use ML models)
        for i, photo_url in enumerate(photos[:3]):  # Analyze first 3 photos
            try:
                # In a real implementation, this would download and analyze images
                # For demo purposes, we'll simulate analysis
                analysis_result = self._simulate_photo_analysis(photo_url, i)
                analysis['authenticity_indicators'].extend(analysis_result.get('indicators', []))
                analysis['risk_factors'].extend(analysis_result.get('risks', []))
                analysis['photo_quality_score'] += analysis_result.get('quality_boost', 0)

            except Exception as e:
                logger.warning(f"Failed to analyze photo {photo_url}: {e}")

        analysis['photo_quality_score'] = max(0, min(1, analysis['photo_quality_score']))

        return analysis

    def _simulate_photo_analysis(self, photo_url, index):
        """Simulate photo analysis (would use OpenCV/ML in real implementation)"""

        # Random but realistic simulation
        results = {
            'indicators': [],
            'risks': [],
            'quality_boost': 0
        }

        # Simulate different photo types
        photo_types = ['profile', 'body', 'group', 'stock_photo', 'professional']
        photo_type = np.random.choice(photo_types, p=[0.4, 0.3, 0.15, 0.1, 0.05])

        if photo_type == 'stock_photo':
            results['risks'].append('stock_photo_detected')
            results['quality_boost'] -= 0.3
        elif photo_type == 'professional':
            results['indicators'].append('professional_photography')
            results['quality_boost'] += 0.2
        elif photo_type == 'group':
            results['indicators'].append('social_proof')
            results['quality_boost'] += 0.1

        # Random quality variations
        quality_variations = np.random.normal(0, 0.1)
        results['quality_boost'] += quality_variations

        return results

    def _analyze_demographics(self, profile):
        """Analyze demographic data for risk patterns"""

        analysis = {
            'demographic_risk_score': 0.0,
            'location_risk': False,
            'age_risk': False,
            'verification_status': profile.get('verified', False)
        }

        # Location analysis
        location = profile.get('location', '').lower()
        for suspicious_loc in self.suspicious_locations:
            if suspicious_loc in location:
                analysis['location_risk'] = True
                analysis['demographic_risk_score'] += 0.3
                break

        # Age analysis (very young or old profiles can be riskier)
        age = profile.get('age', 25)
        if age < 20 or age > 50:
            analysis['age_risk'] = True
            analysis['demographic_risk_score'] += 0.1

        # Verification bonus
        if not analysis['verification_status']:
            analysis['demographic_risk_score'] += 0.2

        return analysis

    def _calculate_overall_risk(self, analysis):
        """Calculate overall risk score from all analysis components"""

        weights = {
            'text_risk_score': 0.4,
            'demographic_risk_score': 0.3,
            'photo_quality_score': 0.3  # Inverted - lower quality = higher risk
        }

        overall_score = (
            analysis.get('text_risk_score', 0) * weights['text_risk_score'] +
            analysis.get('demographic_risk_score', 0) * weights['demographic_risk_score'] +
            (1 - analysis.get('photo_analysis', {}).get('photo_quality_score', 0.5)) * weights['photo_quality_score']
        )

        return min(overall_score, 1.0)

    def _generate_insights(self, analysis):
        """Generate human-readable insights from analysis"""

        insights = []

        risk_score = analysis.get('overall_risk_score', 0)

        if risk_score < 0.3:
            insights.append("Profile appears to be low risk based on available data")
        elif risk_score < 0.7:
            insights.append("Profile shows some concerning patterns that warrant caution")
        else:
            insights.append("Profile exhibits multiple high-risk indicators")

        # Specific insights
        text_analysis = analysis.get('text_risk_score', 0)
        if text_analysis > 0.5:
            insights.append("Text content contains suspicious keywords or patterns")

        photo_analysis = analysis.get('photo_analysis', {})
        if photo_analysis.get('photo_count', 0) < 3:
            insights.append("Limited number of photos may indicate incomplete profile")

        if not analysis.get('verification_status', False):
            insights.append("Profile is not verified, which is common among suspicious accounts")

        return insights

    def _generate_recommendations(self, analysis):
        """Generate safety recommendations based on analysis"""

        recommendations = []

        risk_score = analysis.get('overall_risk_score', 0)

        if risk_score > 0.7:
            recommendations.extend([
                "Do not send money or share financial information",
                "Avoid sharing personal contact details",
                "Report suspicious behavior to Tinder",
                "Consider blocking this profile"
            ])
        elif risk_score > 0.3:
            recommendations.extend([
                "Take conversations slowly and stay cautious",
                "Verify information through other means when possible",
                "Avoid sharing sensitive personal information",
                "Consider meeting in public places only"
            ])
        else:
            recommendations.extend([
                "Profile appears safe, but always practice caution online",
                "Continue normal conversation while staying aware"
            ])

        # Photo-specific recommendations
        photo_count = analysis.get('photo_analysis', {}).get('photo_count', 0)
        if photo_count < 3:
            recommendations.append("Ask for additional photos to verify identity")

        return recommendations

    def batch_analyze_profiles(self, profiles):
        """Analyze multiple profiles efficiently"""

        results = []
        for profile in profiles:
            try:
                analysis = self.analyze_profile(profile)
                results.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze profile {profile.get('id')}: {e}")
                results.append({
                    'profile_id': profile.get('id'),
                    'error': str(e),
                    'overall_risk_score': 0.5  # Default to medium risk on error
                })

        return results

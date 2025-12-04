import json
import random
from faker import Faker
from pathlib import Path
import numpy as np

class TinderDataGenerator:
    def __init__(self):
        self.fake = Faker()
        # Risk patterns for scam detection
        self.scam_indicators = {
            'high_risk_phrases': [
                'send money', 'wire transfer', 'bitcoin', 'crypto', 'investment opportunity',
                'i need help', 'emergency', 'hospital', 'accident', 'inheritance',
                'lottery winner', 'prize money', 'tax refund', 'government grant'
            ],
            'suspicious_questions': [
                'can you send me money?', 'will you help me?', 'can you wire money?',
                'do you have bitcoin?', 'want to invest?', 'can you loan me?'
            ],
            'red_flags': [
                'too perfect profile', 'no social media', 'foreign location',
                'age discrepancy', 'inconsistent story', 'pressure tactics'
            ]
        }

    def generate_profile(self, risk_level='low'):
        """Generate a realistic Tinder profile with optional risk indicators"""

        # Base profile data
        profile_id = self.fake.uuid4()
        profile = {
            'id': profile_id,
            'profile_id': profile_id,  # Alias for app.py compatibility
            'name': self.fake.first_name(),
            'age': random.randint(18, 45),
            'location': self.fake.city(),
            'occupation': self.fake.job(),
            'bio': self.fake.text(max_nb_chars=200),
            'photos': [f"https://picsum.photos/400/600?random={i}" for i in range(random.randint(3, 8))],
            'interests': random.sample([
                'travel', 'music', 'food', 'fitness', 'reading', 'movies',
                'hiking', 'cooking', 'art', 'technology', 'sports', 'pets'
            ], random.randint(3, 6)),
            'education': random.choice(['High School', 'Bachelor\'s', 'Master\'s', 'PhD', None]),
            'height': f"{random.randint(150, 200)}cm",
            'verified': random.choice([True, False, False]),  # Bias toward unverified
        }

        # Add risk elements based on risk_level
        if risk_level == 'medium':
            # Add some suspicious elements
            profile['bio'] += " " + random.choice(self.scam_indicators['high_risk_phrases'])
            profile['location'] = self.fake.city() + ", " + random.choice(['Nigeria', 'Ghana', 'Kenya', 'UK', 'Canada'])
            profile['occupation'] = random.choice(['Business Owner', 'Oil Executive', 'Investor', 'Entrepreneur'])

        elif risk_level == 'high':
            # Add multiple red flags
            scam_phrase = random.choice(self.scam_indicators['high_risk_phrases'])
            profile['bio'] = f"Hi! I'm {profile['name']}. {scam_phrase}. Looking for someone special to share this with."
            profile['location'] = random.choice(['Lagos, Nigeria', 'Accra, Ghana', 'London, UK', 'Toronto, Canada'])
            profile['occupation'] = random.choice(['Oil & Gas Executive', 'Business Consultant', 'Investor'])
            profile['verified'] = False
            profile['age'] = random.randint(35, 55)  # Older profiles more suspicious
            profile['photos'] = [f"https://picsum.photos/400/600?random={random.randint(100, 200)}"] * 2  # Few photos

        # Calculate risk score
        profile['risk_score'] = self._calculate_risk_score(profile)
        profile['risk_category'] = self._categorize_risk(profile['risk_score'])

        return profile

    def generate_conversation(self, profile, is_scam=False):
        """Generate a conversation thread for a profile"""

        conversation = {
            'id': self.fake.uuid4(),
            'profile_id': profile['id'],
            'participants': [profile['id'], 'user_001'],  # Add participants for app.py compatibility
            'messages': [],
            'duration_days': random.randint(1, 30),
            'total_messages': random.randint(5, 50),
            'is_scam': is_scam,
            'sentiment_score': 0.5,  # Will be calculated
            'escalation_score': 0.0  # Will be calculated
        }

        # Generate message thread
        messages = []
        current_time = self.fake.date_time_this_month()

        for i in range(conversation['total_messages']):
            sender = 'them' if random.random() > 0.4 else 'me'  # They send more messages
            message_text = self._generate_message(sender, is_scam, i, conversation['total_messages'])
            messages.append({
                'timestamp': current_time.isoformat(),
                'sender': profile['id'] if sender == 'them' else 'user_001',  # Use profile ID for 'them', user ID for 'me'
                'text': message_text,  # For message_auditor compatibility (uses 'sender' == 'them')
                'content': message_text,  # For app.py compatibility
                'sentiment': self._analyze_sentiment(message_text),
                '_internal_sender': sender  # Keep 'them'/'me' for message_auditor
            })
            # Add some time delay
            current_time = self.fake.date_time_between(start_date=current_time, end_date='+1d')

        conversation['messages'] = messages
        conversation['sentiment_score'] = np.mean([msg['sentiment'] for msg in messages])
        conversation['escalation_score'] = self._calculate_escalation_score(messages)

        return conversation

    def _generate_message(self, sender, is_scam, message_index, total_messages):
        """Generate a realistic message"""

        if sender == 'me':
            # User's responses - mix of interested, cautious, etc.
            responses = [
                "Hi! How are you?",
                "That sounds interesting!",
                "Tell me more about yourself",
                "What do you do for work?",
                "Where are you from?",
                "What are your interests?",
                "That seems too good to be true",
                "I need to be careful online",
                "Let me think about this",
                "Maybe we should meet in person first"
            ]
            return random.choice(responses)

        else:  # Their messages
            if is_scam:
                # Scam conversation progression
                scam_stages = [
                    # Early stage - building rapport
                    ["Hey there! 😊", "How's your day going?", "You look amazing in your photos!"],
                    # Middle stage - personal questions
                    ["So what do you do?", "Where do you live?", "Are you single?"],
                    # Late stage - scam reveal
                    ["I have this amazing opportunity", "I need your help with something", "Can you send me money?"]
                ]

                stage = min(message_index // 5, len(scam_stages) - 1)
                return random.choice(scam_stages[stage])
            else:
                # Normal conversation
                normal_messages = [
                    "Hi! I'm doing well, thanks for asking!",
                    "I'm a software engineer from Seattle",
                    "I love hiking and trying new restaurants",
                    "What about you? What do you enjoy?",
                    "That sounds fun! Want to grab coffee sometime?",
                    "I'm really enjoying talking to you",
                    "Tell me about your hobbies",
                    "I'm originally from Portland",
                    "I work in marketing at a tech company",
                    "I have two dogs that I adore"
                ]
                return random.choice(normal_messages)

    def _calculate_risk_score(self, profile):
        """Calculate risk score based on profile characteristics"""

        score = 0.0

        # Age factor (older profiles slightly more risky)
        if profile['age'] > 35:
            score += 0.1

        # Verification factor
        if not profile.get('verified', False):
            score += 0.2

        # Location factor (certain locations are higher risk)
        high_risk_locations = ['nigeria', 'ghana', 'kenya', 'uk', 'canada']
        if any(loc.lower() in profile['location'].lower() for loc in high_risk_locations):
            score += 0.3

        # Bio content analysis
        bio_text = profile.get('bio', '').lower()
        for phrase in self.scam_indicators['high_risk_phrases']:
            if phrase in bio_text:
                score += 0.4

        # Occupation factor
        high_risk_jobs = ['business owner', 'investor', 'executive', 'consultant']
        if any(job.lower() in profile.get('occupation', '').lower() for job in high_risk_jobs):
            score += 0.2

        # Photo count (few photos = higher risk)
        if len(profile.get('photos', [])) < 3:
            score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    def _categorize_risk(self, score):
        """Categorize risk level"""
        if score < 0.3:
            return 'low'
        elif score < 0.7:
            return 'medium'
        else:
            return 'high'

    def _analyze_sentiment(self, text):
        """Simple sentiment analysis (positive/negative)"""
        positive_words = ['great', 'amazing', 'love', 'wonderful', 'fantastic', 'excited', 'happy']
        negative_words = ['sad', 'angry', 'disappointed', 'worried', 'scared', 'hate', 'terrible']

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return 0.8
        elif neg_count > pos_count:
            return 0.2
        else:
            return 0.5

    def _calculate_escalation_score(self, messages):
        """Calculate how quickly the conversation escalates to personal/financial topics"""
        escalation_indicators = ['money', 'send', 'wire', 'help', 'emergency', 'investment']
        score = 0.0

        for i, msg in enumerate(messages):
            sender = msg.get('_internal_sender', msg.get('sender', ''))
            if sender == 'them':  # Only count their escalation attempts
                text_lower = msg.get('text', msg.get('content', '')).lower()
                if any(indicator in text_lower for indicator in escalation_indicators):
                    # Earlier in conversation = higher escalation score
                    score += (1.0 - (i / len(messages))) * 0.3

        return min(score, 1.0)

    def generate_dataset(self, num_profiles=50):
        """Generate a complete dataset of profiles and conversations"""

        profiles = []
        conversations = []

        # Generate mix of risk levels
        risk_distribution = {'low': 0.6, 'medium': 0.3, 'high': 0.1}

        for _ in range(num_profiles):
            risk_level = random.choices(
                list(risk_distribution.keys()),
                weights=list(risk_distribution.values())
            )[0]

            profile = self.generate_profile(risk_level)
            profiles.append(profile)

            # Generate 1-3 conversations per profile
            num_convos = random.randint(1, 3)
            for _ in range(num_convos):
                is_scam = (risk_level == 'high' and random.random() < 0.8) or \
                         (risk_level == 'medium' and random.random() < 0.3)
                conversation = self.generate_conversation(profile, is_scam)
                conversations.append(conversation)

        return {
            'profiles': profiles,
            'conversations': conversations,
            'generated_at': self.fake.date_time_this_month().isoformat(),
            'total_profiles': len(profiles),
            'total_conversations': len(conversations)
        }

    def save_to_file(self, data, filename='demo_tinder_data.json'):
        """Save generated data to JSON file"""
        output_path = Path('data') / filename
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Data saved to {output_path}")

def generate_synthetic_data(num_profiles=50, num_conversations=None, include_risky=True, include_scams=True):
    """
    Wrapper function to generate synthetic Tinder data
    Compatible with app.py interface
    
    Args:
        num_profiles: Number of profiles to generate
        num_conversations: Number of conversations (if None, auto-generated based on profiles)
        include_risky: Whether to include risky profiles (always True, handled by risk distribution)
        include_scams: Whether to include scam conversations (always True, handled by risk distribution)
    
    Returns:
        Dictionary with 'profiles' and 'conversations' keys
    """
    generator = TinderDataGenerator()
    
    # Generate dataset (risk distribution is handled internally)
    data = generator.generate_dataset(num_profiles)
    
    # If specific number of conversations requested, adjust
    if num_conversations is not None and len(data['conversations']) != num_conversations:
        # Trim or extend conversations to match requested count
        if len(data['conversations']) > num_conversations:
            data['conversations'] = data['conversations'][:num_conversations]
        else:
            # Generate additional conversations
            needed = num_conversations - len(data['conversations'])
            for _ in range(needed):
                # Pick a random profile
                profile = random.choice(data['profiles'])
                is_scam = profile.get('risk_score', 0) > 0.7 and random.random() < 0.8
                conversation = generator.generate_conversation(profile, is_scam)
                data['conversations'].append(conversation)
    
    return data

if __name__ == "__main__":
    generator = TinderDataGenerator()
    data = generator.generate_dataset(100)
    generator.save_to_file(data)

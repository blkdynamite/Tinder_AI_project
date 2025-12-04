"""
Demo Scenarios Module

This module contains pre-built demo data and helper functions for interview scenarios.
Provides realistic test cases for demonstrating the AI detection capabilities.
"""

import random
import hashlib
from faker import Faker
from typing import Dict, List
import streamlit as st

# Demo constants
SNAPCHAT_DEMO_MESSAGES = [
    "Hey! Can we move to Snapchat? My main is broken",
    "Let's continue on Instagram, this app is glitchy",
    "Text me on WhatsApp instead",
    "Want to video call on Telegram?",
]

SNAPCHAT_DEMO_PROFILES = [
    {
        'name': 'Sarah M.',
        'photo_count': 1,
        'account_age_days': 2,
        'verified': False,
        'bio': 'Looking for someone special'
    },
    {
        'name': 'Emma L.',
        'photo_count': 2,
        'account_age_days': 3,
        'verified': False,
        'bio': 'New to this, be kind!'
    },
    {
        'name': 'Jessica T.',
        'photo_count': 5,
        'account_age_days': 60,
        'verified': True,
        'bio': 'Adventurous and fun'
    },
]

TINDER_SWINDLER_CASE = {
    'name': 'Shimon Hayut (Simon Leviev)',
    'estimated_loss': 10000000,
    'victims': 10,
    'pattern': [
        {
            'day': 1,
            'action': 'Create account with 1 luxury photo',
            'bio': 'Son of billionaire Lev Leviev',
            'detection': 'Funnel Detector: Single photo + luxury claims flag as suspicious'
        },
        {
            'day': 3,
            'action': 'Take victim on expensive first date (private jet, 5-star hotel)',
            'detection': 'Ring Detector: Would link to other victims on same timeline'
        },
        {
            'day': 14,
            'action': 'Love bombing + send victim $10K',
            'detection': 'Funnel Detector: Rapid escalation flag'
        },
        {
            'day': 21,
            'action': 'Introduce "enemies threat" with fake bodyguard photo',
            'detection': 'Ring Detector: Same bodyguard photo sent to 5+ victims'
        },
        {
            'day': 28,
            'action': 'Request victim open credit card in her name',
            'detection': 'Funnel Detector: Financial request detected'
        },
        {
            'day': 60,
            'action': 'Victim realizes scam, loses $130K+',
            'prevention': 'PREVENTED: If detected on day 1-3'
        }
    ]
}


@st.cache_data
def generate_coordinated_ring_data(num_accounts: int = 20, num_rings: int = 3) -> dict:
    """
    Generate synthetic profiles and conversations with deliberate ring structure.
    
    Creates realistic test data where some accounts are deliberately connected
    (same device, similar bios) to form detectable rings, while others are random.
    
    Args:
        num_accounts: Total number of profiles to generate
        num_rings: Number of coordinated rings to create
    
    Returns:
        Dictionary containing:
            - profiles: List of profile dictionaries
            - conversations: List of conversation dictionaries
            - ring_structure: Metadata about deliberately created rings
    """
    fake = Faker()
    
    profiles = []
    conversations = []
    ring_structure = {
        'rings': [],
        'total_accounts': num_accounts,
        'ring_accounts': 0,
        'random_accounts': 0
    }
    
    # Calculate accounts per ring
    accounts_per_ring = num_accounts // num_rings
    remaining_accounts = num_accounts - (accounts_per_ring * num_rings)
    
    # Generate ring profiles
    for ring_id in range(num_rings):
        ring_members = []
        ring_size = accounts_per_ring + (1 if ring_id < remaining_accounts else 0)
        
        # Shared infrastructure for this ring
        device_hash = f"device_ring_{ring_id}_{hashlib.md5(str(ring_id).encode()).hexdigest()[:8]}"
        ip_subnet = f"192.168.{ring_id + 1}"
        
        # Create similar bio template for this ring
        bio_templates = [
            "Love traveling and meeting new people! Looking for something real.",
            "Adventurous spirit seeking genuine connections. Let's explore together!",
            "Passionate about life, travel, and meaningful conversations.",
            "World traveler looking for my next adventure partner.",
            "Seeking authentic connections in this digital world."
        ]
        ring_bio_template = random.choice(bio_templates)
        
        # Generate profiles for this ring
        for i in range(ring_size):
            profile_id = f"ring_{ring_id}_user_{i}"
            
            # Vary the bio slightly but keep high similarity
            bio_variations = [
                ring_bio_template,
                ring_bio_template.replace("Love", "Enjoy").replace("!", "."),
                ring_bio_template.replace("traveling", "exploring"),
                ring_bio_template + " DM me!",
                ring_bio_template.replace("!", "! 😊")
            ]
            bio = random.choice(bio_variations)
            
            profile = {
                'id': profile_id,
                'profile_id': profile_id,
                'name': fake.first_name() + " " + fake.last_name()[0] + ".",
                'bio': bio,
                'device_hash': device_hash,  # Shared device
                'ip': f"{ip_subnet}.{random.randint(1, 254)}",  # Same subnet
                'photo_count': random.choice([1, 1, 2, 3]),  # Bias toward low photo count
                'account_age_days': random.randint(1, 10),  # New accounts
                'verified': False,
                'risk_score': random.uniform(0.6, 0.9),  # High risk
                'age': random.randint(25, 45),
                'location': fake.city(),
                'occupation': random.choice(['Business Owner', 'Investor', 'Entrepreneur', 'Consultant'])
            }
            
            profiles.append(profile)
            ring_members.append(profile_id)
        
        ring_structure['rings'].append({
            'ring_id': f'ring_{ring_id}',
            'members': ring_members,
            'size': len(ring_members),
            'device_hash': device_hash,
            'ip_subnet': ip_subnet,
            'bio_template': ring_bio_template
        })
        ring_structure['ring_accounts'] += len(ring_members)
    
    # Generate random profiles (not in rings)
    random_accounts = num_accounts - ring_structure['ring_accounts']
    for i in range(random_accounts):
        profile_id = f"random_user_{i}"
        
        profile = {
            'id': profile_id,
            'profile_id': profile_id,
            'name': fake.first_name() + " " + fake.last_name(),
            'bio': fake.text(max_nb_chars=150),
            'device_hash': f"device_{hashlib.md5(str(i).encode()).hexdigest()[:12]}",  # Unique devices
            'ip': f"{random.randint(10, 223)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 254)}",  # Random IPs
            'photo_count': random.randint(3, 8),  # More photos
            'account_age_days': random.randint(30, 365),  # Older accounts
            'verified': random.choice([True, False, False, False]),  # Some verified
            'risk_score': random.uniform(0.2, 0.6),  # Lower risk
            'age': random.randint(22, 50),
            'location': fake.city(),
            'occupation': fake.job()
        }
        
        profiles.append(profile)
    
    ring_structure['random_accounts'] = random_accounts
    
    # Generate some conversations
    for _ in range(min(30, num_accounts * 2)):
        sender = random.choice(profiles)
        conversation = {
            'id': f"conv_{fake.uuid4()}",
            'profile_id': sender['id'],
            'participants': [sender['id'], 'user_001'],
            'messages': [
                {
                    'sender': sender['id'],
                    'text': random.choice([
                        "Hey! How are you?",
                        "You seem interesting!",
                        "Want to chat?",
                        "Hi there! 😊"
                    ]),
                    'content': random.choice([
                        "Hey! How are you?",
                        "You seem interesting!",
                        "Want to chat?",
                        "Hi there! 😊"
                    ]),
                    'timestamp': fake.date_time_this_month().isoformat(),
                    '_internal_sender': 'them'
                }
            ],
            'duration_days': random.randint(1, 30),
            'is_scam': sender.get('risk_score', 0.5) > 0.7,
            'risk_score': sender.get('risk_score', 0.5)
        }
        conversations.append(conversation)
    
    return {
        'profiles': profiles,
        'conversations': conversations,
        'ring_structure': ring_structure
    }


def get_snapchat_demo_example() -> dict:
    """
    Get pre-built example for Snapchat pivot demo.
    
    Returns:
        Dictionary containing:
            - message: Example message requesting platform switch
            - sender_profile: Profile information for sender
            - explanation: Brief explanation of why this is suspicious
    """
    message = random.choice(SNAPCHAT_DEMO_MESSAGES)
    profile = random.choice(SNAPCHAT_DEMO_PROFILES)
    
    explanation = (
        f"This message requests moving to an external platform ({message.split()[-1] if message.split() else 'external app'}). "
        f"Combined with a profile that has {profile['photo_count']} photo(s), "
        f"account age of {profile['account_age_days']} days, and "
        f"{'unverified' if not profile['verified'] else 'verified'} status, "
        f"this represents a high-risk funnel attempt."
    )
    
    return {
        'message': message,
        'sender_profile': profile.copy(),
        'explanation': explanation
    }


def get_ring_demo_example() -> dict:
    """
    Get pre-built example for ring detection demo.
    
    Returns:
        Dictionary containing:
            - profiles: List of profiles that form a detectable ring
            - ring_structure: Metadata about the ring
            - explanation: Brief explanation of ring characteristics
    """
    # Generate a small coordinated ring
    ring_data = generate_coordinated_ring_data(num_accounts=6, num_rings=1)
    
    ring_info = ring_data['ring_structure']['rings'][0] if ring_data['ring_structure']['rings'] else {}
    
    explanation = (
        f"This ring contains {ring_info.get('size', 0)} accounts that share: "
        f"the same device hash ({ring_info.get('device_hash', 'N/A')[:20]}...), "
        f"IP addresses in the same subnet ({ring_info.get('ip_subnet', 'N/A')}), "
        f"and similar bio content. This pattern indicates coordinated activity."
    )
    
    return {
        'profiles': ring_data['profiles'],
        'ring_structure': ring_info,
        'explanation': explanation
    }


def get_swindler_timeline() -> list:
    """
    Get the Tinder Swindler case study timeline.
    
    Returns:
        List of pattern dictionaries from TINDER_SWINDLER_CASE showing
        the progression of a real-world romance scam case.
    """
    return TINDER_SWINDLER_CASE['pattern'].copy()


def get_swindler_case_summary() -> dict:
    """
    Get summary information about the Tinder Swindler case.
    
    Returns:
        Dictionary with case summary information
    """
    return {
        'name': TINDER_SWINDLER_CASE['name'],
        'estimated_loss': TINDER_SWINDLER_CASE['estimated_loss'],
        'victims': TINDER_SWINDLER_CASE['victims'],
        'timeline_days': max([p['day'] for p in TINDER_SWINDLER_CASE['pattern']]) if TINDER_SWINDLER_CASE['pattern'] else 0
    }


if __name__ == "__main__":
    # Test the demo scenarios
    print("Testing Demo Scenarios Module")
    print("=" * 50)
    
    # Test Snapchat demo
    snapchat_example = get_snapchat_demo_example()
    print("\n1. Snapchat Demo Example:")
    print(f"   Message: {snapchat_example['message']}")
    print(f"   Profile: {snapchat_example['sender_profile']}")
    print(f"   Explanation: {snapchat_example['explanation'][:100]}...")
    
    # Test ring demo
    ring_example = get_ring_demo_example()
    print(f"\n2. Ring Demo Example:")
    print(f"   Profiles: {len(ring_example['profiles'])}")
    print(f"   Ring Structure: {ring_example['ring_structure']}")
    print(f"   Explanation: {ring_example['explanation'][:100]}...")
    
    # Test coordinated ring data
    ring_data = generate_coordinated_ring_data(num_accounts=20, num_rings=3)
    print(f"\n3. Coordinated Ring Data:")
    print(f"   Total Profiles: {len(ring_data['profiles'])}")
    print(f"   Ring Structure: {ring_data['ring_structure']}")
    
    # Test swindler timeline
    timeline = get_swindler_timeline()
    print(f"\n4. Tinder Swindler Timeline:")
    print(f"   Timeline Events: {len(timeline)}")
    for event in timeline[:3]:
        print(f"   Day {event['day']}: {event['action']}")


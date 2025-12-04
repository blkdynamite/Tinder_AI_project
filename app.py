"""
Tinder AI Agent Demo
A scalable detection system for identifying risky Tinder profiles and conversations.

Features:
- Profile Scanner: CV + NLP analysis for photo and bio risk assessment
- Message Auditor: Sequence modeling for escalation and money cues detection
- Trend Monitor: KPI dashboard with risk heatmaps and mitigation recommendations

Author: AI Agent Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path

# Page configuration (must be first)
st.set_page_config(
    page_title="Tinder AI Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules with error handling for cloud deployment
ProfileScanner = None
MessageAuditor = None
TrendMonitor = None
CrossPlatformFunnelDetector = None
BadActorRingDetector = None
generate_synthetic_data = None
MODULES_LOADED = False

try:
    from src.profile_scanner import ProfileScanner
    from src.message_auditor import MessageAuditor
    from src.trend_monitor import TrendMonitor
    from src.funnel_detector import CrossPlatformFunnelDetector
    from src.ring_detector import BadActorRingDetector
    from src.data_generator import generate_synthetic_data
    try:
        from src.demo_scenarios import (
            get_snapchat_demo_example,
            get_ring_demo_example,
            generate_coordinated_ring_data,
            get_swindler_timeline,
            get_swindler_case_summary
        )
    except ImportError:
        # Demo scenarios optional
        get_snapchat_demo_example = None
        get_ring_demo_example = None
        generate_coordinated_ring_data = None
        get_swindler_timeline = None
        get_swindler_case_summary = None
    MODULES_LOADED = True
except ImportError as e:
    # Don't show error at import time - handle in main()
    MODULES_LOADED = False
except Exception as e:
    # Handle any other import errors
    MODULES_LOADED = False

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_profile_scanner():
    """Initialize ProfileScanner with caching - models loaded once"""
    if ProfileScanner is None:
        return None
    try:
        return ProfileScanner()
    except Exception as e:
        return None

@st.cache_resource
def initialize_message_auditor():
    """Initialize MessageAuditor with caching - models loaded once"""
    if MessageAuditor is None:
        return None
    try:
        return MessageAuditor()
    except Exception as e:
        return None

@st.cache_resource
def initialize_trend_monitor():
    """Initialize TrendMonitor with caching"""
    if TrendMonitor is None:
        return None
    try:
        return TrendMonitor()
    except Exception as e:
        return None

@st.cache_resource
def initialize_funnel_detector():
    """Initialize CrossPlatformFunnelDetector with caching"""
    if CrossPlatformFunnelDetector is None:
        return None
    try:
        return CrossPlatformFunnelDetector()
    except Exception as e:
        return None

@st.cache_resource
def initialize_ring_detector():
    """Initialize BadActorRingDetector with caching"""
    if BadActorRingDetector is None:
        return None
    try:
        return BadActorRingDetector()
    except Exception as e:
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_demo_data():
    """Load or generate demo data with caching for performance"""
    data_file = Path("data/demo_tinder_data.json")
    if data_file.exists():
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except:
            pass

    # Generate demo data if file doesn't exist
    try:
        # Use global generate_synthetic_data if available
        if generate_synthetic_data is not None:
            data = generate_synthetic_data(num_profiles=20, num_conversations=10, include_risky=True, include_scams=True)
        else:
            # Fallback: try to import directly
            from src.data_generator import generate_synthetic_data as gen_func
            data = gen_func(num_profiles=20, num_conversations=10, include_risky=True, include_scams=True)
        # Save for future use
        data_file.parent.mkdir(exist_ok=True)
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return data
    except:
        # Fallback demo data
        return {
            "profiles": [
                {
                    "profile_id": "demo_001",
                    "name": "John Doe",
                    "age": 35,
                    "bio": "Military officer overseas. Send money to help my family.",
                    "photos": ["https://picsum.photos/400/600?random=1"],
                    "risk_score": 0.8,
                    "risk_factors": ["Military claims", "Financial requests"]
                }
            ],
            "conversations": [
                {
                    "conversation_id": "conv_001",
                    "participants": ["demo_001", "user_001"],
                    "messages": [
                        {"sender": "demo_001", "content": "Hey! Can we move to Snapchat?", "timestamp": "2024-01-01T10:00:00"},
                        {"sender": "demo_001", "content": "I need $500 urgently", "timestamp": "2024-01-01T10:05:00"}
                    ],
                    "risk_score": 0.9
                }
            ]
        }

def main():
    """Main application function"""

    if not MODULES_LOADED:
        st.error("⚠️ Application modules failed to load.")
        st.info("💡 The app will run in basic mode with limited functionality.")
        # Continue anyway - app can still show basic data

    # Sidebar navigation
    st.sidebar.title("🔍 Tinder AI Agent")
    st.sidebar.markdown("---")

    # Mode selection
    st.sidebar.markdown("### 📊 MODE SELECT")
    
    mode = st.sidebar.radio(
        "Choose Experience Mode",
        ["🎬 DEMO MODE (Interview)", "🔧 Advanced Mode"],
        key="mode_select"
    )
    
    st.sidebar.markdown("---")
    
    if mode == "🎬 DEMO MODE (Interview)":
        demo_scenario = st.sidebar.selectbox(
            "📍 Select Scenario",
            [
                "🎯 Start Here - Your Approach",
                "📱 The Snapchat Pivot",
                "🕸️ The Scam Ring",
                "🎓 The Tinder Swindler Case Study"
            ],
            key="demo_select"
        )
        
        # Store selected page as demo_scenario instead of regular page
        page = demo_scenario
    
    else:  # Advanced Mode
        page = st.sidebar.radio(
            "Advanced Tools",
            ["Dashboard Overview", "Profile Scanner", "Message Auditor", "Funnel Detector", "Ring Detector", "Case Studies", "Trend Monitor", "Data Generator"]
        )

    # Load demo data
    with st.sidebar:
        if st.button("🔄 Refresh Demo Data", help="Regenerate demo data"):
            st.cache_data.clear()
            st.rerun()

    data = load_demo_data()

    # Initialize components with caching (models loaded once, reused across reruns)
    # Use cached initialization functions for better performance
    profile_scanner = initialize_profile_scanner()
    message_auditor = initialize_message_auditor()
    trend_monitor = initialize_trend_monitor()
    funnel_detector = initialize_funnel_detector()
    ring_detector = initialize_ring_detector()
    
    # Check if components loaded successfully
    COMPONENTS_LOADED = (profile_scanner is not None and message_auditor is not None)
    
    # Show warnings only once using session state
    if 'warnings_shown' not in st.session_state:
        if trend_monitor is None:
            st.sidebar.warning("⚠️ Trend monitor not available")
        if not COMPONENTS_LOADED:
            st.sidebar.warning("⚠️ AI components not fully loaded - running in basic mode")
        st.session_state.warnings_shown = True

    # Main content based on selected page
    if mode == "🎬 DEMO MODE (Interview)":
        if "Start Here" in page:
            show_demo_start()
        elif "Snapchat Pivot" in page:
            show_snapchat_pivot_demo(funnel_detector)
        elif "Scam Ring" in page:
            show_ring_detection_demo(ring_detector)
        elif "Tinder Swindler" in page:
            show_swindler_case_study()
    
    elif page == "Dashboard Overview":
        show_dashboard_overview(data, trend_monitor)

    elif page == "Profile Scanner":
        show_profile_scanner(data, profile_scanner, COMPONENTS_LOADED)

    elif page == "Message Auditor":
        show_message_auditor(data, message_auditor, COMPONENTS_LOADED)

    elif page == "Funnel Detector":
        show_funnel_detector(data, funnel_detector)

    elif page == "Ring Detector":
        show_ring_detector(data, ring_detector)

    elif page == "Case Studies":
        show_case_studies(funnel_detector, ring_detector)

    elif page == "Trend Monitor":
        show_trend_monitor(data, trend_monitor)

    elif page == "Data Generator":
        show_data_generator()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Demo Version 1.0**")
    st.sidebar.markdown("*Built with Streamlit & AI*")

def show_dashboard_overview(data, trend_monitor):
    """Display main dashboard with key metrics and risk overview"""
    
    if trend_monitor is None:
        st.warning("⚠️ Trend monitor not available. Showing basic dashboard.")
    
    st.markdown('<h1 class="main-header">🎯 Tinder AI Agent Dashboard</h1>', unsafe_allow_html=True)

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    total_profiles = len(data.get('profiles', []))
    total_conversations = len(data.get('conversations', []))
    high_risk_profiles = sum(1 for p in data.get('profiles', []) if p.get('risk_score', 0) > 0.7)
    flagged_conversations = sum(1 for c in data.get('conversations', []) if c.get('risk_flags', []))

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Profiles", total_profiles)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Conversations", total_conversations)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("High Risk Profiles", high_risk_profiles)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Flagged Conversations", flagged_conversations)
        st.markdown('</div>', unsafe_allow_html=True)

    # Risk Distribution
    st.markdown("### Risk Distribution Overview")

    if data.get('profiles'):
        risk_scores = [p.get('risk_score', 0) for p in data['profiles']]
        risk_df = pd.DataFrame({
            'Risk Score': risk_scores,
            'Category': pd.cut(risk_scores, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
        })

        fig = px.histogram(risk_df, x='Risk Score', color='Category',
                          title="Profile Risk Distribution",
                          color_discrete_map={'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'})
        st.plotly_chart(fig, use_container_width=True)

    # Recent High-Risk Alerts
    st.markdown("### 🚨 Recent High-Risk Alerts")

    high_risk_items = []
    for profile in data.get('profiles', []):
        if profile.get('risk_score', 0) > 0.7:
            high_risk_items.append({
                'type': 'Profile',
                'id': profile.get('profile_id', 'Unknown'),
                'risk_score': profile.get('risk_score', 0),
                'reason': profile.get('risk_reasons', ['Unknown'])[0] if profile.get('risk_reasons') else 'Unknown'
            })

    if high_risk_items:
        for item in high_risk_items[:5]:  # Show top 5
            risk_class = "risk-high" if item['risk_score'] > 0.8 else "risk-medium"
            st.markdown(f"""
            <div class="{risk_class}">
                <strong>{item['type']} ID: {item['id']}</strong><br>
                Risk Score: {item['risk_score']:.2f}<br>
                Reason: {item['reason']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No high-risk items detected in current dataset.")

@st.cache_data(ttl=300)  # Cache analysis results for 5 minutes
def analyze_profile_cached(profile_id, profile_data_json):
    """Cached profile analysis to avoid re-computation - uses hashable parameters"""
    # Get profile_scanner from cached resource
    profile_scanner = initialize_profile_scanner()
    if profile_scanner is None:
        return None
    try:
        # Reconstruct profile_data from JSON
        profile_data = json.loads(profile_data_json)
        return profile_scanner.analyze_profile(profile_data)
    except Exception as e:
        return None

def show_profile_scanner(data, profile_scanner, components_loaded):
    """Profile scanning interface"""

    st.markdown("## 👤 Profile Scanner")
    st.markdown("Analyze Tinder profiles for risk indicators using CV and NLP")
    
    if profile_scanner is None:
        components_loaded = False

    # Profile selection
    if data.get('profiles'):
        profile_options = [f"{p.get('name', 'Unknown')} (ID: {p.get('profile_id', 'Unknown')})"
                          for p in data['profiles']]
        selected_profile = st.selectbox("Select Profile to Analyze", profile_options)

        if selected_profile:
            profile_idx = profile_options.index(selected_profile)
            profile = data['profiles'][profile_idx]

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### Profile Details")
                st.json(profile)

            with col2:
                st.markdown("### Risk Analysis")

                if components_loaded:
                    if st.button("🔍 Analyze Profile", type="primary"):
                        with st.spinner("Analyzing profile..."):
                            # Use cached analysis function
                            profile_id = profile.get('profile_id', profile.get('id', 'unknown'))
                            profile_data_json = json.dumps(profile, sort_keys=True, default=str)
                            analysis_result = analyze_profile_cached(profile_id, profile_data_json)
                            
                            if analysis_result is None:
                                try:
                                    # Fallback to direct call if cache fails
                                    analysis_result = profile_scanner.analyze_profile(profile)
                                except Exception as e:
                                    st.error(f"Analysis failed: {e}")
                                    analysis_result = None
                            
                            if analysis_result:
                                # Display results
                                risk_score = analysis_result.get('risk_score', 0)
                                risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"

                                risk_class = f"risk-{risk_level.lower()}"
                                st.markdown(f"""
                                <div class="{risk_class}">
                                    <h3>Risk Level: {risk_level}</h3>
                                    <p><strong>Risk Score:</strong> {risk_score:.3f}</p>
                                    <p><strong>Analysis:</strong> {analysis_result.get('analysis', 'No analysis available')}</p>
                                </div>
                                """, unsafe_allow_html=True)

                                # Show risk factors
                                if analysis_result.get('risk_factors'):
                                    st.markdown("#### Risk Factors Detected:")
                                    for factor in analysis_result['risk_factors']:
                                        st.write(f"• {factor}")
                            else:
                                # Show pre-computed risk score if analysis failed
                                st.info("Showing pre-computed risk score instead.")
                                risk_score = profile.get('risk_score', 0)
                                risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"
                                st.metric("Risk Level", risk_level)
                                st.metric("Risk Score", f"{risk_score:.3f}")
                else:
                    st.warning("AI components not loaded. Showing basic profile information.")
                    risk_score = profile.get('risk_score', 0)
                    risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"
                    st.metric("Risk Level", risk_level)
                    st.metric("Risk Score", f"{risk_score:.3f}")
    else:
        st.warning("No profiles available. Please generate synthetic data first.")

@st.cache_data(ttl=300)  # Cache audit results for 5 minutes
def audit_conversation_cached(conversation_id, conversation_data_json):
    """Cached conversation audit to avoid re-computation - uses hashable parameters"""
    # Get message_auditor from cached resource
    message_auditor = initialize_message_auditor()
    if message_auditor is None:
        return None
    try:
        # Reconstruct conversation_data from JSON
        conversation_data = json.loads(conversation_data_json)
        return message_auditor.audit_conversation(conversation_data)
    except Exception as e:
        return None

# Note: Trend data generation is fast, so we don't cache it to avoid unhashable parameter issues

def show_message_auditor(data, message_auditor, components_loaded):
    """Message auditing interface"""

    st.markdown("## 💬 Message Auditor")
    st.markdown("Analyze conversation patterns for escalation and money cues")
    
    if message_auditor is None:
        components_loaded = False
        st.warning("⚠️ Message auditor not available. Cannot perform advanced analysis.")

    if data.get('conversations'):
        conv_options = [f"Conversation {i+1} - {len(c.get('messages', []))} messages"
                       for i, c in enumerate(data['conversations'])]
        selected_conv = st.selectbox("Select Conversation to Audit", conv_options)

        if selected_conv:
            conv_idx = conv_options.index(selected_conv)
            conversation = data['conversations'][conv_idx]

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### Conversation Details")
                st.write(f"**Participants:** {conversation.get('participants', [])}")
                st.write(f"**Duration:** {conversation.get('duration_days', 0)} days")

                # Display messages
                st.markdown("#### Messages:")
                for msg in conversation.get('messages', [])[:10]:  # Show first 10
                    st.write(f"**{msg.get('sender', 'Unknown')}:** {msg.get('content', '')}")

            with col2:
                st.markdown("### Risk Audit")

                if message_auditor is None:
                    st.warning("Message auditor not available. Cannot perform analysis.")
                    # Show basic info
                    risk_score = conversation.get('risk_score', 0.5)
                    st.metric("Risk Score", f"{risk_score:.3f}")
                elif st.button("🔍 Audit Conversation", type="primary"):
                    with st.spinner("Auditing conversation..."):
                        # Use cached audit function
                        conversation_id = conversation.get('id', conversation.get('conversation_id', 'unknown'))
                        conversation_data_json = json.dumps(conversation, sort_keys=True, default=str)
                        audit_result = audit_conversation_cached(conversation_id, conversation_data_json)
                        
                        if audit_result is None:
                            try:
                                # Fallback to direct call if cache fails
                                audit_result = message_auditor.audit_conversation(conversation)
                            except Exception as e:
                                st.error(f"Audit failed: {e}")
                                audit_result = {
                                    'risk_score': conversation.get('risk_score', 0.5),
                                    'analysis': 'Analysis unavailable',
                                    'flagged_messages': []
                                }

                    # Display results
                    risk_score = audit_result.get('risk_score', 0)
                    risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"

                    risk_class = f"risk-{risk_level.lower()}"
                    st.markdown(f"""
                    <div class="{risk_class}">
                        <h3>Risk Level: {risk_level}</h3>
                        <p><strong>Risk Score:</strong> {risk_score:.3f}</p>
                        <p><strong>Analysis:</strong> {audit_result.get('analysis', 'No analysis available')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show flagged messages
                    if audit_result.get('flagged_messages'):
                        st.markdown("#### Flagged Messages:")
                        for flag in audit_result['flagged_messages'][:5]:
                            st.write(f"• **{flag.get('reason', 'Unknown')}:** {flag.get('message', '')[:100]}...")
    else:
        st.warning("No conversations available. Please generate synthetic data first.")

def show_trend_monitor(data, trend_monitor):
    """Trend monitoring dashboard"""

    st.markdown("## 📊 Trend Monitor")
    st.markdown("Monitor KPIs and risk trends over time")

    if trend_monitor is None:
        st.error("⚠️ Trend monitor is not available. Please check that all dependencies are installed.")
        st.info("💡 The trend monitor requires all AI components to be initialized successfully.")
        return

    # Generate trend data (not cached - fast operation, avoids unhashable parameter issues)
    try:
        trend_data = trend_monitor.generate_trend_data(data)
    except Exception as e:
        st.error(f"Failed to generate trend data: {e}")
        trend_data = None

    # Time series plot
    if trend_data is not None and not trend_data.empty:
        st.markdown("### Risk Trends Over Time")
        fig = px.line(trend_data, x='date', y='risk_score',
                     title="Average Risk Score Trend",
                     labels={'risk_score': 'Risk Score', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
    elif trend_data is None:
        st.info("Trend data is not available. Please check that data has been loaded.")

    # KPI Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Match-to-Scam Rate", "2.3%", "↓ 0.5%")

    with col2:
        st.metric("False Positive Rate", "1.2%", "↓ 0.3%")

    with col3:
        st.metric("Detection Accuracy", "94.7%", "↑ 1.2%")

    # Risk heatmap
    st.markdown("### Risk Heatmap")
    st.info("Heatmap visualization would show geographic risk patterns")

def show_funnel_detector(data, funnel_detector):
    """Funnel detection interface for cross-platform migration attempts"""
    
    st.markdown("## 🚨 Funnel Detector")
    st.markdown("Detect attempts to move conversations off-platform to external messaging apps")
    st.info("💡 Scammers often try to move victims to Snapchat, Instagram, WhatsApp, or Telegram to avoid platform monitoring.")
    
    if funnel_detector is None:
        st.error("⚠️ Funnel detector is not available. Please check that all dependencies are installed.")
        return
    
    # Create two columns for input and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input")
        
        # Demo scenario button
        if MODULES_LOADED and get_snapchat_demo_example is not None:
            if st.button("📋 Load Demo Scenario", help="Load a pre-built demo example", key="load_funnel_demo"):
                try:
                    demo = get_snapchat_demo_example()
                    st.session_state.funnel_demo_message = demo['message']
                    st.session_state.funnel_demo_profile = demo['sender_profile']
                    st.session_state.funnel_demo_explanation = demo['explanation']
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load demo: {e}")
        
        # Message input
        default_message = st.session_state.get('funnel_demo_message', '')
        message = st.text_area(
            "Message to Analyze",
            value=default_message,
            placeholder="Enter the message text here...\n\nExample: 'Hey! Can we talk on snapchat? It's easier there'",
            height=150,
            key="funnel_message_input"
        )
        
        st.markdown("### Sender Profile Information")
        
        # Quick profile presets
        st.markdown("#### Quick Presets")
        preset_col1, preset_col2 = st.columns(2)
        
        with preset_col1:
            if st.button("🔴 High Risk Profile", use_container_width=True, key="high_risk_preset"):
                st.session_state.funnel_photo_count = 1
                st.session_state.funnel_account_age = 3
                st.session_state.funnel_verified = False
                st.rerun()
        
        with preset_col2:
            if st.button("🟢 Low Risk Profile", use_container_width=True, key="low_risk_preset"):
                st.session_state.funnel_photo_count = 5
                st.session_state.funnel_account_age = 200
                st.session_state.funnel_verified = True
                st.rerun()
        
        # Profile information inputs - use session state if set (demo or preset)
        demo_profile = st.session_state.get('funnel_demo_profile', {})
        default_photo_count = demo_profile.get('photo_count', st.session_state.get('funnel_photo_count', 1))
        default_account_age = demo_profile.get('account_age_days', st.session_state.get('funnel_account_age', 3))
        default_verified = demo_profile.get('verified', st.session_state.get('funnel_verified', False))
        
        # Show demo explanation if loaded
        if st.session_state.get('funnel_demo_explanation'):
            st.info(f"💡 {st.session_state.get('funnel_demo_explanation')}")
        
        photo_count = st.number_input(
            "Number of Photos", 
            min_value=0, 
            max_value=20, 
            value=default_photo_count, 
            step=1,
            key="funnel_photo_input"
        )
        account_age_days = st.number_input(
            "Account Age (days)", 
            min_value=0, 
            max_value=3650, 
            value=default_account_age, 
            step=1,
            key="funnel_age_input"
        )
        verified = st.checkbox(
            "Account Verified", 
            value=default_verified,
            key="funnel_verified_input"
        )
        
        sender_profile = {
            'photo_count': photo_count,
            'account_age_days': account_age_days,
            'verified': verified
        }
        
        if st.button("🔍 Detect Funnel Attempt", type="primary", use_container_width=True):
            if not message.strip():
                st.warning("Please enter a message to analyze.")
            else:
                with st.spinner("Analyzing message..."):
                    detection_result = funnel_detector.detect_snapchat_request(message, sender_profile)
                    action_result = funnel_detector.get_action(detection_result['confidence_score'])
                    
                    st.session_state.detection_result = detection_result
                    st.session_state.action_result = action_result
    
    with col2:
        st.markdown("### Detection Results")
        
        if 'detection_result' in st.session_state and 'action_result' in st.session_state:
            detection = st.session_state.detection_result
            action = st.session_state.action_result
            
            # Action badge with color coding
            action_color = {
                'WARN': '#f44336',  # Red
                'FLAG': '#ff9800',  # Orange
                'ALLOW': '#4caf50'   # Green
            }
            
            action_bg = {
                'WARN': '#ffebee',
                'FLAG': '#fff3e0',
                'ALLOW': '#e8f5e8'
            }
            
            st.markdown(f"""
            <div style="background-color: {action_bg.get(action['action'], '#f5f5f5')}; 
                        padding: 1rem; border-radius: 0.5rem; 
                        border-left: 4px solid {action_color.get(action['action'], '#gray')}; 
                        margin-bottom: 1rem;">
                <h3 style="color: {action_color.get(action['action'], '#333')}; margin-top: 0;">
                    Action: {action['action']}
                </h3>
                <p><strong>Confidence:</strong> {action['confidence']:.1%}</p>
                <p><strong>Reasoning:</strong> {action['reasoning']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed scores
            st.markdown("#### Detailed Scores")
            
            # Platform Request Score
            st.metric(
                "Platform Request Score",
                f"{detection['platform_request_score']:.1%}",
                help="Likelihood that message requests moving to external platform"
            )
            if detection['platform_detected']:
                st.info(f"📱 Platform Detected: **{detection['platform_detected'].title()}**")
            
            # Account Risk Score
            st.metric(
                "Account Risk Score",
                f"{detection['account_risk_score']:.1%}",
                help="Risk level based on profile characteristics"
            )
            
            # Message Velocity Score
            st.metric(
                "Message Velocity Score",
                f"{detection['message_velocity_score']:.1%}",
                help="Urgency/velocity indicators in message"
            )
            
            # Confidence Score (main)
            st.markdown("---")
            st.metric(
                "Overall Confidence Score",
                f"{detection['confidence_score']:.1%}",
                delta=f"{detection['confidence_score']*100:.1f}% confidence"
            )
            
            # Score breakdown visualization
            st.markdown("#### Score Breakdown")
            score_data = pd.DataFrame({
                'Component': ['Platform Request', 'Account Risk', 'Message Velocity'],
                'Score': [
                    detection['platform_request_score'],
                    detection['account_risk_score'],
                    detection['message_velocity_score']
                ],
                'Weight': [0.5, 0.25, 0.25]
            })
            
            fig = px.bar(
                score_data,
                x='Component',
                y='Score',
                title='Detection Score Components',
                color='Score',
                color_continuous_scale='RdYlGn_r',
                range_y=[0, 1]
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👈 Enter a message and click 'Detect Funnel Attempt' to see results")
    
    # Example messages section
    with st.expander("📋 Example Messages to Test"):
        st.markdown("""
        **High Risk Examples:**
        - "Hey! Can we talk on snapchat? It's easier there"
        - "Let's move to Instagram, I'm more active there"
        - "Text me on WhatsApp, it's better for photos"
        
        **Medium Risk Examples:**
        - "Do you have Instagram?"
        - "We should switch to Telegram"
        
        **Low Risk Examples:**
        - "Hi, how are you?"
        - "What do you like to do for fun?"
        """)

def show_ring_detector(data, ring_detector):
    """Ring detection interface for coordinated scam networks"""
    
    st.markdown("## 🔗 Ring Detector")
    st.markdown("Detect coordinated scam networks using graph clustering")
    st.info("💡 Identifies when multiple accounts operate together through shared devices, IPs, or similar content.")
    
    if ring_detector is None:
        st.error("⚠️ Ring detector is not available. Please check that all dependencies are installed.")
        return
    
    # Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Analysis Configuration")
        
        # DBSCAN parameters
        eps = st.slider(
            "Epsilon (Connection Threshold)",
            min_value=0.1,
            max_value=1.0,
            value=0.4,
            step=0.1,
            help="Maximum distance between samples in a cluster"
        )
        
        min_samples = st.slider(
            "Minimum Samples per Ring",
            min_value=2,
            max_value=10,
            value=2,
            step=1,
            help="Minimum number of accounts required to form a ring"
        )
        
        # Option to add synthetic infrastructure data
        add_infrastructure = st.checkbox(
            "Add Synthetic Infrastructure Data",
            value=True,
            help="Adds device hashes and IPs to profiles for demonstration"
        )
    
    with col2:
        st.markdown("### Quick Actions")
        if st.button("🔍 Analyze Current Data", type="primary", use_container_width=True):
            st.session_state.run_ring_analysis = True
            st.session_state.ring_eps = eps
            st.session_state.ring_min_samples = min_samples
            st.session_state.ring_add_infra = add_infrastructure
    
    # Prepare profiles - use demo data if loaded, otherwise use current data
    if st.session_state.get('ring_use_demo') and st.session_state.get('ring_demo_data'):
        profiles = st.session_state['ring_demo_data']['profiles']
        ring_structure_info = st.session_state['ring_demo_data']['ring_structure']
        st.info(f"📋 Using demo data: {ring_structure_info['ring_accounts']} ring accounts, {ring_structure_info['random_accounts']} random accounts")
    else:
        profiles = data.get('profiles', [])
        ring_structure_info = None
    
    if add_infrastructure and profiles and not st.session_state.get('ring_use_demo'):
        # Add synthetic infrastructure data for demo
        import random
        import hashlib
        
        # Create some shared infrastructure patterns
        device_hashes = ['device_abc', 'device_xyz', 'device_123'] * (len(profiles) // 3 + 1)
        ip_subnets = ['192.168.1', '192.168.2', '10.0.1'] * (len(profiles) // 3 + 1)
        
        for i, profile in enumerate(profiles):
            if 'device_hash' not in profile:
                # Assign some profiles to share devices
                profile['device_hash'] = device_hashes[i % len(device_hashes)]
            
            if 'ip' not in profile:
                # Assign some profiles to share IP subnets
                subnet = ip_subnets[i % len(ip_subnets)]
                profile['ip'] = f"{subnet}.{random.randint(1, 254)}"
            
            # Ensure required fields exist
            if 'risk_score' not in profile:
                profile['risk_score'] = random.uniform(0.3, 0.9)
            if 'account_age_days' not in profile:
                profile['account_age_days'] = random.randint(1, 365)
    
    # Run analysis if requested
    if st.session_state.get('run_ring_analysis', False):
        with st.spinner("Building network graph and detecting rings..."):
            try:
                # Use demo conversations if available
                conversations = st.session_state.get('ring_demo_data', {}).get('conversations', data.get('conversations', []))
                
                # Build graph
                G = ring_detector.build_user_graph(profiles, conversations)
                
                if G.number_of_nodes() == 0:
                    st.warning("No profiles available for analysis.")
                elif G.number_of_edges() == 0:
                    st.info("No connections detected between profiles. Try enabling 'Add Synthetic Infrastructure Data'.")
                else:
                    # Detect rings
                    rings = ring_detector.detect_rings(
                        G,
                        eps=st.session_state.get('ring_eps', eps),
                        min_samples=st.session_state.get('ring_min_samples', min_samples)
                    )
                    
                    st.session_state.ring_graph = G
                    st.session_state.detected_rings = rings
                    st.session_state.run_ring_analysis = False
                    
                    if rings:
                        st.success(f"✅ Detected {len(rings)} coordinated ring(s)!")
                    else:
                        st.info("No coordinated rings detected with current parameters.")
                        
            except Exception as e:
                st.error(f"Error during ring detection: {e}")
                logger.error(f"Ring detection error: {e}")
    
    # Display results
    if 'detected_rings' in st.session_state and st.session_state.detected_rings:
        rings = st.session_state.detected_rings
        G = st.session_state.get('ring_graph', None)
        
        st.markdown("---")
        st.markdown("### 🔍 Detected Rings")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rings", len(rings))
        with col2:
            total_members = sum(r['size'] for r in rings)
            st.metric("Total Members", total_members)
        with col3:
            high_severity = sum(1 for r in rings if r['severity_score'] > 0.85)
            st.metric("High Severity", high_severity)
        with col4:
            avg_size = np.mean([r['size'] for r in rings]) if rings else 0
            st.metric("Avg Ring Size", f"{avg_size:.1f}")
        
        # Display each ring
        for i, ring in enumerate(rings):
            with st.expander(f"🔴 {ring['ring_id']} - {ring['size']} members | Severity: {ring['severity_score']:.1%} | Action: {ring['action']}", expanded=(i == 0)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Members ({ring['size']}):**")
                    # Display member IDs in a compact format
                    members_display = ", ".join(ring['members'][:10])
                    if len(ring['members']) > 10:
                        members_display += f" ... and {len(ring['members']) - 10} more"
                    st.code(members_display, language=None)
                    
                    st.markdown(f"**Severity Score:** {ring['severity_score']:.3f}")
                    st.markdown(f"**Recommended Action:** {ring['action']}")
                    
                    # Edge details
                    edge_details = ring.get('edge_details', {})
                    st.markdown("**Connection Details:**")
                    st.write(f"- Shared Devices: {edge_details.get('shared_devices', 0)}")
                    st.write(f"- Shared IPs: {edge_details.get('shared_ips', 0)}")
                    st.write(f"- Similar Bios: {edge_details.get('similar_bios', 0)}")
                    st.write(f"- Total Connections: {edge_details.get('total_connections', 0)}")
                
                with col2:
                    # Action badge
                    action_color = {
                        'SUSPEND': '#f44336',
                        'FLAG': '#ff9800',
                        'MONITOR': '#4caf50'
                    }
                    action_bg = {
                        'SUSPEND': '#ffebee',
                        'FLAG': '#fff3e0',
                        'MONITOR': '#e8f5e8'
                    }
                    
                    st.markdown(f"""
                    <div style="background-color: {action_bg.get(ring['action'], '#f5f5f5')}; 
                                padding: 1rem; border-radius: 0.5rem; 
                                border-left: 4px solid {action_color.get(ring['action'], '#gray')};">
                        <h4 style="color: {action_color.get(ring['action'], '#333')}; margin-top: 0;">
                            {ring['action']}
                        </h4>
                        <p><strong>Severity:</strong> {ring['severity_score']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Network visualization
        if G and G.number_of_edges() > 0:
            st.markdown("---")
            st.markdown("### 📊 Network Visualization")
            
            try:
                # Create a simple network visualization
                # Get positions for nodes
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Create edge trace
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                # Create node trace
                node_x = []
                node_y = []
                node_text = []
                node_color = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(f"ID: {node}")
                    
                    # Color by risk score
                    risk = G.nodes[node].get('risk_score', 0.5)
                    node_color.append(risk)
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        showscale=True,
                        colorscale='RdYlGn_r',
                        reversescale=True,
                        color=node_color,
                        size=10,
                        colorbar=dict(
                            thickness=15,
                            title="Risk Score",
                            xanchor="left",
                            titleside="right"
                        )
                    )
                )
                
                fig = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Network Graph of Detected Rings',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not generate network visualization: {e}")
                st.info("Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    else:
        st.info("👈 Click 'Analyze Current Data' to detect coordinated rings")
        
        # Show data preview
        if profiles:
            st.markdown("### 📋 Available Profiles")
            st.write(f"Total profiles: {len(profiles)}")
            
            # Show sample profiles
            with st.expander("View Sample Profiles"):
                sample_profiles = profiles[:5]
                for profile in sample_profiles:
                    st.json({
                        'id': profile.get('id', profile.get('profile_id', 'Unknown')),
                        'risk_score': profile.get('risk_score', 'N/A'),
                        'account_age_days': profile.get('account_age_days', 'N/A'),
                        'has_device_hash': 'device_hash' in profile,
                        'has_ip': 'ip' in profile
                    })

def show_demo_start():
    """Landing page for demo mode"""
    st.markdown('''
    # 🎬 Tinder Anti-Abuse Demo
    
    **Your Challenge:** Detect romance scams before victims lose money
    
    **Your Approach:** 
    1. Detect individual scam funnels (Snapchat pivot)
    2. Detect coordinated scam networks (ring clustering)
    3. Stop them together = comprehensive defense
    
    **Select a scenario below to see how it works:**
    ''')
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''
        ### 📱 The Snapchat Pivot
        Real pattern: Scammers match on Tinder, immediately ask for Snapchat
        ''')
    
    with col2:
        st.markdown('''
        ### 🕸️ The Scam Ring
        Real problem: Organized criminals operate 20+ coordinated accounts
        ''')
    
    st.markdown("---")
    st.markdown('''
    ### 🎓 The Tinder Swindler
    Real case: Shimon Hayut stole $10M using these patterns
    How your system would have stopped him
    ''')

def show_snapchat_pivot_demo(funnel_detector):
    """Interactive Snapchat funnel detection demo"""
    st.markdown("## 📱 Cross-Platform Funnel Detection")
    
    st.markdown("""
    **The Problem:** Scammers match on Tinder, then immediately ask to move to Snapchat/Instagram.
    Once victims leave Tinder, they're outside your protective infrastructure.
    
    **Your Solution:** Real-time multi-signal detection of this funnel attempt.
    """)
    
    st.markdown("---")
    
    # Example messages
    example_messages = {
        "High Risk": "Hey! Can we move to Snapchat? My main is broken",
        "Medium Risk": "Let's continue on Instagram, this is easier",
        "Low Risk": "How was your day?",
    }
    
    col1, col2 = st.columns([3, 1])
    with col1:
        message = st.text_area(
            "Test Message:",
            value=example_messages["High Risk"],
            height=80,
            key="funnel_message"
        )
    with col2:
        st.markdown("**Quick Examples:**")
        for risk_level, example in example_messages.items():
            if st.button(risk_level, key=f"funnel_ex_{risk_level}"):
                st.session_state.funnel_message = example
                st.rerun()
    
    st.markdown("### Sender Profile")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        photos = st.slider("Photos", 1, 10, 1, key="funnel_photos")
    with col2:
        account_age = st.slider("Account Age (days)", 1, 365, 2, key="funnel_age")
    with col3:
        verified = st.checkbox("Verified?", False, key="funnel_verified")
    with col4:
        st.write("")  # Spacer
    
    if st.button("🔍 Analyze Message", type="primary", key="funnel_analyze"):
        if funnel_detector is None:
            st.error("Funnel detector not available")
        else:
            signals = funnel_detector.detect_snapchat_request(message, {
                'photo_count': photos,
                'account_age_days': account_age,
                'verified': verified
            })
            action = funnel_detector.get_action(signals['confidence_score'])
            
            st.markdown("---")
            
            # Risk box
            if action['action'] == 'WARN':
                risk_class = "risk-high"
                emoji = "⚠️"
            elif action['action'] == 'FLAG':
                risk_class = "risk-medium"
                emoji = "🚩"
            else:
                risk_class = "risk-low"
                emoji = "✅"
            
            st.markdown(f'''
            <div class="{risk_class}" style="background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #333;">
            <h2>{emoji} {action['action']}</h2>
            <p><strong>Confidence:</strong> {action['confidence']*100:.1f}%</p>
            <p><strong>Reasoning:</strong> {action['reasoning']}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Signal breakdown
            st.markdown("### 📊 Signal Breakdown")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Platform Request",
                    f"{signals['platform_request_score']*100:.0f}%",
                    f"({signals.get('platform_detected', 'None') or 'None'})"
                )
            with col2:
                st.metric("Account Risk", f"{signals['account_risk_score']*100:.0f}%")
            with col3:
                st.metric("Message Velocity", f"{signals['message_velocity_score']*100:.0f}%")
            
            st.markdown(f"""
            **What happens next:**
            
            - Recipient receives warning: "This message shows scam indicators"
            - Potential victim stays on Tinder where you can protect them
            - Account flagged for Trust & Safety review
            """)

def show_ring_detection_demo(ring_detector):
    """Interactive ring detection demo"""
    st.markdown("## 🕸️ Scam Ring Detection (Network Clustering)")
    
    st.markdown("""
    **The Problem:** Organized criminals operate 20-50 coordinated accounts.
    They share infrastructure: same device hash, same IP subnet, same message templates.
    
    **Your Solution:** Graph clustering finds entire rings at once.
    **The Win:** Block 1,000 accounts in ONE operation, not 1,000 separate actions.
    """)
    
    st.markdown("---")
    
    if st.button("🏗️ Generate 50 Profiles + Detect Rings", type="primary", key="ring_generate"):
        if ring_detector is None:
            st.error("Ring detector not available")
        elif generate_coordinated_ring_data is None:
            st.error("Demo scenarios not available")
        else:
            with st.spinner("Generating coordinated profiles..."):
                demo_data = generate_coordinated_ring_data(num_accounts=50, num_rings=3)
                
                G = ring_detector.build_user_graph(demo_data['profiles'], demo_data['conversations'])
                rings = ring_detector.detect_rings(G, eps=0.4, min_samples=2)
            
            st.markdown("---")
            st.markdown(f"### 🎯 Detected {len(rings)} Scam Rings")
            
            for ring in sorted(rings, key=lambda r: r['severity_score'], reverse=True):
                with st.expander(
                    f"🔴 {ring['ring_id'].upper()} - {ring['size']} accounts - Severity: {ring['severity_score']:.2f} [{ring['action']}]",
                    expanded=True
                ):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Members", ring['size'])
                    with col2:
                        st.metric("Severity", f"{ring['severity_score']:.2f}")
                    with col3:
                        st.metric("Action", ring['action'])
                    
                    st.markdown("**Member IDs:**")
                    st.write(ring['members'])
                    
                    st.markdown("**Coordination Signals:**")
                    edge_details = ring.get('edge_details', {})
                    for signal, count in edge_details.items():
                        st.write(f"• {signal}: {count}")
            
            # Metrics
            st.markdown("---")
            st.markdown("### 📊 Enforcement Impact")
            col1, col2, col3 = st.columns(3)
            with col1:
                total_accounts = sum(r['size'] for r in rings)
                st.metric("Accounts Flagged", total_accounts)
            with col2:
                st.metric("Operations Reduced", "98%")
            with col3:
                st.metric("Speed vs Manual", "Instant vs weeks")

def show_swindler_case_study():
    """The Tinder Swindler case study walkthrough"""
    st.markdown("## 🎓 The Tinder Swindler: How Your System Would Stop Him")
    
    st.markdown("""
    **Background:** Shimon Hayut stole **$10 million** from victims using dating apps.
    Netflix documentary "The Tinder Swindler" exposed his pattern.
    
    **Even after exposure, he scammed new victims.**
    
    **How your system would have prevented it:**
    """)
    
    if get_swindler_case_summary is None or get_swindler_timeline is None:
        st.error("Demo scenarios not available")
        return
    
    try:
        case_summary = get_swindler_case_summary()
        timeline = get_swindler_timeline()
        
        st.markdown("---")
        
        for i, stage in enumerate(timeline, 1):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### Day {stage['day']}")
            
            with col2:
                st.markdown(f"**Action:** {stage['action']}")
                if 'detection' in stage:
                    st.markdown(f"✅ **Detection:** {stage['detection']}")
                if 'prevention' in stage:
                    st.markdown(f"🎯 **Prevention:** {stage['prevention']}")
            
            if i < len(timeline):
                st.markdown("---")
        
        st.markdown("---")
        st.markdown(f"""
        ### 💰 Impact Summary
        
        **Total Loss:** ${case_summary['estimated_loss']:,}  
        **Victims:** {case_summary['victims']}  
        **With Your System:** Would have been detected and stopped on **Day 1-3**
        
        **Why This Matters for Tinder:**
        
        - Senators asked: "How do you protect users from algorithmic exploitation?"
        - Answer: "We use AI to proactively detect and intercept scams before victims are harmed"
        - Regulatory win: Demonstrates systematic defense, not reactive response
        """)
    except Exception as e:
        st.error(f"Error loading case study: {e}")

def show_case_studies(funnel_detector, ring_detector):
    """Case studies and demo scenarios interface"""
    
    st.markdown("## 📚 Case Studies & Demo Scenarios")
    st.markdown("Explore real-world scam patterns and test detection capabilities")
    
    # Tabs for different case studies
    tab1, tab2, tab3 = st.tabs(["🎯 Tinder Swindler Case", "📱 Funnel Detection Demo", "🔗 Ring Detection Demo"])
    
    with tab1:
        st.markdown("### The Tinder Swindler Case Study")
        
        if not MODULES_LOADED or get_swindler_case_summary is None:
            st.warning("Demo scenarios module not loaded. Some features may be unavailable.")
            return
        
        try:
            case_summary = get_swindler_case_summary()
            timeline = get_swindler_timeline()
            
            # Case summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Perpetrator", case_summary['name'])
            with col2:
                st.metric("Estimated Loss", f"${case_summary['estimated_loss']:,}")
            with col3:
                st.metric("Victims", case_summary['victims'])
            with col4:
                st.metric("Timeline", f"{case_summary['timeline_days']} days")
            
            st.markdown("---")
            st.markdown("### Timeline of Events")
            
            # Display timeline
            for event in timeline:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown(f"### Day {event['day']}")
                
                with col2:
                    st.markdown(f"**Action:** {event['action']}")
                    if 'bio' in event:
                        st.markdown(f"*Bio:* {event['bio']}")
                    if 'detection' in event:
                        st.success(f"✅ {event['detection']}")
                    if 'prevention' in event:
                        st.error(f"❌ {event['prevention']}")
                
                st.markdown("---")
            
            # Key learnings
            st.markdown("### 🔍 Key Detection Points")
            st.markdown("""
            **Early Detection Opportunities:**
            1. **Day 1**: Single photo + luxury claims → Funnel Detector flags
            2. **Day 3**: Multiple victims on same timeline → Ring Detector links accounts
            3. **Day 14**: Rapid escalation → Funnel Detector velocity score
            4. **Day 21**: Shared bodyguard photo → Ring Detector detects coordination
            
            **Prevention Impact:**
            - Early detection (Day 1-3) could have prevented $130K+ losses per victim
            - Ring detection would have linked all 10 victims immediately
            - Total prevention potential: $1.3M+ across all victims
            """)
            
        except Exception as e:
            st.error(f"Error loading case study: {e}")
    
    with tab2:
        st.markdown("### 📱 Funnel Detection Demo")
        st.info("Test the Funnel Detector with pre-built scenarios")
        
        if funnel_detector is None:
            st.warning("Funnel detector not available")
        elif get_snapchat_demo_example is None:
            st.warning("Demo scenarios not available")
        else:
            try:
                demo = get_snapchat_demo_example()
                
                st.markdown("#### Demo Example")
                st.markdown(f"**Message:** {demo['message']}")
                st.markdown(f"**Profile:** {demo['sender_profile']}")
                st.markdown(f"**Explanation:** {demo['explanation']}")
                
                if st.button("🔍 Run Detection on This Example", type="primary"):
                    result = funnel_detector.detect_snapchat_request(
                        demo['message'],
                        demo['sender_profile']
                    )
                    action = funnel_detector.get_action(result['confidence_score'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Detection Results")
                        st.metric("Platform Request Score", f"{result['platform_request_score']:.1%}")
                        st.metric("Account Risk Score", f"{result['account_risk_score']:.1%}")
                        st.metric("Message Velocity Score", f"{result['message_velocity_score']:.1%}")
                        st.metric("Confidence Score", f"{result['confidence_score']:.1%}")
                        if result['platform_detected']:
                            st.info(f"Platform Detected: {result['platform_detected'].title()}")
                    
                    with col2:
                        st.markdown("#### Action Recommendation")
                        action_color = {
                            'WARN': '#f44336',
                            'FLAG': '#ff9800',
                            'ALLOW': '#4caf50'
                        }
                        st.markdown(f"""
                        <div style="background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; 
                                    border-left: 4px solid {action_color.get(action['action'], '#gray')};">
                            <h3 style="color: {action_color.get(action['action'], '#333')};">
                                {action['action']}
                            </h3>
                            <p><strong>Confidence:</strong> {action['confidence']:.1%}</p>
                            <p><strong>Reasoning:</strong> {action['reasoning']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error loading funnel demo: {e}")
    
    with tab3:
        st.markdown("### 🔗 Ring Detection Demo")
        st.info("Test the Ring Detector with pre-built coordinated ring data")
        
        if ring_detector is None:
            st.warning("Ring detector not available")
        elif get_ring_demo_example is None:
            st.warning("Demo scenarios not available")
        else:
            try:
                demo = get_ring_demo_example()
                
                st.markdown("#### Demo Ring Structure")
                st.json(demo['ring_structure'])
                st.markdown(f"**Explanation:** {demo['explanation']}")
                
                if st.button("🔍 Detect Rings in This Example", type="primary"):
                    with st.spinner("Building graph and detecting rings..."):
                        G = ring_detector.build_user_graph(demo['profiles'], [])
                        rings = ring_detector.detect_rings(G, eps=0.4, min_samples=2)
                        
                        if rings:
                            st.success(f"✅ Detected {len(rings)} ring(s)!")
                            
                            for ring in rings:
                                with st.expander(f"{ring['ring_id']} - {ring['size']} members | Severity: {ring['severity_score']:.1%}"):
                                    st.write(f"**Members:** {', '.join(ring['members'])}")
                                    st.write(f"**Action:** {ring['action']}")
                                    st.write(f"**Edge Details:** {ring['edge_details']}")
                        else:
                            st.info("No rings detected. Try adjusting parameters.")
                
            except Exception as e:
                st.error(f"Error loading ring demo: {e}")

def show_data_generator():
    """Data generation interface"""

    st.markdown("## 🏭 Data Generator")
    st.markdown("Generate synthetic Tinder data for testing")

    col1, col2 = st.columns(2)

    with col1:
        num_profiles = st.slider("Number of Profiles", 10, 1000, 100)
        num_conversations = st.slider("Number of Conversations", 5, 500, 50)

    with col2:
        include_risky = st.checkbox("Include Risky Profiles", value=True)
        include_scams = st.checkbox("Include Scam Conversations", value=True)

    if st.button("🚀 Generate Synthetic Data", type="primary"):
        with st.spinner("Generating synthetic data..."):
            if generate_synthetic_data is None:
                st.error("Data generator not available. Please check that modules are loaded.")
            else:
                data = generate_synthetic_data(
                    num_profiles=num_profiles,
                    num_conversations=num_conversations,
                    include_risky=include_risky,
                    include_scams=include_scams
                )

                # Save to file
                data_path = Path("data/demo_tinder_data.json")
                data_path.parent.mkdir(exist_ok=True)
                with open(data_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)

                # Clear cache to force reload of new data
                st.cache_data.clear()
                
                st.success(f"Generated {len(data['profiles'])} profiles and {len(data['conversations'])} conversations!")
                st.balloons()
                
                # Trigger rerun to update dashboard with new data
                st.rerun()

# Streamlit runs the script directly, so call main() at module level
main()

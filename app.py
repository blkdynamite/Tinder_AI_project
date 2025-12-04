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

# Import custom modules with error handling for cloud deployment
try:
    from src.profile_scanner import ProfileScanner
    from src.message_auditor import MessageAuditor
    from src.trend_monitor import TrendMonitor
    from src.data_generator import generate_synthetic_data
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"Failed to load custom modules: {e}")
    st.error("Please ensure all dependencies are installed correctly.")
    MODULES_LOADED = False

# Page configuration
st.set_page_config(
    page_title="Tinder AI Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_data
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
        from src.data_generator import generate_synthetic_data
        data = generate_synthetic_data(num_profiles=20, num_conversations=10, include_risky=True, include_scams=True)
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
        st.error("Application modules failed to load. Please check the sidebar for setup instructions.")
        return

    # Sidebar navigation
    st.sidebar.title("🔍 Tinder AI Agent")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard Overview", "Profile Scanner", "Message Auditor", "Trend Monitor", "Data Generator"]
    )

    # Load demo data
    with st.sidebar:
        if st.button("🔄 Refresh Demo Data", help="Regenerate demo data"):
            st.cache_data.clear()
            st.rerun()

    data = load_demo_data()

    # Initialize components with error handling
    profile_scanner = None
    message_auditor = None
    trend_monitor = None
    COMPONENTS_LOADED = False
    
    # TrendMonitor doesn't require heavy dependencies, so create it first
    try:
        trend_monitor = TrendMonitor()
    except Exception as e:
        st.warning(f"⚠️ Trend monitor failed to initialize: {e}")
    
    # Try to initialize AI components (may fail if spaCy model missing)
    try:
        profile_scanner = ProfileScanner()
        message_auditor = MessageAuditor()
        COMPONENTS_LOADED = True
    except Exception as e:
        st.warning(f"⚠️ Some AI components failed to initialize: {e}")
        st.info("💡 The app will work in simplified mode. Some advanced features may be limited.")
        # Components will remain None, but app will still work

    # Main content based on selected page
    if page == "Dashboard Overview":
        show_dashboard_overview(data, trend_monitor)

    elif page == "Profile Scanner":
        show_profile_scanner(data, profile_scanner, COMPONENTS_LOADED)

    elif page == "Message Auditor":
        show_message_auditor(data, message_auditor, COMPONENTS_LOADED)

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
                            try:
                                analysis_result = profile_scanner.analyze_profile(profile)

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
                            except Exception as e:
                                st.error(f"Analysis failed: {e}")
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
                        try:
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

    # Generate trend data
    try:
        trend_data = trend_monitor.generate_trend_data(data)
    except Exception as e:
        st.error(f"Failed to generate trend data: {e}")
        trend_data = None

    # Time series plot
    if trend_data:
        st.markdown("### Risk Trends Over Time")
        fig = px.line(trend_data, x='date', y='risk_score',
                     title="Average Risk Score Trend",
                     labels={'risk_score': 'Risk Score', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)

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
            data = generate_synthetic_data(
                num_profiles=num_profiles,
                num_conversations=num_conversations,
                include_risky=include_risky,
                include_scams=include_scams
            )

            # Save to file
            with open("data/synthetic_tinder_data.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)

            st.success(f"Generated {len(data['profiles'])} profiles and {len(data['conversations'])} conversations!")
            st.balloons()

if __name__ == "__main__":
    main()

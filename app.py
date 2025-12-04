import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title='Tinder AI Agent', page_icon='', layout='wide')

st.title(' Tinder AI Agent Dashboard')
st.markdown('AI-powered detection system for Tinder scams and risky behavior')

# Load demo data
def load_demo_data():
    data_file = Path('data/demo_tinder_data.json')
    if data_file.exists():
        with open(data_file, 'r') as f:
            return json.load(f)
    return {'profiles': [], 'conversations': []}

data = load_demo_data()

# Dashboard metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric('Total Profiles', len(data.get('profiles', [])))

with col2:
    st.metric('Total Conversations', len(data.get('conversations', [])))

with col3:
    high_risk = sum(1 for p in data.get('profiles', []) if p.get('risk_score', 0) > 0.7)
    st.metric('High Risk Profiles', high_risk)

with col4:
    scams = sum(1 for c in data.get('conversations', []) if c.get('is_scam', False))
    st.metric('Detected Scams', scams)

st.success(' Dashboard ready! Deploy this to Streamlit Cloud for live demo.')
st.info(' To deploy: Go to share.streamlit.io and connect this GitHub repo')

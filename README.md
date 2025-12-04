# Tinder AI Agent

Interactive Streamlit demo that showcases AI-powered detection of risky Tinder-style profiles and conversations. Generate synthetic data, run CV+NLP scanners, and view KPI dashboards from one place.

## Project Layout
```
Tinder_AI_Project/
├── app.py              # Streamlit entrypoint
├── requirements.txt    # Python dependencies
├── data/               # Demo JSON data & outputs
├── src/                # Core detection modules
├── tests/              # Basic unit tests
└── docs/               # Notes & artifacts
```

## Local Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this folder to a public GitHub repo
2. Visit https://share.streamlit.io and create an app pointing to `app.py`
3. Streamlit Cloud installs `requirements.txt` automatically

## Features
- Profile Scanner: OpenCV + spaCy + transformers heuristics
- Message Auditor: regex + sentiment + escalation scoring
- Trend Monitor: Plotly KPI dashboards & mock heatmaps
- Data Generator: Faker-powered realistic demo data

## Tests
```
pip install -r requirements.txt
python -m pytest tests

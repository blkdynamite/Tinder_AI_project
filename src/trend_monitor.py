import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendMonitor:
    def __init__(self):
        """Initialize the trend monitor for KPI tracking and visualization"""
        pass

    def generate_trend_data(self, data, days=30):
        """
        Generate time series trend data from profiles and conversations
        
        Args:
            data: Dictionary containing 'profiles' and 'conversations'
            days: Number of days to generate trend data for
        
        Returns:
            DataFrame with date and risk_score columns
        """
        if not data or not data.get('profiles'):
            # Generate mock trend data
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            risk_scores = np.random.normal(0.4, 0.15, days)
            risk_scores = np.clip(risk_scores, 0, 1)  # Clip to [0, 1]
            
            return pd.DataFrame({
                'date': dates,
                'risk_score': risk_scores
            })

        # Extract risk scores from profiles
        profiles = data.get('profiles', [])
        conversations = data.get('conversations', [])

        # Generate daily aggregates
        trend_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Group profiles by date (simulate based on risk scores)
        for i in range(days):
            date = start_date + timedelta(days=i)
            
            # Calculate average risk for this day (simulated)
            # In real implementation, would use actual timestamps
            daily_profiles = profiles[i % len(profiles):(i % len(profiles)) + 5] if profiles else []
            daily_risk = np.mean([p.get('risk_score', 0.5) for p in daily_profiles]) if daily_profiles else 0.4
            
            trend_data.append({
                'date': date,
                'risk_score': daily_risk,
                'profile_count': len(daily_profiles)
            })

        return pd.DataFrame(trend_data)

    def generate_risk_heatmap(self, data):
        """
        Generate a risk heatmap showing geographic or temporal risk patterns
        
        Args:
            data: Dictionary containing profiles and conversations
        
        Returns:
            Plotly figure object
        """
        profiles = data.get('profiles', [])
        
        if not profiles:
            # Generate mock heatmap data
            locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
            risk_levels = ['Low', 'Medium', 'High']
            
            heatmap_data = []
            for loc in locations:
                for level in risk_levels:
                    heatmap_data.append({
                        'location': loc,
                        'risk_level': level,
                        'count': np.random.randint(0, 20)
                    })
            
            df = pd.DataFrame(heatmap_data)
            fig = px.density_heatmap(
                df, 
                x='location', 
                y='risk_level', 
                z='count',
                title='Risk Distribution by Location',
                color_continuous_scale='RdYlGn_r'
            )
            return fig

        # Extract location and risk data from profiles
        location_risk = defaultdict(list)
        
        for profile in profiles:
            location = profile.get('location', 'Unknown')
            risk_score = profile.get('risk_score', 0.5)
            location_risk[location].append(risk_score)

        # Aggregate by location
        heatmap_data = []
        for location, risks in location_risk.items():
            avg_risk = np.mean(risks)
            count = len(risks)
            
            # Categorize risk level
            if avg_risk < 0.3:
                risk_level = 'Low'
            elif avg_risk < 0.7:
                risk_level = 'Medium'
            else:
                risk_level = 'High'
            
            heatmap_data.append({
                'location': location,
                'risk_level': risk_level,
                'count': count,
                'avg_risk': avg_risk
            })

        df = pd.DataFrame(heatmap_data)
        
        if df.empty:
            return None

        fig = px.density_heatmap(
            df,
            x='location',
            y='risk_level',
            z='count',
            title='Risk Distribution by Location',
            color_continuous_scale='RdYlGn_r',
            labels={'count': 'Number of Profiles'}
        )
        
        return fig

    def calculate_kpis(self, data):
        """
        Calculate key performance indicators from the data
        
        Args:
            data: Dictionary containing profiles and conversations
        
        Returns:
            Dictionary of KPI metrics
        """
        profiles = data.get('profiles', [])
        conversations = data.get('conversations', [])

        kpis = {
            'total_profiles': len(profiles),
            'total_conversations': len(conversations),
            'high_risk_profiles': 0,
            'medium_risk_profiles': 0,
            'low_risk_profiles': 0,
            'scam_conversations': 0,
            'avg_risk_score': 0.0,
            'detection_rate': 0.0,
            'false_positive_rate': 0.0
        }

        if profiles:
            risk_scores = [p.get('risk_score', 0) for p in profiles]
            kpis['avg_risk_score'] = np.mean(risk_scores)
            
            kpis['high_risk_profiles'] = sum(1 for score in risk_scores if score > 0.7)
            kpis['medium_risk_profiles'] = sum(1 for score in risk_scores if 0.3 < score <= 0.7)
            kpis['low_risk_profiles'] = sum(1 for score in risk_scores if score <= 0.3)

        if conversations:
            kpis['scam_conversations'] = sum(1 for c in conversations if c.get('is_scam', False))
            
            # Calculate detection rate (scams detected / total conversations)
            if len(conversations) > 0:
                kpis['detection_rate'] = kpis['scam_conversations'] / len(conversations)

        # Mock false positive rate (would need ground truth data in real implementation)
        kpis['false_positive_rate'] = 0.012  # 1.2%

        return kpis

    def generate_risk_distribution_chart(self, data):
        """
        Generate a risk distribution chart
        
        Args:
            data: Dictionary containing profiles
        
        Returns:
            Plotly figure object
        """
        profiles = data.get('profiles', [])
        
        if not profiles:
            return None

        risk_scores = [p.get('risk_score', 0) for p in profiles]
        
        # Categorize risks
        risk_categories = []
        for score in risk_scores:
            if score < 0.3:
                risk_categories.append('Low')
            elif score < 0.7:
                risk_categories.append('Medium')
            else:
                risk_categories.append('High')

        df = pd.DataFrame({
            'Risk Category': risk_categories,
            'Count': [1] * len(risk_categories)
        })

        category_counts = df.groupby('Risk Category').sum().reset_index()

        fig = px.pie(
            category_counts,
            values='Count',
            names='Risk Category',
            title='Risk Distribution',
            color='Risk Category',
            color_discrete_map={
                'Low': '#4caf50',
                'Medium': '#ff9800',
                'High': '#f44336'
            }
        )

        return fig

    def generate_timeline_analysis(self, data, metric='risk_score'):
        """
        Generate timeline analysis for a specific metric
        
        Args:
            data: Dictionary containing profiles and conversations
            metric: Metric to analyze over time
        
        Returns:
            Plotly figure object
        """
        trend_df = self.generate_trend_data(data)
        
        if trend_df.empty:
            return None

        fig = px.line(
            trend_df,
            x='date',
            y='risk_score',
            title=f'{metric.replace("_", " ").title()} Over Time',
            labels={'risk_score': 'Risk Score', 'date': 'Date'},
            markers=True
        )

        # Add trend line
        z = np.polyfit(range(len(trend_df)), trend_df['risk_score'], 1)
        p = np.poly1d(z)
        trend_df['trend'] = p(range(len(trend_df)))

        fig.add_scatter(
            x=trend_df['date'],
            y=trend_df['trend'],
            mode='lines',
            name='Trend',
            line=dict(dash='dash', color='gray')
        )

        return fig

    def generate_comparison_chart(self, data, compare_by='location'):
        """
        Generate comparison chart grouped by a specific attribute
        
        Args:
            data: Dictionary containing profiles
            compare_by: Attribute to group by (e.g., 'location', 'age_group')
        
        Returns:
            Plotly figure object
        """
        profiles = data.get('profiles', [])
        
        if not profiles:
            return None

        comparison_data = []
        
        for profile in profiles:
            if compare_by == 'location':
                group = profile.get('location', 'Unknown')
            elif compare_by == 'age_group':
                age = profile.get('age', 25)
                if age < 25:
                    group = '18-24'
                elif age < 35:
                    group = '25-34'
                elif age < 45:
                    group = '35-44'
                else:
                    group = '45+'
            else:
                group = 'Unknown'
            
            comparison_data.append({
                'group': group,
                'risk_score': profile.get('risk_score', 0.5)
            })

        df = pd.DataFrame(comparison_data)
        grouped = df.groupby('group')['risk_score'].mean().reset_index()
        grouped = grouped.sort_values('risk_score', ascending=False)

        fig = px.bar(
            grouped,
            x='group',
            y='risk_score',
            title=f'Average Risk Score by {compare_by.replace("_", " ").title()}',
            labels={'risk_score': 'Average Risk Score', 'group': compare_by.replace('_', ' ').title()},
            color='risk_score',
            color_continuous_scale='RdYlGn_r'
        )

        return fig

if __name__ == "__main__":
    # Test the TrendMonitor
    monitor = TrendMonitor()
    
    # Test with empty data
    test_data = {'profiles': [], 'conversations': []}
    trend_df = monitor.generate_trend_data(test_data)
    print("Trend data shape:", trend_df.shape)
    print(trend_df.head())
    
    kpis = monitor.calculate_kpis(test_data)
    print("\nKPIs:", kpis)

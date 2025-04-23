import os
import sys
import subprocess
import json
import re
import time
from datetime import datetime

# Imports
import streamlit as st
import pandas as pd

# Configure page
st.set_page_config(
    page_title="QueryInsight Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styles
st.markdown("""
<style>
    .main { background-color: #f9fafb; }
    h1, h2, h3, h4 { color: #1e40af; font-weight: 600; }
    h1 { font-size: 1.8rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.8rem; margin-bottom: 1.5rem; }
    .stButton>button { background-color: #1e40af; color: white; border-radius: 4px; font-weight: 500; border: none; padding: 0.5rem 1rem; transition: all 0.2s; }
    .stButton>button:hover { background-color: #1e3a8a; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .metric-card { background: white; border-radius: 8px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05); border: 1px solid #f3f4f6; }
    .metric-value { font-size: 1.5rem; font-weight: 600; color: #1e40af; }
    .stTextInput>div>div>input, .stSelectbox, .stTextArea { border-radius: 4px; border: 1px solid #e5e7eb; }
    div.row-widget.stRadio > div { flex-direction: row; align-items: center; }
    .entity-display { background: white; border-radius: 8px; padding: 1rem; border: 1px solid #e5e7eb; overflow-x: auto; }
    .stProgress > div > div > div > div { background-color: #1e40af; }
    .sample-query { background: #f3f4f6; padding: 0.5rem 1rem; border-radius: 4px; margin: 0.25rem; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# Session state defaults
if 'custom_categories' not in st.session_state:
    default_categories = {
        "travel": ["flight", "book", "trip", "hotel", "vacation"],
        "finance": ["loan", "bank", "invest", "money", "stock"],
        "health": ["doctor", "symptom", "vaccine", "medical", "health"],
        "technology": ["computer", "software", "app", "tech", "digital"],
    }
    st.session_state['custom_categories'] = json.dumps(default_categories, indent=2)
if 'download_json' not in st.session_state:
    st.session_state['download_json'] = None
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None

# Using fallback models to avoid dependency issues
nlp = None
sentiment_model = None

st.warning("Running in fallback mode - using rule-based analysis without NLP models.")

# Analysis utilities
def normalize_query(query):
    """Normalize the query by converting to lowercase and removing extra spaces"""
    return " ".join(query.lower().split())

def detect_intent(query):
    """Detect the intent of the query"""
    query = query.lower()
    # Simple rule-based approach
    if any(word in query for word in ["how", "what", "why", "who", "when", "where", "explain"]):
        return "Informational"
    elif any(word in query for word in ["buy", "price", "cost", "purchase", "shop"]):
        return "Transactional"
    elif any(word in query for word in ["best", "top", "review", "compare", "vs", "versus"]):
        return "Commercial"
    elif any(word in query for word in ["find", "location", "address", "near", "nearby", "directions"]):
        return "Navigational"
    else:
        return "Informational"  # Default fallback

def extract_entities(query, nlp):
    """Extract named entities from the query"""
    # Fallback simple entity extraction using regex patterns
    entities = []
    
    # Simple patterns for common entities
    patterns = {
        "DATE": r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec),?\s+\d{4}\b|\b(?:next|last|this)\s+(?:weekend|week|month|year)\b|\b(?:yesterday|today|tomorrow)\b',
        "LOCATION": r'\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose|Austin|Jacksonville|San Francisco|Columbus|Indianapolis|Fort Worth|Charlotte|Seattle|Denver|Washington|Boston|El Paso|Nashville|Detroit|Oklahoma City|Portland|Las Vegas|Memphis|Louisville|Baltimore|Milwaukee|Albuquerque|Tucson|Fresno|Sacramento|Kansas City|Long Beach|Mesa|Atlanta|Colorado Springs|Miami|Raleigh|Omaha|Minneapolis|Tulsa|Cleveland|Wichita|Arlington|New Orleans|Bakersfield|Tampa|Aurora|Honolulu|Anaheim|Santa Ana|Corpus Christi|Riverside|St. Louis|Lexington|Pittsburgh|Anchorage|Stockton|Cincinnati|Saint Paul|Toledo|Newark|Greensboro|Plano|Henderson|Lincoln|Buffalo|Fort Wayne|Jersey City|Chula Vista|Orlando|St. Petersburg|Norfolk|Chandler|Laredo|Madison|Durham|Lubbock|Winston-Salem|Garland|Glendale|Hialeah|Reno|Baton Rouge|Irvine|Chesapeake|Irving|Scottsdale|North Las Vegas|Fremont|Gilbert|San Bernardino|Boise|Birmingham)\b',
        "PERSON": r'\b(?:[A-Z][a-z]+\s+[A-Z][a-z]+)\b',
        "MONEY": r'\$\d+(?:,\d+)*(?:\.\d+)?|\d+\s+dollars|\d+\s+USD',
        "ORGANIZATION": r'\b(?:Google|Microsoft|Apple|Amazon|Facebook|Tesla|Twitter|IBM|Intel|Nvidia|Netflix|Disney|Walmart|Target|Nike|Adidas|Samsung|Sony|Coca Cola|Pepsi|McDonald\'s|Burger King|Starbucks|Uber|Lyft|Airbnb|SpaceX|Boeing|Ford|Toyota|Honda|BMW|Mercedes|Ferrari|Porsche|Audi|Volkswagen|General Motors|Hyundai|Kia|Chrysler|Fiat|Chevrolet|Lexus|Bank of America|Wells Fargo|JPMorgan Chase|Citibank|Goldman Sachs|Morgan Stanley|Visa|Mastercard|American Express|PayPal|Venmo|Square|Stripe|Shopify|Adobe|Oracle|Salesforce|Slack|Zoom|Dropbox|Spotify|YouTube|Instagram|TikTok|Snapchat|LinkedIn|Reddit|Pinterest|WhatsApp|Telegram|Signal|WeChat|Line)\b',
    }
    
    for entity_type, pattern in patterns.items():
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            entities.append({
                "text": match.group(0),
                "type": entity_type,
                "start": match.start(),
                "end": match.end()
            })
    
    return entities

def detect_language(query):
    """Detect the language of the query"""
    # Fallback to assuming English if detection fails
    return "en"

def assign_category(query, custom_categories=None):
    """Assign a category to the query based on keywords"""
    query = query.lower()
    
    if not custom_categories:
        custom_categories = {
            "travel": ["flight", "book", "trip", "hotel", "vacation"],
            "finance": ["loan", "bank", "invest", "money", "stock"],
            "health": ["doctor", "symptom", "vaccine", "medical", "health"],
            "technology": ["computer", "software", "app", "tech", "digital"],
        }
    
    if isinstance(custom_categories, str):
        try:
            custom_categories = json.loads(custom_categories)
        except:
            # Fall back to default if JSON parsing fails
            custom_categories = {
                "travel": ["flight", "book", "trip", "hotel", "vacation"],
                "finance": ["loan", "bank", "invest", "money", "stock"],
                "health": ["doctor", "symptom", "vaccine", "medical", "health"],
                "technology": ["computer", "software", "app", "tech", "digital"],
            }
    
    # Find matching categories
    matches = {}
    for category, keywords in custom_categories.items():
        category_score = 0
        for keyword in keywords:
            if keyword.lower() in query:
                category_score += 1
        if category_score > 0:
            matches[category] = category_score
    
    if matches:
        # Return the category with the highest score
        return max(matches.items(), key=lambda x: x[1])[0]
    else:
        return "uncategorized"

def generate_relevance_score(query):
    """Generate a mock relevance score between 0-100"""
    # This is a simplified example - in reality would use more sophisticated methods
    score = 50  # Default score
    
    # Add points for query length (not too short, not too long)
    query_length = len(query.split())
    if 3 <= query_length <= 7:
        score += 20
    elif query_length > 7:
        score += 10
    
    # Add points for specificity indicators
    specificity_terms = ["specific", "exact", "particular", "precise", "detailed"]
    for term in specificity_terms:
        if term in query.lower():
            score += 5
    
    # Add some randomness (would be replaced by actual relevance metrics)
    import random
    score += random.randint(-10, 10)
    
    # Ensure score is between 0-100
    return max(0, min(100, score))

def detect_sentiment(query, model=None):
    """Detect sentiment of the query using rule-based approach"""
    # Rule-based sentiment analysis
    positive_words = ["good", "great", "excellent", "best", "like", "love", "awesome", "amazing"]
    negative_words = ["bad", "worst", "terrible", "hate", "dislike", "awful", "poor"]
    
    query_lower = query.lower()
    positive_count = sum(1 for word in positive_words if word in query_lower)
    negative_count = sum(1 for word in negative_words if word in query_lower)
    
    if positive_count > negative_count:
        return "POSITIVE", 0.7
    elif negative_count > positive_count:
        return "NEGATIVE", 0.7
    else:
        return "NEUTRAL", 0.9

def analyze_query_complexity(query):
    """Analyze the complexity of the query"""
    words = query.split()
    word_count = len(words)
    
    # Complexity factors
    complexity = {
        "word_count": word_count,
        "avg_word_length": sum(len(word) for word in words) / max(1, word_count),
        "has_conditionals": any(word in query.lower() for word in ["if", "when", "unless", "until"]),
        "has_comparisons": any(word in query.lower() for word in ["vs", "versus", "compared", "better", "than"]),
        "level": "Simple"  # default
    }
    
    # Determine complexity level
    if word_count > 10 or complexity["has_conditionals"] or complexity["has_comparisons"]:
        complexity["level"] = "Complex"
    elif word_count > 5:
        complexity["level"] = "Moderate"
    
    return complexity

def annotate_query(query, nlp, sentiment_model):
    """Perform complete analysis on a query"""
    normalized = normalize_query(query)
    
    # Get custom categories if available
    custom_categories = st.session_state.get('custom_categories', None)
    
    results = {
        "query": query,
        "normalized": normalized,
        "intent": detect_intent(query),
        "entities": extract_entities(query, nlp),
        "language": detect_language(query),
        "category": assign_category(query, custom_categories),
        "relevance": generate_relevance_score(query),
        "complexity": analyze_query_complexity(query),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Add sentiment analysis
    sentiment, confidence = detect_sentiment(query, sentiment_model)
    results["sentiment"] = {
        "label": sentiment,
        "confidence": confidence
    }
    
    return results

# Sidebar UI
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding-bottom:15px;">
            <h2 style="color:#1e40af; margin-bottom:0;">QueryInsight Pro</h2>
            <p style="color:#6b7280; font-size:0.9rem;">Query Analysis Tool</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<hr style='margin:0.5rem 0 1.5rem 0'>", unsafe_allow_html=True)
    with st.expander("✏️ Custom Categories", expanded=False):
        st.markdown("Define custom categories and keywords")
        custom_input = st.text_area(
            "Categories (JSON format)",
            value=st.session_state.get("custom_categories"),
            height=150,
        )
        if st.button("Save Categories", key="save_cats"):
            try:
                json.loads(custom_input)
                st.session_state["custom_categories"] = custom_input
                st.success("✅ Categories updated")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
    with st.expander("ℹ️ About", expanded=False):
        st.markdown(
            """
            **QueryInsight Pro** analyzes search queries using NLP techniques to extract:
            - Search intent
            - Named entities
            - Topic categories
            - Sentiment analysis
            - Query complexity
            """
        )
    with st.expander("🛠️ Installation", expanded=False):
        st.markdown(
            """
            If dependencies fail at runtime, ensure you have:
            ```
            pip install streamlit pandas
            ```
            For advanced NLP features (currently disabled):
            ```
            pip install spacy transformers langdetect
            python -m spacy download en_core_web_sm
            ```
            """
        )

# Main UI
st.markdown("# 🔍 QueryInsight Pro")
st.markdown("Analyze search queries with AI-powered insights")

query = st.text_input(
    "",
    placeholder="Type a search query (e.g., 'best flights to New York next weekend')",
    key="query_input",
)

# Example queries as sample labels (non-clickable)
st.markdown("### Sample Queries:")
example_queries = [
    "best flights to New York next weekend",
    "how to invest in cryptocurrency",
    "symptoms of flu vs covid vaccine"
]
example_html = ""
for example in example_queries:
    example_html += f'<span class="sample-query">{example}</span>'
st.markdown(f'<div>{example_html}</div>', unsafe_allow_html=True)

if query:
    with st.spinner("Analyzing query..."):
        results = annotate_query(query, nlp, sentiment_model)
        st.session_state['analysis_results'] = results
        
    # Display results
    st.markdown("## Analysis Results")
    
    # Top level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <p>Intent</p>
                <div class="metric-value">{results["intent"]}</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <p>Category</p>
                <div class="metric-value">{results["category"].title()}</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col3:
        sentiment_color = {
            "POSITIVE": "#10B981", 
            "NEUTRAL": "#6B7280", 
            "NEGATIVE": "#EF4444"
        }.get(results["sentiment"]["label"], "#6B7280")
        
        st.markdown(
            f"""
            <div class="metric-card">
                <p>Sentiment</p>
                <div class="metric-value" style="color:{sentiment_color}">
                    {results["sentiment"]["label"].title()}
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <p>Relevance Score</p>
                <div class="metric-value">{results["relevance"]}%</div>
            </div>
            """, unsafe_allow_html=True
        )
    
    # Entities section
    st.markdown("### Named Entities")
    if results["entities"]:
        entity_html = "<div class='entity-display'><table width='100%'><tr><th>Entity</th><th>Type</th></tr>"
        for entity in results["entities"]:
            entity_html += f"<tr><td>{entity['text']}</td><td>{entity['type']}</td></tr>"
        entity_html += "</table></div>"
        st.markdown(entity_html, unsafe_allow_html=True)
    else:
        st.markdown("<div class='entity-display'>No entities detected</div>", unsafe_allow_html=True)
    
    # Query complexity
    st.markdown("### Query Complexity")
    complexity = results["complexity"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <p>Complexity Level</p>
                <div class="metric-value">{complexity["level"]}</div>
                <p>Word Count: {complexity["word_count"]}</p>
                <p>Avg Word Length: {complexity["avg_word_length"]:.1f}</p>
            </div>
            """, unsafe_allow_html=True
        )
    with col2:
        complexity_factors = []
        if complexity["has_conditionals"]:
            complexity_factors.append("Contains conditional statements")
        if complexity["has_comparisons"]:
            complexity_factors.append("Contains comparisons")
        
        factors_html = "<ul>"
        if complexity_factors:
            for factor in complexity_factors:
                factors_html += f"<li>{factor}</li>"
        else:
            factors_html += "<li>Simple query structure</li>"
        factors_html += "</ul>"
        
        st.markdown(
            f"""
            <div class="metric-card">
                <p>Complexity Factors</p>
                {factors_html}
            </div>
            """, unsafe_allow_html=True
        )
    
    # JSON output
    st.markdown("### Raw Analysis Data")
    with st.expander("View JSON", expanded=False):
        st.json(results)
    
    # Download button for JSON
    json_str = json.dumps(results, indent=2)
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name=f"query_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )
    
else:
    st.markdown(
        """
        <div style='background:white; padding:2rem; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.05); border:1px solid #f3f4f6;'>
            <h3 style='margin-top:0; color:#1e40af;'>Welcome to QueryInsight Pro</h3>
            <p>Enter a search query above to analyze:</p>
            <ul>
                <li>Search intent classification</li>
                <li>Named entity extraction</li>
                <li>Category detection</li>
                <li>Sentiment analysis</li>
                <li>Relevance scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div style='margin-top:3rem; text-align:center; color:#6b7280; font-size:0.8rem;'>
        QueryInsight Pro • AI-Powered Search Query Analysis • v1.0
    </div>
    """, unsafe_allow_html=True)

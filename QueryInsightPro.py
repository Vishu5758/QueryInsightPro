import streamlit as st
import pandas as pd
import spacy
import re
import json
import time
from langdetect import detect, LangDetectException
from datetime import datetime
from transformers import pipeline
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="✨ QueryInsight Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS with more polished Google-like styling
st.markdown(
    """
<style>
    /* Global styling */
    body {
        font-family: 'Google Sans', 'Roboto', Arial, sans-serif;
        color: #202124;
        background-color: #f8f9fa;
    }
    
    h1 {
        color: #1a73e8;
        font-weight: 500;
        padding-bottom: 16px;
        border-bottom: 1px solid #e0e0e0;
        font-size: 2.125rem;
    }
    
    h2 {
        color: #202124;
        font-weight: 500;
        font-size: 1.5rem;
    }
    
    h3 {
        color: #202124;
        font-weight: 500;
        font-size: 1.25rem;
        margin-top: 24px;
        margin-bottom: 16px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 8px 24px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #1765cc;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border-radius: 4px;
        border: 1px solid #dadce0;
        padding: 12px 16px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #1a73e8;
        box-shadow: 0 0 0 1px #1a73e8;
    }
    
    /* Card styling */
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(60, 64, 67, 0.3), 0 1px 3px rgba(60, 64, 67, 0.15);
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 500;
        color: #1a73e8;
    }
    
    /* Results container */
    .results-container {
        background: #ffffff;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
        margin-bottom: 24px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Entity display */
    .entity-display mark {
        border-radius: 4px;
        padding: 2px 5px;
    }
    
    /* Example queries */
    .example-query {
        display: inline-block;
        background-color: #e8f0fe;
        color: #1a73e8;
        border-radius: 16px;
        padding: 4px 12px;
        margin-right: 8px;
        margin-bottom: 8px;
        font-size: 0.875rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #1a73e8;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_intent_model():
    return pipeline(
        "text-classification",
        model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    )


@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"
    )


# Load models
with st.spinner("Loading NLP models... This may take a moment."):
    nlp = load_nlp_model()
    intent_model = load_intent_model()
    sentiment_model = load_sentiment_model()


def normalize_query(query):
    query = query.lower()
    query = re.sub(r"\b(tmrrw|tmrw)\b", "tomorrow", query)
    query = re.sub(r"\bu\b", "you", query)
    query = re.sub(r"\br\b", "are", query)
    query = re.sub(r"\b2moro\b", "tomorrow", query)
    query = re.sub(r"\b2day\b", "today", query)
    query = re.sub(r"\b4\b", "for", query)
    query = re.sub(r"\bur\b", "your", query)
    query = re.sub(r"\bcuz\b|\bcause\b|\bcz\b", "because", query)
    query = re.sub(r"\&", "and", query)
    query = re.sub(r"[^\w\s]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query


def detect_intent(query):
    q = query.lower()
    # informational if it's a question
    if any(
        q.startswith(w) or f" {w} " in q
        for w in ["how", "what", "why", "when", "where"]
    ):
        return "informational"
    # transactional if it mentions buying or ordering
    if any(w in q for w in ["buy", "order", "purchase", "price", "book"]):
        return "transactional"
    # otherwise navigational
    return "navigational"


def extract_entities(query):
    doc = nlp(query)
    entities = {ent.label_: [] for ent in doc.ents}
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return entities


def detect_language(query):
    try:
        return detect(query)
    except LangDetectException:
        return "unknown"


def assign_category(query):
    query = query.lower()
    categories = {
        "travel": ["flight", "book", "ticket", "hotel", "vacation", "trip", "tourism"],
        "finance": ["loan", "bank", "stock", "invest", "money", "finance", "credit"],
        "health": [
            "covid",
            "vaccine",
            "doctor",
            "health",
            "medical",
            "symptom",
            "disease",
        ],
        "education": ["course", "learn", "study", "college", "university", "education"],
        "shopping": [
            "buy",
            "price",
            "order",
            "review",
            "shopping",
            "purchase",
            "product",
        ],
        "technology": ["computer", "software", "hardware", "tech", "digital", "app"],
        "entertainment": ["movie", "music", "game", "play", "show", "concert", "event"],
        "food": [
            "recipe",
            "restaurant",
            "food",
            "meal",
            "diet",
            "nutrition",
            "cooking",
        ],
    }

    custom_json = st.session_state.get("custom_categories")
    if custom_json:
        try:
            categories.update(json.loads(custom_json))
        except json.JSONDecodeError:
            pass

    scores = {cat: 0 for cat in categories}
    for cat, keywords in categories.items():
        for keyword in keywords:
            if keyword in query:
                scores[cat] += 1

    return max(scores, key=scores.get) if max(scores.values()) > 0 else "other"


def generate_relevance_score(query):
    length = len(query.split())
    specificity = min(len(set(query.split())) / max(1, length), 1)
    has_specific_terms = any(
        term in query.lower() for term in ["how", "why", "what", "when", "where"]
    )

    base_score = 0.4 + min(length, 8) / 10
    specificity_boost = specificity * 0.2
    clarity_boost = 0.1 if has_specific_terms else 0
    final_score = min(base_score + specificity_boost + clarity_boost, 0.99)

    return round(final_score, 2)


def detect_sentiment(query):
    raw = sentiment_model(query)[0]["label"]
    mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    return mapping.get(raw, raw.lower())


def analyze_query_complexity(query):
    words = query.split()
    avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
    sentences = re.split(r"[.!?]+", query)

    complexity = {
        "word_count": len(words),
        "avg_word_length": round(avg_word_length, 2),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "complexity_score": min(0.2 + (len(words) / 20) + (avg_word_length / 10), 0.99),
    }
    if complexity["complexity_score"] < 0.3:
        complexity["level"] = "Simple"
    elif complexity["complexity_score"] < 0.6:
        complexity["level"] = "Moderate"
    else:
        complexity["level"] = "Complex"
    return complexity


def annotate_query(query):
    normalized = normalize_query(query)
    complexity = analyze_query_complexity(query)
    return {
        "original_query": query,
        "normalized_query": normalized,
        "intent": detect_intent(normalized),
        "entities": extract_entities(normalized),
        "category": assign_category(normalized),
        "language": detect_language(normalized),
        "relevance_score": generate_relevance_score(normalized),
        "sentiment": detect_sentiment(normalized),
        "complexity": complexity,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# Sidebar with advanced options
with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center; padding-bottom:20px;">
        <h2 style="color:#1a73e8; font-weight:500;">🧠 QueryInsight Pro</h2>
        <p style="color:#5f6368; font-size:0.9rem;">Advanced Analysis Settings</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='margin-bottom:24px; border-color:#e0e0e0;'>", unsafe_allow_html=True)

    st.markdown(
        """
        <h3 style="color:#202124; font-weight:500; font-size:1.1rem;">🔧 Category Settings</h3>
        """, 
        unsafe_allow_html=True
    )
    
    with st.expander("Configure Custom Categories", expanded=False):
        default_categories = {
            "travel": ["flight", "book", "trip"],
            "finance": ["loan", "bank", "invest"],
        }

        custom_input = st.text_area(
            "Custom Categories (JSON format)",
            value=st.session_state.get(
                "custom_categories", json.dumps(default_categories, indent=2)
            ),
            height=200,
            help="Add custom categories with associated keywords in JSON format",
        )

        if st.button("💾 Save Categories", key="save_cats"):
            try:
                json.loads(custom_input)
                st.session_state["custom_categories"] = custom_input
                st.success("✅ Categories updated successfully!")
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON format. Please check your syntax.")

# Main content area
st.markdown(
    """
<div style="text-align:center; padding-bottom:16px;">
    <h1>✨ QueryInsight Pro</h1>
    <p style="color:#5f6368; font-size:1.1rem; margin-top:-8px;">
        AI-Powered Search Query Annotation Suite
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Query input section
st.markdown(
    """
<div class="results-container">
    <h3 style="color:#1a73e8; margin-top:0;">🔍 Enter Your Search Query</h3>
""",
    unsafe_allow_html=True,
)

query = st.text_input(
    "",
    placeholder="Type your search query here (e.g., 'best flights to New York next weekend')",
    key="query_input",
    help="Enter any search query to analyze its intent, entities, sentiment and more",
)

# Example queries as labels (not buttons)
st.markdown(
    """
<p style="font-size:0.9rem; color:#5f6368; margin:12px 0 16px 0;">
    Example queries:
</p>
<div>
    <span class="example-query">best flights to New York next weekend</span>
    <span class="example-query">how to invest in cryptocurrency 2025</span>
    <span class="example-query">symptoms of flu vs covid vaccine side effects</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)  # Close the results-container div

# Process query if submitted
if query:
    # Show analysis progress with better visual feedback
    with st.container():
        progress_placeholder = st.empty()
        status_text = st.empty()
        
        # Function to update progress with animation
        def update_progress(step, text):
            progress_placeholder.progress(step)
            status_text.markdown(f"<p style='color:#5f6368;'>{text}</p>", unsafe_allow_html=True)
            time.sleep(0.3)
        
        # Simulate processing steps with smoother transitions
        update_progress(10, "Normalizing query...")
        update_progress(30, "Analyzing intent...")
        update_progress(50, "Extracting entities...")
        update_progress(70, "Determining relevance...")
        update_progress(90, "Finalizing analysis...")
        
        # Get full analysis
        data = annotate_query(query)
        
        # Complete the progress
        update_progress(100, "Analysis complete!")
        time.sleep(0.5)
        
        # Clear the progress indicators
        progress_placeholder.empty()
        status_text.empty()

    # Display results in a nice layout
    st.markdown(
        """
    <div class="results-container">
        <h2 style="color:#1a73e8; margin-top:0;">📊 Query Analysis Results</h2>
    """,
        unsafe_allow_html=True,
    )

    # Display original and normalized query
    query_cols = st.columns(2)
    query_cols[0].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#5f6368; margin-bottom:8px; font-size:0.9rem;">Original Query</p>
        <p style="font-size:1.2rem; font-weight:500; color:#202124; margin-top:0;">{data["original_query"]}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    query_cols[1].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#5f6368; margin-bottom:8px; font-size:0.9rem;">Normalized Query</p>
        <p style="font-size:1.2rem; font-weight:500; color:#202124; margin-top:0;">{data["normalized_query"]}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Display key metrics
    st.markdown(
        "<h3 style='margin-top:32px;'>Key Insights</h3>", unsafe_allow_html=True
    )
    metric_cols = st.columns(4)

    # Intent with icon
    intent_icon = (
        "🔍"
        if data["intent"] == "informational"
        else "🧭" if data["intent"] == "navigational" else "🛒"
    )
    metric_cols[0].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#5f6368; margin-bottom:8px; font-size:0.9rem;">Intent</p>
        <p class="metric-value">{intent_icon} {data["intent"].capitalize()}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Category with icon
    category_icons = {
        "travel": "✈️",
        "finance": "💰",
        "health": "🏥",
        "education": "🎓",
        "shopping": "🛍️",
        "technology": "💻",
        "entertainment": "🎬",
        "food": "🍽️",
        "other": "📌",
    }
    category_icon = category_icons.get(data["category"], "📌")
    metric_cols[1].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#5f6368; margin-bottom:8px; font-size:0.9rem;">Category</p>
        <p class="metric-value">{category_icon} {data["category"].capitalize()}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sentiment with icon and color
    sentiment_icons = {"positive": "😊", "neutral": "😐", "negative": "😟"}
    sentiment_colors = {
        "positive": "#34a853",
        "neutral": "#5f6368",
        "negative": "#ea4335",
    }
    sentiment_icon = sentiment_icons.get(data["sentiment"], "😐")
    sentiment_color = sentiment_colors.get(data["sentiment"], "#5f6368")
    metric_cols[2].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#5f6368; margin-bottom:8px; font-size:0.9rem;">Sentiment</p>
        <p class="metric-value" style="color:{sentiment_color};">{sentiment_icon} {data["sentiment"].capitalize()}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Relevance score with visual indicator
    relevance_score = data["relevance_score"]
    score_color = (
        "#34a853"
        if relevance_score > 0.7
        else "#fbbc04" if relevance_score > 0.4 else "#ea4335"
    )
    metric_cols[3].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#5f6368; margin-bottom:8px; font-size:0.9rem;">Relevance Score</p>
        <p class="metric-value" style="color:{score_color};">{relevance_score}</p>
        <div style="background:#e8eaed; height:6px; border-radius:3px; margin-top:10px;">
            <div style="background:{score_color}; width:{relevance_score*100}%; height:6px; border-radius:3px;"></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Entity visualization with improved styling
    st.markdown(
        "<h3 style='margin-top:32px;'>Entity Visualization</h3>", unsafe_allow_html=True
    )

    doc = nlp(data["normalized_query"])
    entity_html = spacy.displacy.render(doc, style="ent", jupyter=False)
    
    # Replace the default styling with more Google-like styling
    entity_html = entity_html.replace("class=\"entities\"", "class=\"entities\" style=\"line-height: 2.5; font-family: 'Google Sans', 'Roboto', Arial, sans-serif;\"")
    
    st.markdown(
        f"""
    <div class="entity-display">
        {entity_html}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Complexity analysis visualization
    st.markdown(
        "<h3 style='margin-top:32px;'>Query Complexity</h3>", unsafe_allow_html=True
    )
    
    complexity_cols = st.columns(4)
    
    complexity_cols[0].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#5f6368; margin-bottom:8px; font-size:0.9rem;">Word Count</p>
        <p class="metric-value" style="color:#1a73e8;">{data["complexity"]["word_count"]}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    
    complexity_cols[1].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#5f6368; margin-bottom:8px; font-size:0.9rem;">Avg Word Length</p>
        <p class="metric-value" style="color:#1a73e8;">{data["complexity"]["avg_word_length"]}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    
    complexity_cols[2].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#5f6368; margin-bottom:8px; font-size:0.9rem;">Sentence Count</p>
        <p class="metric-value" style="color:#1a73e8;">{data["complexity"]["sentence_count"]}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    
    complexity_level = data["complexity"]["level"]
    level_color = {
        "Simple": "#34a853",
        "Moderate": "#fbbc04",
        "Complex": "#ea4335"
    }.get(complexity_level, "#1a73e8")
    
    complexity_cols[3].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#5f6368; margin-bottom:8px; font-size:0.9rem;">Complexity Level</p>
        <p class="metric-value" style="color:{level_color};">{complexity_level}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Editing section with better styling
    st.markdown(
        "<h3 style='margin-top:32px;'>Edit Annotations</h3>", unsafe_allow_html=True
    )

    with st.form("edit_annotations_form"):
        col1, col2 = st.columns(2)

        with col1:
            normalized_query = st.text_input(
                "Normalized Query", value=data["normalized_query"]
            )
            intent = st.selectbox(
                "Intent",
                ["informational", "navigational", "transactional"],
                index=["informational", "navigational", "transactional"].index(
                    data["intent"]
                ),
            )
            category = st.text_input("Category", value=data["category"])
            language = st.text_input("Language", value=data["language"])

        with col2:
            relevance_score = st.slider(
                "Relevance Score",
                0.0,
                1.0,
                float(data["relevance_score"]),
                step=0.01,
                format="%.2f",
            )
            sentiment = st.selectbox(
                "Sentiment",
                ["positive", "neutral", "negative"],
                index=["positive", "neutral", "negative"].index(data["sentiment"]),
            )
            entities_text = st.text_area(
                "Entities (JSON format)",
                value=json.dumps(data["entities"], indent=2),
                height=150,
            )

        submitted = st.form_submit_button("💾 Save & Generate JSON")

        if submitted:
            try:
                entities = json.loads(entities_text)
                output = {
                    "original_query": data["original_query"],
                    "normalized_query": normalized_query,
                    "intent": intent,
                    "category": category,
                    "language": language,
                    "relevance_score": relevance_score,
                    "sentiment": sentiment,
                    "entities": entities,
                    "complexity": data["complexity"],
                    "timestamp": data["timestamp"],
                }
                download_data = json.dumps(output, indent=2)
                st.session_state["download_json"] = download_data
                st.success(
                    "✅ Annotation saved successfully! Download your JSON below."
                )
            except json.JSONDecodeError:
                st.error(
                    "❌ Invalid JSON format in Entities field. Please check your syntax."
                )

    # Download button outside the form with improved styling
    if st.session_state.get("download_json"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Download JSON",
                data=st.session_state["download_json"],
                file_name=f"query_annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_button",
            )

    # Entity distribution chart with better styling
    entity_types = list(data["entities"].keys())
    entity_counts = [len(data["entities"][et]) for et in entity_types]

    if entity_types:
        st.markdown("<h3 style='margin-top:32px;'>Entity Distribution</h3>", unsafe_allow_html=True)
        chart_data = pd.DataFrame({"Entity Type": entity_types, "Count": entity_counts})
        fig = px.bar(
            chart_data,
            x="Entity Type",
            y="Count",
            color="Entity Type",
            color_discrete_sequence=px.colors.qualitative.Google,
        )
        fig.update_layout(
            plot_bgcolor="white",
            margin=dict(t=0, l=0, r=0, b=0),
            xaxis=dict(
                title_font=dict(size=14, family="Roboto, Arial"),
                tickfont=dict(family="Roboto, Arial"),
                gridcolor="#f0f0f0",
            ),
            yaxis=dict(
                title_font=dict(size=14, family="Roboto, Arial"),
                tickfont=dict(family="Roboto, Arial"),
                gridcolor="#f0f0f0",
            ),
            legend_title_font=dict(family="Roboto, Arial"),
            legend_font=dict(family="Roboto, Arial"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No entities were detected in this query.")

    st.markdown("</div>", unsafe_allow_html=True)  # Close results-container div
else:
    # Welcome screen when no query is entered - more polished Google-like welcome
    st.markdown(
        """
    <div class="results-container" style="text-align:center; padding:48px 24px;">
        <img src="https://www.gstatic.com/images/branding/product/2x/search_48dp.png" alt="Google Search" width="72" style="margin-bottom:24px;">
        <h2 style="color:#1a73e8; font-weight:500; margin-bottom:16px;">Welcome to QueryInsight Pro</h2>
        <p style="color:#5f6368; max-width:600px; margin:0 auto 24px auto; font-size:1.1rem;">
            Enter a search query above to begin your comprehensive query analysis
        </p>
        <div style="background:#f8f9fa; border-radius:8px; padding:16px; max-width:600px; margin:0 auto; text-align:left;">
            <p style="color:#5f6368; margin-bottom:8px; font-weight:500;">What you can analyze:</p>
            <ul style="color:#5f6368; margin-left:24px; padding-left:0;">
                <li>Query intent and category classification</li>
                <li>Named entity recognition</li>
                <li>Sentiment analysis</li>
                <li>Relevance scoring</li>
                <li>Query complexity metrics</li>
            </ul>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

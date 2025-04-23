import streamlit as st
import pandas as pd
import spacy
import re
import json
import time
from langdetect import detect
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

# Add custom CSS (simplified)
st.markdown(
    """
<style>
    h1 {
        color: #1E3A8A;
        font-weight: 700;
        padding-bottom: 20px;
        border-bottom: 2px solid #e0e0e0;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 6px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e6e6e6;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1E3A8A;
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
    # informational if it’s a question
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


from langdetect import detect, LangDetectException

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
    # ─── BUG-FIXED: map LABEL_{0,1,2} to negative/neutral/positive ───────────────────────────────────
    raw = sentiment_model(query)[0]["label"]
    mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    return mapping.get(raw, raw.lower())
    # ───────────────────────────────────────────────────────────────────────────────────────────────────


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


# ───────────────────────────────────────────────────────────────────────────────────────────────────
# Rest of your UI, forms, download button placement, charts, etc., remain **unchanged** from before.
# ───────────────────────────────────────────────────────────────────────────────────────────────────


# Sidebar with advanced options
with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center; padding-bottom:20px;">
        <h2 style="color:#1E3A8A;">🧠 QueryInsight Pro</h2>
        <p style="color:#64748b;">Advanced Analysis Options</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.header("🔧 Category Settings")
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
<div style="text-align:center; padding-bottom:10px;">
    <h1 style="font-size:2.5rem;">✨ QueryInsight Pro: AI-Powered Search Query Annotation Suite</h1>
    <p style="color:#64748b; font-size:1.1rem;">
        Analyze, annotate and enhance search queries with state-of-the-art AI technology
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Query input section
st.markdown(
    """
<div class="results-container" style="background:linear-gradient(135deg,
#EFF6FF 0%, #FFFFFF 100%);">
    <h3 style="color:#1E3A8A; margin-top:0;">🔍 Enter Your Search Query</h3>
""",
    unsafe_allow_html=True,
)

query = st.text_input(
    "",
    placeholder="Type your search query here (e.g., 'best flights to New York next weekend')",
    key="query_input",
    help="Enter any search query to analyze its intent, entities, sentiment and more",
)

# Example queries (now just static labels, not buttons)
st.markdown(
    """
<p style="font-size:0.9rem; color:#64748b; margin-bottom:5px;">
    Example queries:&nbsp;
    <span style="margin-right:15px;">best flights to New York next weekend</span>
    <span style="margin-right:15px;">how to invest in cryptocurrency 2025</span>
    <span>symptoms of flu vs covid vaccine side effects</span>
</p>
""",
    unsafe_allow_html=True,
)


# Process query if submitted
if query:
    # Show analysis progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Simulate processing steps
    status_text.text("Normalizing query...")
    progress_bar.progress(10)
    time.sleep(0.3)

    status_text.text("Analyzing intent...")
    progress_bar.progress(30)
    time.sleep(0.3)

    status_text.text("Extracting entities...")
    progress_bar.progress(50)
    time.sleep(0.3)

    status_text.text("Determining relevance...")
    progress_bar.progress(70)
    time.sleep(0.3)

    status_text.text("Finalizing analysis...")
    progress_bar.progress(90)
    time.sleep(0.3)

    # Get full analysis
    data = annotate_query(query)

    # Complete the progress
    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    time.sleep(0.5)

    # Clear the progress indicators
    progress_bar.empty()
    status_text.empty()

    # Display results in a nice layout
    st.markdown(
        """
    <div class="results-container">
        <h2 style="color:#1E3A8A; margin-top:0;">📊 Query Analysis Results</h2>
    """,
        unsafe_allow_html=True,
    )

    # Display original and normalized query
    query_cols = st.columns(2)
    query_cols[0].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#64748b; margin-bottom:5px;">Original Query</p>
        <p style="font-size:1.2rem; font-weight:500; color:#1E3A8A; margin-top:0;">{data["original_query"]}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    query_cols[1].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#64748b; margin-bottom:5px;">Normalized Query</p>
        <p style="font-size:1.2rem; font-weight:500; color:#1E3A8A; margin-top:0;">{data["normalized_query"]}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Display key metrics
    st.markdown(
        "<h3 style='margin-top:25px;'>Key Insights</h3>", unsafe_allow_html=True
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
        <p style="color:#64748b; margin-bottom:5px;">Intent</p>
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
        <p style="color:#64748b; margin-bottom:5px;">Category</p>
        <p class="metric-value">{category_icon} {data["category"].capitalize()}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sentiment with icon and color
    sentiment_icons = {"positive": "😊", "neutral": "😐", "negative": "😟"}
    sentiment_colors = {
        "positive": "#10B981",
        "neutral": "#6B7280",
        "negative": "#EF4444",
    }
    sentiment_icon = sentiment_icons.get(data["sentiment"], "😐")
    sentiment_color = sentiment_colors.get(data["sentiment"], "#6B7280")
    metric_cols[2].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#64748b; margin-bottom:5px;">Sentiment</p>
        <p class="metric-value" style="color:{sentiment_color};">{sentiment_icon} {data["sentiment"].capitalize()}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Relevance score with visual indicator
    relevance_score = data["relevance_score"]
    score_color = (
        "#10B981"
        if relevance_score > 0.7
        else "#FBBF24" if relevance_score > 0.4 else "#EF4444"
    )
    metric_cols[3].markdown(
        f"""
    <div class="metric-card">
        <p style="color:#64748b; margin-bottom:5px;">Relevance Score</p>
        <p class="metric-value" style="color:{score_color};">{relevance_score}</p>
        <div style="background:#e2e8f0; height:6px; border-radius:3px; margin-top:10px;">
            <div style="background:{score_color}; width:{relevance_score*100}%; height:6px; border-radius:3px;"></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Entity visualization
    st.markdown(
        "<h3 style='margin-top:25px;'>Entity Visualization</h3>", unsafe_allow_html=True
    )

    doc = nlp(data["normalized_query"])
    entity_html = spacy.displacy.render(doc, style="ent", jupyter=False)
    st.markdown(
        f"""
    <div class="entity-display">
        {entity_html}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Editing section
    st.markdown(
        "<h3 style='margin-top:25px;'>Edit Annotations</h3>", unsafe_allow_html=True
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

    # Download button outside the form
    if st.session_state.get("download_json"):
        st.download_button(
            label="📥 Download JSON",
            data=st.session_state["download_json"],
            file_name=f"{data['original_query'].replace(' ', '_')}_annotation.json",
            mime="application/json",
        )

    # Entity distribution chart
    entity_types = list(data["entities"].keys())
    entity_counts = [len(data["entities"][et]) for et in entity_types]

    if entity_types:
        chart_data = pd.DataFrame({"Entity Type": entity_types, "Count": entity_counts})
        fig = px.bar(
            chart_data,
            x="Entity Type",
            y="Count",
            title="Entity Distribution",
            color="Entity Type",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No entities were detected in this query.")
else:
    # Welcome screen when no query is entered
    st.markdown(
        """
    <div class="results-container" style="text-align:center; padding:40px 20px;">
        <h2 style="color:#1E3A8A;">Welcome to QueryInsight Pro</h2>
        <p style="color:#64748b; max-width:600px; margin:0 auto 20px auto;">
            Enter a search query above to start analyzing .
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

import streamlit as st
import joblib
import pandas as pd

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🎬 Movie Sentiment Analyzer")

st.sidebar.info(
"""
### Machine Learning Project

**Algorithm**
- Logistic Regression

**Vectorizer**
- TF-IDF

**Dataset**
- IMDB 50K Reviews

Developed using:
- Streamlit
- Scikit-learn
- Python
"""
)

# -------------------------------
# Title
# -------------------------------
st.title("🎬 Movie Review Sentiment Analysis")

st.write(
"Predict whether a movie review is **Positive 😊** or **Negative 😔**."
)

st.divider()

# -------------------------------
# Movie Selection
# -------------------------------
movie = st.selectbox(
    "Select Movie",
    [
        "Leo",
        "Jailer",
        "Vikram",
        "Master",
        "Kaithi",
        "KGF",
        "Pushpa",
        "RRR",
        "Other"
    ]
)

# -------------------------------
# Review Input
# -------------------------------
review = st.text_area(
    "Enter your movie review",
    height=180,
    placeholder="Example: Leo movie is fantastic. Vijay's acting was amazing!"
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)[0]

        probability = model.predict_proba(review_vector)[0]

        confidence = max(probability) * 100

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😔 Negative Review")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Prediction Probability")

        chart = pd.DataFrame(
            {
                "Sentiment": ["Negative", "Positive"],
                "Probability": probability
            }
        )

        st.bar_chart(chart.set_index("Sentiment"))

        st.subheader("Review Statistics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Movie", movie)

        col2.metric("Words", len(review.split()))

        col3.metric("Characters", len(review))

        st.divider()

        st.subheader("Your Review")

        st.write(review)

# -------------------------------
# Footer
# -------------------------------
st.divider()

st.markdown(
"""
### 🚀 Technologies Used

- Python
- Streamlit
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Joblib
"""
)
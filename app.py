import streamlit as st
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import joblib
import time

# Configure page
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }

    .result-container {
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .spam-result {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
    }

    .ham-result {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }

    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }

    .feature-box {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(78, 205, 196, 0.3);
        margin: 1rem 0;
    }

    .stats-box h4, .feature-box h4 {
        color: white;
        margin-bottom: 1rem;
    }

    .stats-box p {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Porter Stemmer
ps = PorterStemmer()


def transform_text(text):
    """Enhanced text preprocessing with progress indication"""
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load models with error handling
@st.cache_resource
def load_models():
    try:
        tfidf = joblib.load('vectorizer.joblib')
        mnb = joblib.load('model_mnb.joblib')
        return tfidf, mnb
    except FileNotFoundError:
        st.error(
            "⚠️ Model files not found! Please ensure 'vectorizer.joblib' and 'model_mnb.joblib' are in the same directory.")
        return None, None


# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Smart Spam Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Protect yourself from spam messages with AI-powered detection</p>',
                unsafe_allow_html=True)

    # Sidebar with information
    with st.sidebar:
        st.markdown("### 📊 About This Tool")
        st.markdown("""
        <div class="feature-box">
            <h4>🤖 How it works:</h4>
            <ul>
                <li>Text preprocessing & cleaning</li>
                <li>Feature extraction using TF-IDF</li>
                <li>Classification with Naive Bayes</li>
                <li>Real-time spam detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🚀 Features")
        st.markdown("""
        - ⚡ **Fast Processing**
        - 🎯 **High Accuracy**
        - 📱 **SMS & Email Support**
        - 🔒 **Privacy Focused**
        """)

        st.markdown("### 💡 Tips")
        st.info("For best results, paste the complete message including any suspicious links or formatting.")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Enter Your Message")
        input_sms = st.text_area(
            "Paste your email or SMS content here:",
            height=200,
            placeholder="Enter the message you want to check for spam...",
            help="Paste any email or SMS message to check if it's spam or legitimate"
        )

        # Button with custom styling
        predict_button = st.button('🔍 Analyze Message', type='primary', use_container_width=True)

    with col2:
        st.markdown("### 📈 Quick Stats")
        st.markdown("""
        <div class="stats-box">
            <h4>🎯 Model Accuracy</h4>
            <p style="font-size: 2rem; margin: 0;"><strong>~95%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="stats-box">
            <h4>⚡ Processing Speed</h4>
            <p style="font-size: 2rem; margin: 0;"><strong>< 1s</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # Prediction logic
    if predict_button:
        if input_sms.strip():
            # Load models
            tfidf, mnb = load_models()

            if tfidf is not None and mnb is not None:
                # Show processing
                with st.spinner('🔄 Analyzing message...'):
                    time.sleep(0.5)  # Small delay for UX

                    # Process the message
                    transformed_sms = transform_text(input_sms)
                    vector_input = tfidf.transform([transformed_sms])
                    result = mnb.predict(vector_input)[0]
                    confidence = mnb.predict_proba(vector_input)[0]

                # Display results
                st.markdown("---")

                if result == 1:
                    st.markdown("""
                    <div class="result-container spam-result">
                        <h2>🚨 SPAM DETECTED</h2>
                        <p style="font-size: 1.2rem;">This message appears to be spam!</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.error("⚠️ **Warning**: This message shows characteristics of spam. Be cautious of:")
                    st.markdown("""
                    - Suspicious links or attachments
                    - Requests for personal information
                    - Urgent or threatening language
                    - Too-good-to-be-true offers
                    """)
                else:
                    st.markdown("""
                    <div class="result-container ham-result">
                        <h2>✅ LEGITIMATE MESSAGE</h2>
                        <p style="font-size: 1.2rem;">This message appears to be safe!</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.success("✅ **Good news**: This message appears to be legitimate and safe.")

                # Show confidence scores
                st.markdown("### 📊 Confidence Scores")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Not Spam",
                        value=f"{confidence[0]:.1%}",
                        delta=None
                    )

                with col2:
                    st.metric(
                        label="Spam",
                        value=f"{confidence[1]:.1%}",
                        delta=None
                    )

                # Progress bar for confidence
                st.progress(confidence[1])

                # Show processed text (expandable)
                with st.expander("🔍 View Processed Text"):
                    st.code(transformed_sms, language=None)
        else:
            st.warning("⚠️ Please enter a message to analyze!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🛡️ Stay safe online! Always verify suspicious messages through official channels.</p>
        <p><small>Powered by Machine Learning • Built with Streamlit</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
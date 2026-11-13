import streamlit as st
import joblib

# Load the trained model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_models()

st.title("📩 SMS Spam Detection")
st.write("Enter an SMS message below to detect whether it is Spam or Not Spam (Ham).")

# GUI feature: SMS message input
user_input = st.text_area("SMS Message:", placeholder="Paste text message here...", height=150)

# GUI feature: Spam Detection button
if st.button("Spam Detection"):
    if user_input.strip():
        # Transform input and predict
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]
        
        st.divider()
        
        # GUI feature: Display Spam or Not Spam & Prediction confidence
        if prediction == 1:
            st.error("🚨 **Classification: Spam**")
            st.write(f"**Confidence:** {probabilities[1]:.2%}")
        else:
            st.success("✅ **Classification: Not Spam (Ham)**")
            st.write(f"**Confidence:** {probabilities[0]:.2%}")
    else:
        st.warning("Please enter an SMS message to analyze.")
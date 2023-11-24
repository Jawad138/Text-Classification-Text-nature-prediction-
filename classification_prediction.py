import streamlit as st
from transformers import pipeline

# Load pre-trained model from Hugging Face
classifier = pipeline('sentiment-analysis')

# Streamlit app
def main():
    st.title("Text Classification App")

    # User input
    user_input = st.text_area("Enter text for classification:")

    if st.button("Classify"):
        if user_input:
            # Get classification prediction
            prediction = classify_text(user_input)
            st.success(f"Prediction: {prediction}")
        else:
            st.warning("Please enter text for classification.")

# Function to perform classification
def classify_text(text):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    return f"{label} (confidence: {score:.2f})"

if __name__ == "__main__":
    main()

import streamlit as st
from transformers import pipeline

# Load pre-trained text classification model (sentiment analysis)
classifier = pipeline('sentiment-analysis')

# Streamlit web application
def main():
    st.title("Text Classification App")

    # Input text area
    text_input = st.text_area("Enter your text here:", "Type your text here...")

    # Button to make predictions
    if st.button("Get Classification"):
        # Make prediction
        result = classifier(text_input)

        # Display result
        st.write("Prediction:")
        st.write(f"Label: {result[0]['label']}")
        st.write(f"Score: {result[0]['score']:.4f}")

if __name__ == "__main__":
    main()

# streamlit_app.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelWithHead

print("Import Sucessful")

model_name = "/content/fine_tuned_disBERTU"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Using the same function used for Manual Testing
def example_input(text):
    inputs = tokenizer.encode(text, return_tensors="pt").to("cuda")
    outputs = model(inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    return id2label[predictions.tolist()[0]]

# Load the model with adapter
model = AutoModelWithHead.from_pretrained(model_name, adapter_config="pfeiffer")

# Streamlit app
st.title("Movie Review Sentiment Analysis App")

# User input text box
user_input = st.text_area("Enter your movie review here:")

# Sentiment analysis button
if st.button("Analyze Sentiment"):
    st.success(example_input(user_input))

# Footer
st.markdown("---")
st.write("This app uses a pre-trained DistilBERT model with adapter for sentiment analysis on movie reviews.")
import streamlit as st
import numpy as np
#import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the model and tokenizer
model = load_model('cnn_lstm.h5')

#with open('tokenizer.pickle', 'rb') as handle:
#    tokenizer = pickle.load(handle)


# Load or define the tokenizer
# Ensure you have the same tokenizer used for training
#tokenizer = Tokenizer(num_words=max_words)
#tokenizer.fit_on_texts(training_texts)  # or load from a saved tokenizer if you saved it


# Function to preprocess the input text
def preprocess_input(text):
    # Tokenize and pad the input text
    tokenizer = Tokenizer(num_words=150)
    tokenizer.fit_on_texts([text])  # or load from a saved tokenizer if you saved it
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=150)  # Adjust maxlen as needed
    return padded_sequences

# Streamlit app layout

# Add the header image
st.image("image.jpg", use_column_width=True)


st.title("Email Spam/Ham Classifier")
st.write("This app uses a CNN-LSTM model to classify emails as spam or ham based on the input text. Simply enter your email content below and click 'Predict' to see the result.")

# Input section
st.subheader("Input Section")
st.write("Enter your email content to get a prediction:")
input_text = st.text_area("Input Text")

if st.button("Predict"):
    if input_text:
        # Preprocess input text
        processed_input = preprocess_input(input_text)
        
        # Make prediction
        prediction = model.predict(processed_input)
        
        # Assuming the model's output is a binary classification with 0 for Ham and 1 for Spam
        st.write("Prediction:", "Spam" if prediction[0][0] > 0.5 else "Ham")
    else:
        st.write("Please enter some text to predict.")

# Project details
st.subheader("About the Project")
st.write("""
This project is an Email Spam/Ham Classifier that leverages a Convolutional Neural Network (CNN) combined with a Long Short-Term Memory (LSTM) network.
The model is trained to analyze the content of emails and predict whether they are spam or ham. 
This tool can be useful for filtering unwanted emails and keeping your inbox clean.
""")

# Usage instructions
st.subheader("How to Use")
st.write("""
1. Enter the text of the email you want to classify in the text area above.
2. Click the 'Predict' button.
3. The app will display whether the email is classified as 'Spam' or 'Ham'.
""")

# Contact information
st.subheader("Contact")
st.write("""
If you have any questions, feedback, or want to know more about the project, feel free to reach out to the creator at:

**Name:** Kaosarat Ololade Malik, Advanced Computer Science(Msc), Informatics and Engineering Department 
         
**Email:** km744@sussex.ac.uk 
         
**GitHub:** [Your GitHub Profile](https://github.com/tttt)  
""")

# Footer with a cool design
st.markdown("---")
st.write("**Thank you for using the Email Spam/Ham Classifier!**")
st.markdown("Made with ❤️ by Kaosarat Ololade Malik.")

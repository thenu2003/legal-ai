
import streamlit as st
from pdf_extractor import process_uploaded_file
import speech_recognition as sr
from langdetect import detect
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import tempfile
# from IPC_ID_NLP import train_ipc_model, predict_ipc_section
# from CRPC_ID_NLP import train_crpc_model, predict_crpc_section
import pytesseract
from PIL import Image
from pdf_extractor import process_uploaded_file
import speech_recognition as sr
from langdetect import detect
import tempfile
from googletrans import Translator


# Function to detect language using langdetect library
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Function to transcribe audio and get the transcribed text
def transcribe_audio():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Speak something...")
        audio = recognizer.listen(source)
        st.success("Audio recorded successfully!")

    try:
        st.subheader("Transcription:")
        text = recognizer.recognize_google(audio)
        detected_language = detect(text)  # Detect the language of the transcribed text
        st.write(f"Detected Language: {detected_language}")
        st.write(f"Transcribed Text: {text}")
        return text, detected_language
    except sr.UnknownValueError:
        st.warning("Speech Recognition could not understand audio.")
        return None, None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None, None

# Function to translate text to English
def translate_to_english(text, source_language):
    translator = Translator()
    translation = translator.translate(text, src=source_language, dest="en")
    return translation.text

# Function to translate text to Hindi
def translate_to_hindi(text, source_language):
    translator = Translator()
    translation = translator.translate(text, src=source_language, dest="hi")
    return translation.text

# 1) Prompt entering section with Audio ico
# Load IPC data
ipc_data = pd.read_json('data\ipc.json').fillna('UNKNOWN')
label_encoder_ipc = LabelEncoder()
ipc_data['label'] = label_encoder_ipc.fit_transform(ipc_data['section_desc'])
train_data_ipc, _ = train_test_split(ipc_data, test_size=0.2, random_state=42)
vectorizer_ipc = TfidfVectorizer(max_features=1000)
X_train_ipc = vectorizer_ipc.fit_transform(train_data_ipc['section_desc']).toarray()
y_train_ipc = train_data_ipc['label'].values
model_ipc = Sequential([
    Dense(32, activation='relu', input_dim=1000),
    Dense(len(label_encoder_ipc.classes_), activation='softmax')
])
model_ipc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_ipc.fit(X_train_ipc, y_train_ipc, epochs=100, batch_size=32)

# Load CRPC data
crpc_data = pd.read_json('data\crpc.json').fillna('UNKNOWN')
label_encoder_crpc = LabelEncoder()
crpc_data['label'] = label_encoder_crpc.fit_transform(crpc_data['section_desc'])
train_data_crpc, _ = train_test_split(crpc_data, test_size=0.2, random_state=42)
vectorizer_crpc = TfidfVectorizer(max_features=1000)
X_train_crpc = vectorizer_crpc.fit_transform(train_data_crpc['section_desc']).toarray()
y_train_crpc = train_data_crpc['label'].values
model_crpc = Sequential([
    Dense(32, activation='relu', input_dim=1000),
    Dense(len(label_encoder_crpc.classes_), activation='softmax')
])
model_crpc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_crpc.fit(X_train_crpc, y_train_crpc, epochs=100, batch_size=32)

# Streamlit UI
st.set_page_config(page_title="LegalAssist", page_icon="🔍", layout="wide")
st.header("🔍 LegalAssist - AI for Legal Section Suggestions")

st.sidebar.title("Rajasthan Police Hackathon")
st.sidebar.markdown("Welcome to the Rajasthan Police Hackathon application. Use the tools and options available to extract text from various formats, transcribe audio, and get legal section suggestions.")

import nltk
import streamlit as st
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Function to perform sentiment analysis
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score

# Function to create embedding matrix
def create_embedding_matrix(model, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, model.vector_size))
    for word, i in word_index.items():
        if word in model.wv:
            embedding_matrix[i] = model.wv[word]
    return embedding_matrix
# Function to train IPC model
def train_ipc_model(user_input):
    # Load IPC dataset (adjust file path accordingly)
    with open('ipc.json', encoding='utf-8') as f:
        ipc_data = pd.read_json(f)

    # Handle missing values (replace NaN with a placeholder)
    ipc_data = ipc_data.fillna('UNKNOWN')

    # Encode categorical labels
    label_encoder_ipc = LabelEncoder()
    ipc_data['label'] = label_encoder_ipc.fit_transform(ipc_data['section_desc'])

    # Split the data into training and testing sets
    train_data_ipc, test_data_ipc = train_test_split(ipc_data, test_size=0.2, random_state=42)

    # Tokenize text data (using TF-IDF for simplicity, you may need a more sophisticated approach)
    vectorizer_ipc = TfidfVectorizer(max_features=1000)
    X_train_ipc = vectorizer_ipc.fit_transform(train_data_ipc['section_desc']).toarray()
    y_train_ipc = train_data_ipc['label'].values

    # Define the model for IPC
    model_ipc = Sequential([
        Dense(32, activation='relu', input_dim=1000),
        Dense(len(label_encoder_ipc.classes_), activation='softmax')
    ])

    # Compile the model
    model_ipc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model_ipc.fit(X_train_ipc, y_train_ipc, epochs=75, batch_size=32)

    # Tokenize and predict IPC sections for user input
    user_input_vectorized = vectorizer_ipc.transform([user_input]).toarray()
    predicted_ipc_probs = model_ipc.predict(user_input_vectorized)

    # Decode the predicted IPC section
    predicted_ipc_label = label_encoder_ipc.inverse_transform([predicted_ipc_probs.argmax()])[0]

    return model_ipc, vectorizer_ipc, label_encoder_ipc, predicted_ipc_label
import streamlit as st
import google.generativeai as genai
import textwrap
from IPython.display import display, Markdown
import os

# Set up Google Generative AI with your API key
GOOGLE_API_KEY = os.getenv('AIzaSyCkkja3hTj8nwDwbZWcT7eEzQUYK2UsvVE')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Function to generate content using the entered text
def generate_content(text):
    response = model.generate_content([text+"semantic representation like uml"])
    response.resolve()
    return response.generated_content[0]

def generate_content(text):
    response = model.generate_content([text+"semantic representation like uml diagram"])
    response.resolve()
    return response.generated_content[0]

# Function to transcribe audio (replace with your actual implementation)
def transcribe_audio():
    # Replace with your audio transcription logic
    return "Transcribed Text", "Detected Language"

# 1) Prompt entering section with Audio icon
with st.form(key="prompt_form"):
    st.subheader("Prompt Section")

    # Create a microphone icon using Unicode
    audio_icon = "🎤"

    # Add an audio button to switch input mode
    audio_button_clicked = st.form_submit_button(audio_icon)
    
    # Use transcribed text as the default input for the prompt if the audio button is clicked
    transcribed_text = transcribe_audio() if audio_button_clicked else None
    gen_inp = st.text_input("Enter some text:", transcribed_text if transcribed_text else "Default Text")
    st.write("You entered:", gen_inp)
    prompt_submit = st.form_submit_button("Submit Prompt")

    # Display audio icon button status
    st.write("Audio Recording Status:", "Recording..." if audio_button_clicked else "Not Recording")

if st.button("Generate Answer"):
    st.spinner("Generating answer...")

    new_description_ipc = gen_inp
    new_description_vectorized_ipc = vectorizer_ipc.transform([new_description_ipc]).toarray()
    predicted_ipc_probs = model_ipc.predict(new_description_vectorized_ipc)
    predicted_ipc_label = label_encoder_ipc.inverse_transform([predicted_ipc_probs.argmax()])[0]
        
    
    new_description_crpc = gen_inp
    new_description_vectorized_crpc = vectorizer_crpc.transform([new_description_crpc]).toarray()
    predicted_crpc_probs = model_crpc.predict(new_description_vectorized_crpc)
    predicted_crpc_label = label_encoder_crpc.inverse_transform([predicted_crpc_probs.argmax()])[0] 


    # Display the results

    if not any(new_description_vectorized_ipc.flatten()):
        st.write("Predicted IPC Section Information:\n", "No IPC Section Found")
    else:
        matching_rows_ipc = ipc_data[ipc_data['section_desc'] == predicted_ipc_label].iloc[0]
        st.write("Predicted IPC Section Information:\n", matching_rows_ipc)

    if not any(new_description_vectorized_crpc.flatten()):
        st.write("Predicted CRPC Section Information:\n", "No CRPC Section Found")
    else:
        matching_rows_crpc = crpc_data[crpc_data['section_desc'] == predicted_crpc_label].iloc[0]
        st.write("Predicted CRPC Section Information:\n", matching_rows_crpc)




# Check if the audio icon button is clicked
#if audio_button_clicked:
    # Start transcribing the recorded audio
   # transcribed_text = transcribe_audio()
   # st.subheader("Transcription:")
   # st.write(transcribed_text)
   # st.text("You entered: " + transcribed_text)

translator = Translator()
# 4) File upload section
with st.form(key="file_upload_form"):
    st.subheader("File Upload Section")
    uploaded_file = st.file_uploader(
        "Upload a pdf, docx, or txt file",
        type=["pdf", "docx", "txt"],
        help="Supported image formats: JPG, JPEG, PNG, GIF",
    )
    file_submit = st.form_submit_button("Submit File")

    if file_submit and uploaded_file is not None:
    # Check the file suffix
        file_suffix = uploaded_file.name.split(".")[-1].lower()

        if file_suffix == "pdf":
            # If it's a PDF, process as usual
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Process the uploaded PDF file
            text_result = process_uploaded_file(temp_file_path)

            # Display extracted text
            if text_result:
                for i, text in enumerate(text_result, start=1):
                    st.text(f"Page {i}:\n{text}\n")
                    text_english = translator.translate(text, src='hi', dest='en').text
                    st.header("Translated Text:")
                st.write(text_english)
            else:
                st.warning("Unsupported file type. Please upload a PDF.")
    # Display the translated English text using Streamlit
from googletrans import Translator
def load_and_prep(file):
    img = Image.open(file).convert("RGB")
    return img
def translate_hindi_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='hi', dest='en')
    return translation.origin, translation.text

if 'text_hindi' not in st.session_state:
    st.session_state.text_hindi = ""

with st.form(key="image_upload_form"):
    st.subheader("Image Upload Section")
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Supported image formats: JPG, JPEG, PNG",
    )
    image_submit = st.form_submit_button("Submit Image")

    if image_submit and uploaded_image is not None:
        # If it's an image, perform OCR
        img = load_and_prep(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.session_state.text_hindi = pytesseract.image_to_string(img, lang='hin')

        st.header("Extracted Hindi Text:")
        st.write(st.session_state.text_hindi)

        if st.session_state.text_hindi:
            # Translate Hindi text to English
            input_text, english_translation = translate_hindi_to_english(st.session_state.text_hindi)
            st.header("English Translation:")
            st.write(english_translation)

# 3) Advanced sections
with st.expander("Advanced Options"):
    st.subheader("Advanced Options Section")
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")
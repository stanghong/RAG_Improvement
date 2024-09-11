# %%
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

import sys
from rag_chromadb_qa import load_documents, chunk_documents, chromadb_retrieval_qa
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 1. Voice to wav file
# %%
# Define the sample rate (samples per second) and duration of the recording
sample_rate = 44100  # Standard sample rate for audio
duration = 10  # Duration in seconds

# Prompt the user to start recording
print("Recording audio from the microphone for {} seconds...".format(duration))
# %%
# Capture audio data from the microphone
recorded_audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
sd.wait()  # Wait until the recording is finished

# Save the recorded audio to a file (e.g., 'output_microphone_audio.wav')
write("output_microphone_audio.wav", sample_rate, np.array(recorded_audio))

print("Recording complete. Audio saved to 'output_microphone_audio.wav'.")

# 2. Voice to Text: Whisper
client = OpenAI()
# %%
audio_file = open("output_microphone_audio.wav", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text"
)

print(transcription)

# %%
# RAG: chromaDB
# %%
file_name ='./data/Self-correcting LLM-controlled Diffusion Models.pdf'
question = transcription

# Load the documents
data = load_documents(file_name)

# Chunk the documents
texts = chunk_documents(data)

# Perform the QA retrieval
result = chromadb_retrieval_qa(texts, question)
# %%
# Print the result
print(f"Answer: {result}")
# %%
#4.  playback questions, answers in voice
from gtts import gTTS
import os

def speak_text(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("afplay output.mp3")  # Use 'afplay' to play mp3 on macOS

# Test the function
speak_text(transcription)
speak_text(result)


# %%

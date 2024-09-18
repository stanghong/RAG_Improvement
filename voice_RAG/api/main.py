# %%
from fastapi import FastAPI, HTTPException, Body
from typing import Optional
from pydantic import BaseModel, Field
from pydantic import ValidationError
from typing import Dict, Any
from typing import List, Optional, Tuple
from fastapi.middleware.cors import CORSMiddleware
import os
import openai
from openai import OpenAI
from getpass import getpass
from dotenv import load_dotenv, find_dotenv
import boto3
from io import BytesIO


_ = load_dotenv('.env')
openai.api_key = os.environ['OPENAI_API_KEY']

app = FastAPI()
# %%
# Set up CORS middleware options
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# %%


# %%
class QueryInput(BaseModel): #.wav url
    input_wav_url: str

class QueryResponse(BaseModel):
    output_wav_url: Optional[str]
    return_text: str

# 
@app.post("/voicebot/", response_model=QueryResponse,)
async def voicebot_endpoint(input_data: QueryInput):

    # TODO: first section, retrieve the .wav file from the input_data.input_wav_url
    bucket_name = 'voice-rag'
    object_key = 'output_microphone_audio1.wav'
    local_filename = 'downloaded_audio.wav'

    # Initialize an S3 client
    s3 = boto3.client('s3')

    # the file Download from S3
    try:
        s3.download_file(bucket_name, object_key, local_filename)
        print(f"File downloaded successfully from {bucket_name}/{object_key}")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        exit(1)

    # Convert the voice to text using Whisper
    client = OpenAI()
    audio_file = open(local_filename, "rb") # read from frontend, wav file?
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file, 
        response_format="text"
    )

    # Send the transcript to OpenAI's GPT-4 model to get the answer
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": transcript}
        ]
    )

    response = completion.choices[0].message.content
    print(f'response is {response}')

    # Return the QueryResponse with the response as a string
    return QueryResponse(output_wav_url=None, return_text=response)

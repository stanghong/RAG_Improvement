# %%
from fastapi import FastAPI, HTTPException, Body, Form, UploadFile, File
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
@app.post("/api/voicebot/", response_model=QueryResponse)
async def voicebot_endpoint(
     audio: UploadFile = File(None),
    text: str = Form(None)
):
    
    if audio:
        print(f'Audio file received: {audio.filename}')

        # Read the content of the uploaded file
        content = await audio.read()

        # Prepare the file as a tuple (filename, content)
        audio_data = (audio.filename, content)
        # 使用 Whisper 将语音转换为文本
        client = OpenAI()
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_data,
            response_format="text"
        )
    elif text:
        # 如果没有音频文件,使用提供的文本
        transcript = text
    else:
        raise HTTPException(status_code=400, detail="需要提供音频文件或文本")
    print(f'transcript is {transcript}')
    # 发送转录文本到 GPT-4 模型获取回答
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": transcript}
        ]
    )
    response = completion.choices[0].message.content
    print(f'response is {response}')
    # 返回 QueryResponse
    return QueryResponse(output_wav_url=None, return_text=response)

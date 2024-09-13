from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS

# Set OpenAI API key
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

# Define root route
@app.route('/', methods=['GET'])
def home():
    return "Flask server is running."

# Define /transcribe route
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio_data']
    audio_filename = 'uploaded_audio.wav'
    audio_file.save(audio_filename)

    try:
        client = OpenAI()
        with open(audio_filename, 'rb') as af:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=af,
                response_format="text"
            )

        # Now call ChatGPT with the transcription
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use a valid model name
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": transcription}
            ]
        )

        gpt_answer = response.choices[0].message.content

        # Return both the user's question and GPT's answer
        return jsonify({
            'transcription': f"User question: {transcription}",
            'gpt_answer': f"GPT Ans: {gpt_answer}"
        })

    except Exception as e:
        app.logger.error(f"Error during transcription or GPT request: {e}")
        return jsonify({'error': 'Failed to process request', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)

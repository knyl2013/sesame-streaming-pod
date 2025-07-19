import os
import asyncio
import base64
import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from transformers import pipeline, AutoProcessor, AutoModelForTextToSpectrogram
from scipy.io.wavfile import write
import numpy as np
import io
from groq import Groq

# --- 1. CONFIGURATION AND MODEL LOADING (Happens once on startup) ---

print("Starting server and loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# --- NEW: GET API KEYS FROM ENVIRONMENT ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN") # <-- ADD THIS LINE

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set!")
if not HF_TOKEN: # <-- ADD THIS CHECK
    raise ValueError("HUGGING_FACE_TOKEN environment variable not set!")

llm_client = Groq(api_key=GROQ_API_KEY)

# ASR (Whisper) Pipeline
print("Loading ASR model...")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch_dtype,
    device=device,
)
print("ASR model loaded.")

# TTS (Sesame CSM) Models
print("Loading TTS model...")
# --- MODIFIED: PASS THE TOKEN TO HUGGING FACE ---
tts_processor = AutoProcessor.from_pretrained("sesame-ai/csm-1b", token=HF_TOKEN)
tts_model = AutoModelForTextToSpectrogram.from_pretrained("sesame-ai/csm-1b", token=HF_TOKEN).to(device)
TTS_SAMPLE_RATE = 24000
print("TTS model loaded.")

app = FastAPI()

# --- THE REST OF THE FILE (websocket handler, etc.) STAYS EXACTLY THE SAME ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")
    
    conversation_history = []
    
    try:
        while True:
            user_audio_bytes = await websocket.receive_bytes()
            
            print("Transcribing user audio...")
            with open("temp_user_audio.wav", "wb") as f:
                f.write(user_audio_bytes)
            
            user_text = asr_pipeline("temp_user_audio.wav")["text"]
            print(f"User said: {user_text}")
            
            conversation_history.append({"role": "user", "content": user_text})

            print("Getting LLM response...")
            chat_completion = llm_client.chat.completions.create(
                messages=conversation_history,
                model="llama3-8b-8192",
            )
            ai_response_text = chat_completion.choices[0].message.content
            print(f"AI response: {ai_response_text}")
            
            conversation_history.append({"role": "assistant", "content": ai_response_text})

            print("Generating and streaming AI speech...")
            sentences = ai_response_text.replace('!', '.').replace('?', '.').split('.')
            
            history_waveform = torch.from_numpy(np.frombuffer(user_audio_bytes, dtype=np.int16)).float().to(device) / 32768.0

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                inputs = tts_processor(text=sentence, speech_history=history_waveform, return_tensors="pt").to(device)
                with torch.no_grad():
                    speech_values = tts_model.generate(**inputs, do_sample=True)
                
                output_waveform = speech_values.cpu().numpy().squeeze()
                buffer = io.BytesIO()
                write(buffer, TTS_SAMPLE_RATE, output_waveform.astype(np.float32))
                
                await websocket.send_bytes(buffer.getvalue())

            await websocket.send_text('{"type": "end_of_stream"}')
            print("Finished streaming AI response.")

    except WebSocketDisconnect:
        print("Client disconnected.")
    finally:
        if os.path.exists("temp_user_audio.wav"):
            os.remove("temp_user_audio.wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
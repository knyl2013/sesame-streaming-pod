import os
import asyncio
import base64
import torch
import uvicorn
import subprocess
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from transformers import pipeline, CsmProcessor, CsmForConditionalGeneration
from scipy.io.wavfile import write, read
import numpy as np
import io
from groq import Groq

print("Starting server and loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
if not GROQ_API_KEY: raise ValueError("GROQ_API_KEY not set!")
if not HUGGING_FACE_TOKEN: raise ValueError("HUGGING_FACE_TOKEN not set!")
llm_client = Groq(api_key=GROQ_API_KEY)
print("Loading ASR model...")
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch_dtype, device=device)
print("ASR model loaded.")
print("Loading TTS model...")
TTS_REPO_ID = "sesame/csm-1b"
tts_processor = CsmProcessor.from_pretrained(TTS_REPO_ID, token=HUGGING_FACE_TOKEN) 
tts_model = CsmForConditionalGeneration.from_pretrained(TTS_REPO_ID, token=HUGGING_FACE_TOKEN).to(device)
TTS_SAMPLE_RATE = 24000
print("TTS model loaded successfully.")
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")
    
    conversation_history = []
    input_audio_path = "temp_user_audio_input.webm"
    converted_audio_path = "temp_user_audio_converted.wav"

    try:
        while True:
            # This line waits for the client to send audio data
            user_audio_bytes = await websocket.receive_bytes()
            
            with open(input_audio_path, "wb") as f:
                f.write(user_audio_bytes)

            # 1. Convert audio in a separate thread
            print("Converting received audio to WAV format...")
            ffmpeg_command = ["ffmpeg", "-i", input_audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-y", converted_audio_path]
            await asyncio.to_thread(subprocess.run, ffmpeg_command, check=True, capture_output=True)
            print("Conversion successful.")

            # 2. Transcribe audio in a separate thread
            print("Transcribing user audio...")
            transcription_result = await asyncio.to_thread(asr_pipeline, converted_audio_path)
            user_text = transcription_result["text"]
            print(f"User said: {user_text}")
            
            conversation_history.append({"role": "user", "content": user_text})

            # 3. Get LLM response in a separate thread
            print("Getting LLM response...")
            chat_completion = await asyncio.to_thread(llm_client.chat.completions.create, messages=conversation_history, model="llama3-8b-8192")
            ai_response_text = chat_completion.choices[0].message.content
            print(f"AI response: {ai_response_text}")
            
            conversation_history.append({"role": "assistant", "content": ai_response_text})

            print("Generating and streaming AI speech...")
            
            sr, wav_data = read(converted_audio_path)
            history_waveform = torch.from_numpy(wav_data).float().to(device) / 32768.0
            
            sentences = ai_response_text.replace('!', '.').replace('?', '.').split('.')

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence: continue

                inputs = tts_processor(text=sentence, speech_history=history_waveform, return_tensors="pt").to(device)
                
                # 4. Generate speech in a separate thread
                with torch.no_grad():
                    speech_values = await asyncio.to_thread(tts_model.generate, **inputs, do_sample=True)
                
                output_waveform = speech_values.cpu().numpy().squeeze()
                buffer = io.BytesIO()
                write(buffer, TTS_SAMPLE_RATE, output_waveform.astype(np.float32))
                
                # Sending data back is async, so it doesn't need a thread
                await websocket.send_bytes(buffer.getvalue())

            await websocket.send_text('{"type": "end_of_stream"}')
            print("Finished streaming AI response.")

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if os.path.exists(input_audio_path): os.remove(input_audio_path)
        if os.path.exists(converted_audio_path): os.remove(converted_audio_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
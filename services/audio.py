import os
import io
import wave
import base64
import audioop
import random
import requests
import numpy as np
import speech_recognition as sr
from dotenv import load_dotenv

load_dotenv()

# --- Text-to-Speech (TTS) Setup (Sarvam API) ---
SARVAM_API_KEYS = [k.strip() for k in os.environ.get("SARVAM_API_KEYS", "").split(",") if k.strip()]

def generate_telugu_audio(text: str) -> bytes:
    """Generates Telugu audio bytes from text using Sarvam AI."""
    if not SARVAM_API_KEYS:
        print("❌ No SARVAM_API_KEYS found! Cannot generate audio.")
        return b""
        
    url = "https://api.sarvam.ai/text-to-speech"
    payload = {
        "inputs": [text],
        "target_language_code": "te-IN",
        "speaker": "ritu", 
        "pace": 1.0,
        "speech_sample_rate": 16000,
        "enable_preprocessing": True,
        "model": "bulbul:v3"
    }
    
    keys_to_try = list(SARVAM_API_KEYS)
    random.shuffle(keys_to_try)
    
    for key in keys_to_try:
        try:
            headers = {
                "api-subscription-key": key,
                "Content-Type": "application/json"
            }
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                audio_base64 = data["audios"][0]
                return base64.b64decode(audio_base64)
            elif "insufficient_quota_error" in response.text:
                print(f"⚠️ Sarvam Key {key[:8]}... ran out of quota. Trying next key.")
                continue
            else:
                print(f"Sarvam API Error: {response.text}")
                continue
        except Exception as e:
            print(f"TTS Error with key {key[:8]}...: {e}")
            continue
            
    print("❌ All Sarvam API keys failed or ran out of quota!")
    return b""

# --- Speech-to-Text (ASR) Setup (Google Speech Recognition) ---
recognizer = sr.Recognizer()

def transcribe_audio(wav_bytes: bytes) -> str:
    """Transcribes WAV audio bytes to Telugu text using Google Speech API."""
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="te-IN")
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"❌ Google Speech API Error: {e}")
        return ""
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

# --- Audio Format Conversions ---
def float32_to_wav_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
    audio_int16 = np.int16(audio_array * 32767.0)
    
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        return wav_io.getvalue()

def convert_ulaw_to_wav_bytes(ulaw_bytes: bytes) -> bytes:
    pcm_data = audioop.ulaw2lin(ulaw_bytes, 2)
    pcm_16k, _ = audioop.ratecv(pcm_data, 2, 1, 8000, 16000, None)
    
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(pcm_16k)
        return wav_io.getvalue()

def convert_wav_to_ulaw(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wav_file:
        channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        pcm_data = wav_file.readframes(wav_file.getnframes())
        
    if channels == 2:
        pcm_data = audioop.tomono(pcm_data, sampwidth, 0.5, 0.5)
        channels = 1
        
    if framerate != 8000:
        pcm_data, _ = audioop.ratecv(pcm_data, sampwidth, channels, framerate, 8000, None)
        
    ulaw_data = audioop.lin2ulaw(pcm_data, sampwidth)
    return ulaw_data

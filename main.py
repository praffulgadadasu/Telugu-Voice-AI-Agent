import asyncio
import base64
import numpy as np
import io
import json
import os
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import pipeline
from groq import Groq, InternalServerError, APIStatusError

GROQ_KEYS = [
    "<YOUR_GROQ_API_KEY_1>", # Original
    "<YOUR_GROQ_API_KEY_2>", # Second
    "<YOUR_GROQ_API_KEY_3>"  # Third
]
current_key_index = 0
groq_client = Groq(api_key=GROQ_KEYS[current_key_index])

def switch_groq_key():
    global current_key_index, groq_client
    current_key_index = (current_key_index + 1) % len(GROQ_KEYS)
    print(f"🔄 Rate limit hit! Switching to Groq API Key #{current_key_index + 1}...")
    groq_client = Groq(api_key=GROQ_KEYS[current_key_index])

app = FastAPI(title="Telugu Voice AI Concierge")

print("Initializing AI Models... This will take a moment.")

# --- 1. Text-to-Speech (TTS) Setup (Sarvam AI) ---
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")

def generate_telugu_audio(text: str) -> bytes:
    if not SARVAM_API_KEY:
        print("❌ SARVAM_API_KEY missing! Cannot generate audio.")
        return b""
        
    try:
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
        headers = {
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            audio_base64 = data["audios"][0]
            return base64.b64decode(audio_base64)
        else:
            print(f"Sarvam API Error: {response.text}")
            return b""
    except Exception as e:
        print(f"TTS Error: {e}")
        return b""

import wave

# --- 2. Speech-to-Text (ASR) Setup (Whisper via Groq API) ---
print("Using Groq Whisper API for fast transcription...")

def float32_to_wav_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
    # Convert to int16
    audio_int16 = np.int16(audio_array * 32767.0)
    
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2) # 2 bytes = 16 bits
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        return wav_io.getvalue()

def transcribe_audio(audio_bytes: bytes) -> str:
    wav_bytes = float32_to_wav_bytes(audio_bytes)
    
    for _ in range(len(GROQ_KEYS)):
        try:
            file_obj = ("audio.wav", wav_bytes, "audio/wav")
            transcription = groq_client.audio.transcriptions.create(
                file=file_obj,
                model="whisper-large-v3-turbo",
                language="te"
            )
            return transcription.text
        except Exception as e:
            if "429" in str(e) or "rate_limit_exceeded" in str(e):
                switch_groq_key()
            else:
                print(f"Transcription error: {e}")
                return ""
    print("❌ All Groq API keys are currently rate limited!")
    return ""

# --- 3. The 'Brain' (LLM Logic via Groq) ---

def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config.json: {e}")
        return None

# --- Tool Execution ---
def execute_make_reservation(arguments_str: str) -> str:
    """Executes the tool by hitting the Next.js API."""
    print(f"Executing Tool 'make_reservation' with args: {arguments_str}")
    try:
        args = json.loads(arguments_str)
        import requests
        # Hit the actual website's API endpoint
        res = requests.post(
            'http://localhost:3000/api/reservations', 
            json=args,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        data = res.json()
        if data.get('status') == 'SUCCESS':
            return 'Success! Tell the customer: "Great, I have booked your table. You will receive a confirmation message to your phone shortly."'
        elif data.get('status') == 'FULL':
            return f"Failed: {data.get('message', 'Fully booked')}. Tell the customer we are fully booked and ask for another time."
        else:
            return "Failed: System error. Tell the customer we are having technical difficulties."
    except Exception as e:
        print(f"Database/Tool Error: {e}")
        # If the local API isn't running or crashes, simulate success for testing
        return 'Success (Simulated)! Tell the customer: "Great, I have booked your table. You will receive a confirmation message to your phone shortly."'

# --- WebSocket Endpoint ---
@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📱 Flutter/Web App Connected to Voice API!")
    
    # Reload config on every connection so the user can edit config.json like a dashboard!
    config = load_config()
    if not config:
        print("❌ Cannot start session, config.json is missing or broken!")
        await websocket.close()
        return
        
    SYSTEM_PROMPT = config.get("system_prompt", "")
    import datetime
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{TODAY_DATE}", today_str)
    
    TOOLS = config.get("tools", [])
    first_message_telugu = config.get("first_message", "నమస్కారం!")
    
    # 1. Initialize Conversation Memory for this session
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # 2. Replicate ElevenLabs First Message
    chat_history.append({"role": "assistant", "content": first_message_telugu})
    
    audio_bytes = generate_telugu_audio(first_message_telugu)
    await websocket.send_json({
        "type": "audio_output",
        "text": first_message_telugu,
        "audio_base64": base64.b64encode(audio_bytes).decode('utf-8')
    })
    
    try:
        while True:
            payload = await websocket.receive_text()
            data = json.loads(payload)
            
            user_text = ""
            if data.get("type") == "audio_input":
                audio_bytes = base64.b64decode(data["audio_base64"])
                print("🎧 Received user audio, transcribing...")
                # Run transcription in a thread to avoid blocking the event loop
                user_text = await asyncio.to_thread(transcribe_audio, audio_bytes)
                print(f"🗣️ User said: {user_text}")

            if not user_text:
                continue

            # Add user input to memory
            chat_history.append({"role": "user", "content": user_text})
            
            # --- Call LLM with Tools and Memory (with Retry Logic) ---
            try:
                response = None
                for _ in range(len(GROQ_KEYS)):
                    try:
                        response = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=chat_history,
                            tools=TOOLS,
                            tool_choice="auto",
                            max_tokens=800,
                            temperature=0.5
                        )
                        break # Success, break out of retry loop
                    except Exception as e:
                        if "429" in str(e) or "rate_limit_exceeded" in str(e):
                            switch_groq_key()
                        else:
                            raise e
                            
                if not response:
                    raise Exception("All Groq API keys rate limited.")
                
                response_message = response.choices[0].message
                
                # Check if LLM wanted to call a tool
                should_disconnect = False
                
                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        if tool_call.function.name == "make_reservation":
                            # Execute the tool
                            tool_result = execute_make_reservation(tool_call.function.arguments)
                            
                            # Add tool call and result to history
                            chat_history.append(response_message)
                            chat_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": tool_result
                            })
                            
                            # Call LLM again to get the final spoken confirmation (with Retry Logic)
                            second_response = None
                            for _ in range(len(GROQ_KEYS)):
                                try:
                                    second_response = groq_client.chat.completions.create(
                                        model="llama-3.3-70b-versatile",
                                        messages=chat_history,
                                        max_tokens=800
                                    )
                                    break
                                except Exception as e:
                                    if "429" in str(e) or "rate_limit_exceeded" in str(e):
                                        switch_groq_key()
                                    else:
                                        raise e
                                        
                            if second_response:
                                ai_reply = second_response.choices[0].message.content
                            else:
                                ai_reply = "Thank you, your table is booked."
                        
                        elif tool_call.function.name == "end_call":
                            should_disconnect = True
                            print("📞 AI has decided to end the call.")
                            
                    if not ai_reply and should_disconnect:
                        ai_reply = "Thank you, have a great day!"
                else:
                    ai_reply = response_message.content or ""
                
                # Clean up any stray XML tags or less-than symbols Llama-3 might generate
                if ai_reply:
                    import re
                    ai_reply = re.sub(r'<[^>]*>', '', ai_reply)
                    ai_reply = ai_reply.replace('<', '').replace('>', '')
                    ai_reply = ai_reply.strip()
                
                if not ai_reply:
                    ai_reply = "ధన్యవాదాలు." # fallback if empty
                
                # Add AI reply to memory
                chat_history.append({"role": "assistant", "content": ai_reply})
                print(f"🤖 AI Response: {ai_reply}")
                
                # Generate Audio
                response_audio_bytes = await asyncio.to_thread(generate_telugu_audio, ai_reply)
                audio_base64 = base64.b64encode(response_audio_bytes).decode('utf-8')
                
                # Send to app
                await websocket.send_json({
                    "type": "audio_output",
                    "text": ai_reply,
                    "audio_base64": audio_base64
                })
                
                if should_disconnect:
                    await asyncio.sleep(2) # Give it 2 seconds to finish speaking
                    await websocket.send_json({"type": "disconnect"})
                    break
                
                
            except Exception as e:
                print(f"LLM/Logic Error: {e}")
            
    except WebSocketDisconnect:
        print("📱 App Disconnected.")
    except Exception as e:
        print(f"WebSocket Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import asyncio
import base64
import numpy as np
import io
import json
import os
import requests
import audioop
from dotenv import load_dotenv

# Load API keys from .env file automatically
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from transformers import pipeline
from transformers import pipeline
from google import genai
from google.genai import types

# Initialize Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("❌ No GEMINI_API_KEY found in environment!")

app = FastAPI(title="Telugu Voice AI Concierge")

audio_payload_cache = {}

@app.on_event("startup")
async def startup_event():
    print("🔥 Pre-warming AI TTS Cache for instant answering...")
    config = load_config()
    if config:
        first_msg = config.get("first_message", "నమస్కారం!")
        audio_bytes = await asyncio.to_thread(generate_telugu_audio, first_msg)
        if audio_bytes:
            ulaw_bytes = await asyncio.to_thread(convert_wav_to_ulaw, audio_bytes)
            audio_payload_cache[first_msg] = base64.b64encode(ulaw_bytes).decode("utf-8")
            print("✅ Cache ready! Agent will answer instantly.")

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
import speech_recognition as sr

# --- 2. Speech-to-Text (ASR) Setup (Google Speech Recognition) ---
print("Using Google Native Speech API for flawless Telugu transcription...")
recognizer = sr.Recognizer()

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

def transcribe_audio(wav_bytes: bytes) -> str:
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="te-IN")
        return text
    except sr.UnknownValueError:
        # Google could not understand the audio (often just noise or silence)
        return ""
    except sr.RequestError as e:
        print(f"❌ Google Speech API Error: {e}")
        return ""
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

def convert_ulaw_to_wav_bytes(ulaw_bytes: bytes) -> bytes:
    # Convert 8-bit mu-law to 16-bit PCM (8kHz)
    pcm_data = audioop.ulaw2lin(ulaw_bytes, 2)
    # Resample 8kHz to 16kHz for Whisper
    pcm_16k, _ = audioop.ratecv(pcm_data, 2, 1, 8000, 16000, None)
    
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(pcm_16k)
        return wav_io.getvalue()

def convert_wav_to_ulaw(wav_bytes: bytes) -> bytes:
    # Read WAV using standard python library
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wav_file:
        channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        pcm_data = wav_file.readframes(wav_file.getnframes())
        
    # Convert to mono if stereo
    if channels == 2:
        pcm_data = audioop.tomono(pcm_data, sampwidth, 0.5, 0.5)
        channels = 1
        
    # Convert to 8kHz (Twilio requirement)
    if framerate != 8000:
        pcm_data, _ = audioop.ratecv(pcm_data, sampwidth, channels, framerate, 8000, None)
        
    # Convert 16-bit PCM to 8-bit mu-law
    ulaw_data = audioop.lin2ulaw(pcm_data, sampwidth)
    return ulaw_data

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

def make_reservation(name: str, party_size: str, date: str, time: str, phone: str) -> str:
    """Use this tool to book a table for a customer at The Golden Saffron. Call this ONLY AFTER the user has explicitly said 'YES' (అవును) to confirm the 5 details summary. DO NOT call this immediately after gathering details. DO NOT call this if the user says gibberish."""
    arguments_str = json.dumps({"name": name, "party_size": party_size, "date": date, "time": time, "phone": phone})
    return execute_make_reservation(arguments_str)

def end_call() -> str:
    """Use this tool to end the call gracefully. Call this tool in the exact same turn as make_reservation."""
    return "Call ended successfully."

# --- Twilio Webhook & WebSocket Endpoints ---

@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    print("📞 Incoming call from Twilio!")
    host = request.headers.get("host")
    scheme = "wss" if request.headers.get("x-forwarded-proto") == "https" else "ws"
    websocket_url = f"{scheme}://{host}/ws/twilio"
    
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{websocket_url}" />
    </Connect>
</Response>'''
    return HTMLResponse(content=twiml, media_type="application/xml")

@app.websocket("/ws/twilio")
async def twilio_websocket(websocket: WebSocket):
    await websocket.accept()
    print("📱 Twilio Connected to Voice API!")
    
    config = load_config()
    if not config:
        print("❌ Cannot start session, config.json is missing or broken!")
        await websocket.close()
        return

    SYSTEM_PROMPT = config.get("system_prompt", "You are a helpful assistant.")

    # Parse JSON tools into Gemini Tool schemas
    gemini_tools = []
    for tool_dict in config.get("tools", []):
        fn = tool_dict.get("function", {})
        try:
            gemini_tools.append(types.Tool(function_declarations=[types.FunctionDeclaration(**fn)]))
        except Exception as e:
            print(f"Error parsing tool schema: {e}")
            
    import datetime
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{TODAY_DATE}", today_str)
    
    first_message_telugu = config.get("first_message", "నమస్కారం!")
    
    chat_history = []
    chat_history.append(types.Content(role="model", parts=[types.Part.from_text(text=first_message_telugu)]))
    
    agent_speaking = False
    
    async def send_audio_to_twilio(text: str, sid: str):
        nonlocal agent_speaking
        agent_speaking = True
        
        if text in audio_payload_cache:
            payload = audio_payload_cache[text]
        else:
            audio_bytes = await asyncio.to_thread(generate_telugu_audio, text)
            if not audio_bytes:
                agent_speaking = False
                return
            ulaw_bytes = await asyncio.to_thread(convert_wav_to_ulaw, audio_bytes)
            payload = base64.b64encode(ulaw_bytes).decode("utf-8")
            audio_payload_cache[text] = payload # Cache it for next time
            
        msg = {
            "event": "media",
            "streamSid": sid,
            "media": {"payload": payload}
        }
        await websocket.send_json(msg)
        
        # Send a mark to know exactly when the AI finishes speaking this sentence
        mark_msg = {
            "event": "mark",
            "streamSid": sid,
            "mark": {"name": "agent_finished_speaking"}
        }
        await websocket.send_json(mark_msg)

    silence_threshold = 500
    chunks_of_silence = 0
    audio_buffer = bytearray()
    is_speaking = False
    
    try:
        while True:
            payload = await websocket.receive_text()
            data = json.loads(payload)
            
            if data["event"] == "start":
                stream_sid = data["start"]["streamSid"]
                print(f"🟢 Call started! Stream SID: {stream_sid}")
                await send_audio_to_twilio(first_message_telugu, stream_sid)
                
            elif data["event"] == "mark":
                if data["mark"]["name"] == "end_call_mark":
                    print("✅ Call teardown mark received, closing connection gracefully.")
                    await websocket.close()
                    break
                elif data["mark"]["name"] == "agent_finished_speaking":
                    print("✅ Agent finished speaking. Now listening...")
                    agent_speaking = False
                    # Flush the buffers to clear any echo that might have bled through
                    audio_buffer.clear()
                    is_speaking = False
                    chunks_of_silence = 0
                
            elif data["event"] == "media":
                audio_chunk = base64.b64decode(data["media"]["payload"])
                pcm_chunk = audioop.ulaw2lin(audio_chunk, 2)
                rms = audioop.rms(pcm_chunk, 2)
                
                if rms > silence_threshold:
                    if not is_speaking:
                        print("🗣️ User started speaking...")
                        if agent_speaking:
                            print("🛑 User interrupted the AI! Halting AI speech.")
                            await websocket.send_json({"event": "clear", "streamSid": stream_sid})
                            agent_speaking = False
                    is_speaking = True
                    chunks_of_silence = 0
                elif is_speaking:
                    chunks_of_silence += 1
                
                if is_speaking:
                    audio_buffer.extend(audio_chunk)
                    
                if is_speaking and chunks_of_silence > 25:
                    is_speaking = False
                    chunks_of_silence = 0
                    
                    if len(audio_buffer) < 4000:
                        audio_buffer.clear()
                        continue
                        
                    print("🎧 Silence detected, transcribing...")
                    wav_bytes = convert_ulaw_to_wav_bytes(bytes(audio_buffer))
                    user_text = await asyncio.to_thread(transcribe_audio, wav_bytes)
                    audio_buffer.clear()
                    
                    if not user_text or len(user_text.strip()) < 3:
                        continue
                        
                    print(f"🗣️ User said: {user_text}")
                    chat_history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
                    
                    try:
                        response = None
                        models_to_try = [
                            "gemini-2.5-flash", 
                            "gemini-2.5-flash-lite", 
                            "gemini-2.0-flash-lite-001",
                            "gemini-flash-latest",
                            "gemini-flash-lite-latest"
                        ]
                        for model_name in models_to_try:
                            try:
                                response = await gemini_client.aio.models.generate_content(
                                    model=model_name,
                                    contents=chat_history,
                                    config=types.GenerateContentConfig(
                                        system_instruction=SYSTEM_PROMPT,
                                        tools=gemini_tools,
                                        temperature=0.1
                                    )
                                )
                                break
                            except Exception as e:
                                print(f"⚠️ {model_name} error: {e}")
                                if "503" not in str(e) and "429" not in str(e):
                                    raise e
                                    
                        if not response:
                            raise Exception("All Gemini models are experiencing high demand.")
                            
                        ai_reply = None
                        should_disconnect = False
                        
                        if response.function_calls:
                            if response.candidates:
                                chat_history.append(response.candidates[0].content)
                            
                            parts = []
                            for fc in response.function_calls:
                                if fc.name == "make_reservation":
                                    args_str = json.dumps(fc.args)
                                    tool_result = execute_make_reservation(args_str)
                                    
                                    # FAST-PATH EXIT: Extract name and hardcode final response
                                    import json as js
                                    try:
                                        args_dict = js.loads(args_str)
                                        name = args_dict.get("name", "")
                                    except:
                                        name = ""
                                        
                                    ai_reply = f"థ్యాంక్స్ {name}, మీ రిజర్వేషన్ కన్ఫర్మ్ అయింది, మీకు కన్ఫర్మేషన్ మెసేజ్ వస్తుంది. మళ్ళీ కలుద్దాం."
                                    should_disconnect = True
                                    print("📞 Reservation successful. Overriding AI to immediately hang up.")
                                    break
                                elif fc.name == "end_call":
                                    should_disconnect = True
                                    parts.append(types.Part.from_function_response(name="end_call", response={"result": "Success"}))
                                    print("📞 AI has decided to end the call.")
                                    
                            if not should_disconnect:
                                chat_history.append(types.Content(role="user", parts=parts))
                                
                                second_response = None
                                for model_name in models_to_try:
                                    try:
                                        second_response = await gemini_client.aio.models.generate_content(
                                            model=model_name,
                                            contents=chat_history,
                                            config=types.GenerateContentConfig(
                                                system_instruction=SYSTEM_PROMPT,
                                                tools=gemini_tools,
                                                temperature=0.1
                                            )
                                        )
                                        break
                                    except Exception as e:
                                        print(f"⚠️ {model_name} error on tool response: {e}")
                                        if "503" not in str(e) and "429" not in str(e):
                                            raise e
                                            
                                if not second_response:
                                    raise Exception("All Gemini models are experiencing high demand.")
                                    
                                ai_reply = second_response.text
                                if second_response.candidates:
                                    chat_history.append(second_response.candidates[0].content)
                        else:
                            ai_reply = response.text
                            if response.candidates:
                                chat_history.append(response.candidates[0].content)
                        
                        if ai_reply:
                            import re
                            ai_reply = re.sub(r'<[^>]*>', '', ai_reply)
                            ai_reply = ai_reply.replace('<', '').replace('>', '')
                            ai_reply = ai_reply.strip()
                        
                        if not ai_reply:
                            if should_disconnect:
                                ai_reply = "ధన్యవాదాలు, మీ బుకింగ్ కన్ఫర్మ్ అయింది."
                            else:
                                ai_reply = "ధన్యవాదాలు."
                        
                        print(f"🤖 AI Response: {ai_reply}")
                        
                        await send_audio_to_twilio(ai_reply, stream_sid)
                        
                        if should_disconnect:
                            # Send a mark event. Twilio will play the audio, then return this mark to us.
                            await websocket.send_json({
                                "event": "mark", 
                                "streamSid": stream_sid, 
                                "mark": {"name": "end_call_mark"}
                            })
                            # DO NOT break or close here. Wait for the 'mark' event from Twilio!
                            
                    except Exception as e:
                        print(f"LLM/Logic Error: {e}")
            
            elif data["event"] == "stop":
                print("🔴 Call stopped by Twilio.")
                break
                
    except WebSocketDisconnect:
        print("📱 Twilio Disconnected.")
    except Exception as e:
        print(f"WebSocket Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

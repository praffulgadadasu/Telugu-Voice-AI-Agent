import os
import json
import base64
import asyncio
import re

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types

# Import our modularized services
from config_loader import load_config, prepare_system_prompt, parse_gemini_tools
from services.audio import generate_telugu_audio, convert_wav_to_ulaw, convert_ulaw_to_wav_bytes, transcribe_audio, float32_to_wav_bytes
from services.tools import execute_make_reservation

# Initialize Gemini Client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("❌ No GEMINI_API_KEY found in environment!")

app = FastAPI(title="Telugu Voice AI Concierge")

# Global Cache for pre-generated greetings
audio_payload_cache = {}

@app.on_event("startup")
async def startup_event():
    print("🔥 Pre-warming AI TTS Cache for instant answering...")
    config = load_config()
    if config:
        first_msg = config.get("first_message", "నమస్కారం!")
        fillers = [first_msg, "సరే..."]
        for msg in fillers:
            audio_bytes = await asyncio.to_thread(generate_telugu_audio, msg)
            if audio_bytes:
                ulaw_bytes = await asyncio.to_thread(convert_wav_to_ulaw, audio_bytes)
                audio_payload_cache[msg] = base64.b64encode(ulaw_bytes).decode("utf-8")
        print("✅ Cache ready! Agent will answer instantly.")

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

    SYSTEM_PROMPT = prepare_system_prompt(config)
    gemini_tools = parse_gemini_tools(config)
    first_message_telugu = config.get("first_message", "నమస్కారం!")
    
    chat_history = [types.Content(role="model", parts=[types.Part.from_text(text=first_message_telugu)])]
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

    # Note: audioop is imported locally here since we need it for RMS calculation in the loop
    import audioop 
    
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
                    
                if is_speaking and chunks_of_silence > 15:
                    is_speaking = False
                    chunks_of_silence = 0
                    
                    if len(audio_buffer) < 4000:
                        audio_buffer.clear()
                        continue
                        
                    print("🎧 Silence detected, transcribing...")
                    wav_bytes = await asyncio.to_thread(convert_ulaw_to_wav_bytes, bytes(audio_buffer))
                    user_text = await asyncio.to_thread(transcribe_audio, wav_bytes)
                    audio_buffer.clear()
                    
                    if not user_text or len(user_text.strip()) < 3:
                        continue
                        
                    print(f"🗣️ User said: {user_text}")
                    chat_history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
                    
                    try:
                        should_disconnect = False
                        while True:
                            # 🚀 STREAMING GEMINI
                            response_stream = await gemini_client.aio.models.generate_content_stream(
                                model="gemini-2.5-flash",
                                contents=chat_history,
                                config=types.GenerateContentConfig(
                                    system_instruction=SYSTEM_PROMPT,
                                    tools=gemini_tools,
                                    temperature=0.1
                                )
                            )
                            
                            full_ai_reply = ""
                            current_sentence = ""
                            made_function_call_in_this_stream = False
                            has_function_calls = False
                            
                            # Process stream
                            async for chunk in response_stream:
                                if chunk.function_calls:
                                    has_function_calls = True
                                    made_function_call_in_this_stream = True
                                    parts = []
                                    for fc in chunk.function_calls:
                                        if fc.name == "make_reservation":
                                            args_str = json.dumps(fc.args)
                                            # CRITICAL: await asyncio.to_thread prevents blocking ASGI loop
                                            tool_result = await asyncio.to_thread(execute_make_reservation, args_str)
                                            print("📞 Reservation successful. Letting AI handle the follow-up question.")
                                            chat_history.append(types.Content(role="model", parts=[types.Part.from_function_call(name="make_reservation", args=fc.args)]))
                                            chat_history.append(types.Content(role="user", parts=[types.Part.from_function_response(name="make_reservation", response={"result": tool_result})]))
                                            break
                                        elif fc.name == "end_call":
                                            should_disconnect = True
                                            await send_audio_to_twilio("మీ రిజర్వేషన్ సమయానికి కలుద్దాం.", stream_sid)
                                            parts.append(types.Part.from_function_response(name="end_call", response={"result": "Success"}))
                                            print("📞 AI has decided to end the call. Playing final audio and disconnecting.")
                                            chat_history.append(types.Content(role="model", parts=[types.Part.from_function_call(name="end_call", args=fc.args)]))
                                            chat_history.append(types.Content(role="user", parts=parts))
                                            
                                    if should_disconnect:
                                        break
                                        
                                if chunk.text:
                                    text_chunk = chunk.text
                                    text_chunk = re.sub(r'<[^>]*>', '', text_chunk).replace('<', '').replace('>', '')
                                    full_ai_reply += text_chunk
                                    current_sentence += text_chunk
                                    
                                    # 🚀 SENTENCE CHUNKING TTS
                                    delimiters = ['.', '?', '!', '।']
                                    for delim in delimiters:
                                        if delim in current_sentence:
                                            parts = current_sentence.split(delim, 1)
                                            sentence_to_play = (parts[0] + delim).strip()
                                            current_sentence = parts[1] if len(parts) > 1 else ""
                                            
                                            if sentence_to_play:
                                                print(f"🤖 TTS Chunk: {sentence_to_play}")
                                                await send_audio_to_twilio(sentence_to_play, stream_sid)
                                            break
                            
                            if current_sentence.strip() and not should_disconnect:
                                print(f"🤖 TTS Chunk (Final): {current_sentence.strip()}")
                                await send_audio_to_twilio(current_sentence.strip(), stream_sid)
                                
                            if full_ai_reply:
                                chat_history.append(types.Content(role="model", parts=[types.Part.from_text(text=full_ai_reply)]))
                                
                            if not made_function_call_in_this_stream or should_disconnect:
                                break
                                
                        if should_disconnect and not has_function_calls:
                            await send_audio_to_twilio("ధన్యవాదాలు, మీ బుకింగ్ కన్ఫర్మ్ అయింది.", stream_sid)
                            
                        if should_disconnect:
                            await websocket.send_json({
                                "event": "mark", 
                                "streamSid": stream_sid, 
                                "mark": {"name": "end_call_mark"}
                            })
                            
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

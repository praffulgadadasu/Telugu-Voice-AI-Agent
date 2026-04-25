import os
from main import generate_telugu_audio

def test():
    # A simple Telugu phrase: "Hello, how are you?"
    test_text = "నమస్కారం, మీరు ఎలా ఉన్నారు?"
    
    print(f"Generating audio for: {test_text}")
    print("Please wait, the model is processing...")
    
    try:
        audio_bytes = generate_telugu_audio(test_text)
        
        output_file = "test_output.wav"
        with open(output_file, "wb") as f:
            f.write(audio_bytes)
            
        print(f"Success! Audio saved to {os.path.abspath(output_file)}")
        print("Please play this file to check the Telugu pronunciation.")
    except Exception as e:
        print(f"Error generating audio: {e}")

if __name__ == "__main__":
    test()

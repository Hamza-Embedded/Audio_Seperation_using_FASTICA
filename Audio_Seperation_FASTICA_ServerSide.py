import socket
import numpy as np
import soundfile as sf
import os
import io
import time
from faster_whisper import WhisperModel
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
from sklearn.decomposition import FastICA
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

# ==============================
# GEMINI CONFIG
# ==============================


GEMINI_API_KEY = "Add your Gemini API Key"

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction="You are a helpful assistant that provides concise, complete answers. Keep your responses around 150 words or less."
)


def ask_llm(text):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                text,
                generation_config={
                    "max_output_tokens": 250,  # Reduced for 30-word limit
                    "temperature": 0.8,
                }
            )
            return response.text.strip()

        except ResourceExhausted:
            wait_time = 2 ** attempt
            print(f"⚠️ Rate limit hit. Retrying in {wait_time}s...")
            time.sleep(wait_time)

        except GoogleAPIError as e:
            print(f"⚠️ Gemini API error: {e}")
            return "[LLM call failed]"

        except Exception as e:
            print(f"⚠️ Unknown error: {e}")
            return "[LLM call failed]"

    return "[LLM call failed after retries]"


# ==============================
# TEXT-TO-SPEECH FUNCTION
# ==============================
def text_to_speech(text, output_dir="outputs", play_audio=True):
    """
    Convert text to speech and optionally play it.
    Returns the path to the saved audio file.
    """
    try:
        # Generate speech
        print("🔊 Generating speech audio...")
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to file
        audio_file = os.path.join(output_dir, f"response_{int(time.time())}.mp3")
        tts.save(audio_file)
        print(f"✅ Audio saved: {audio_file}")
        
        # Play audio if requested
        if play_audio:
            print("🔊 Playing audio response...")
            audio = AudioSegment.from_mp3(audio_file)
            play(audio)
        
        return audio_file
    
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return None


# ==============================
# HELPER: SELECT BEST TRANSCRIPTION
# ==============================
def select_best_transcription(transcriptions):
    """
    Select the best transcription based on:
    1. Length (longer is often better)
    2. Word count
    3. Confidence (presence of actual words vs empty/noise)
    """
    best_text = ""
    best_score = 0
    best_speaker = -1
    
    for i, text in enumerate(transcriptions):
        if not text or len(text.strip()) == 0:
            continue
        
        # Calculate quality score
        word_count = len(text.split())
        char_count = len(text.strip())
        
        # Score based on length and word count
        score = word_count * 2 + char_count
        
        # Bonus for having punctuation (indicates better structure)
        if any(p in text for p in ['.', '?', '!']):
            score += 10
        
        if score > best_score:
            best_score = score
            best_text = text
            best_speaker = i + 1
    
    return best_text, best_speaker


# ==============================
# HELPER: CALCULATE AUDIO QUALITY METRICS
# ==============================
def calculate_audio_quality(audio_signal):
    """
    Calculate signal quality metrics to help select best channel.
    Returns RMS energy and signal-to-noise estimate.
    """
    # RMS energy
    rms = np.sqrt(np.mean(audio_signal ** 2))
    
    # Simple SNR estimate (higher is better)
    signal_power = np.mean(audio_signal ** 2)
    noise_estimate = np.mean(np.abs(np.diff(audio_signal)) ** 2)
    
    if noise_estimate > 0:
        snr = 10 * np.log10(signal_power / noise_estimate)
    else:
        snr = 0
    
    return rms, snr


# ==============================
# SERVER CONFIG
# ==============================
SERVER_IP = "0.0.0.0"
SERVER_PORT = 5000
BUFFER_SIZE = 4096

# ==============================
# AUDIO CONFIG
# ==============================
RATE = 16000
CHANNELS = 4
CHUNK_DURATION = 12  # seconds per chunk for real-time processing
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

chunk_samples = RATE * CHUNK_DURATION
chunk_bytes = chunk_samples * CHANNELS * 2  # 2 bytes per int16 sample

# ==============================
# LOAD WHISPER
# ==============================
print("🧠 Loading Whisper model...")
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
print("✅ Whisper loaded")

# ==============================
# HELPER FUNCTION: TRANSCRIBE FROM NUMPY ARRAY
# ==============================
def transcribe_audio_array(audio_channel, sample_rate=16000):
    """
    Transcribe audio directly from numpy array without saving to disk.
    """
    # Normalize audio to float32 in range [-1, 1]
    if audio_channel.dtype == np.int16:
        audio_float = audio_channel.astype(np.float32) / 32768.0
    else:
        audio_float = audio_channel.astype(np.float32)
    
    # Create temporary file (faster-whisper requires file path)
    temp_file = os.path.join(OUTPUT_DIR, "temp_transcribe.wav")
    sf.write(temp_file, audio_float, sample_rate)
    
    # Transcribe
    segments, info = whisper_model.transcribe(temp_file, language="en")
    text = " ".join([segment.text for segment in segments])
    
    # Clean up temp file
    try:
        os.remove(temp_file)
    except:
        pass
    
    return text.strip()


# ==============================
# START SERVER
# ==============================
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((SERVER_IP, SERVER_PORT))
sock.listen(1)

print(f"🖥️ Server listening on {SERVER_IP}:{SERVER_PORT}")
conn, addr = sock.accept()
print(f"✅ Connection established from {addr}")
print("🎙️ Receiving and processing audio in real-time...\n")

# ==============================
# RECEIVE AND PROCESS IN CHUNKS
# ==============================
chunk_count = 0
data_buffer = b""
all_audio_chunks = []  # Store all chunks for final processing if needed

try:
    while True:
        # Receive data
        packet = conn.recv(BUFFER_SIZE)
        if not packet:
            break
        
        data_buffer += packet
        
        # Process when we have enough data for one chunk
        while len(data_buffer) >= chunk_bytes:
            # Extract one chunk
            chunk_data = data_buffer[:chunk_bytes]
            data_buffer = data_buffer[chunk_bytes:]
            
            # Convert to numpy array
            audio_chunk = np.frombuffer(chunk_data, dtype=np.int16)
            audio_chunk = audio_chunk.reshape(-1, CHANNELS).T  # (channels, samples)
            
            all_audio_chunks.append(audio_chunk)
            
            chunk_count += 1
            print(f"\n{'='*70}")
            print(f"📦 Processing Chunk #{chunk_count}")
            print(f"{'='*70}")
            
            # ==============================
            # FASTICA SOURCE SEPARATION
            # ==============================
            print("🧪 Running FastICA source separation...")
            
            # Convert to float32 and normalize
            audio_f = audio_chunk.astype(np.float32) / 32768.0
            
            # FastICA expects (samples, channels)
            X = audio_f.T  # (samples, channels)
            
            ica = FastICA(
                n_components=CHANNELS,
                random_state=0,
                max_iter=1000,
                tol=1e-4
            )
            
            sources = ica.fit_transform(X)  # (samples, components)
            sources = sources.T  # (components, samples)
            
            print(f"✅ FastICA output shape: {sources.shape}")
            
            # ==============================
            # TRANSCRIBE ALL SEPARATED SOURCES
            # ==============================
            transcriptions = []
            audio_qualities = []
            
            print("\n📝 Transcribing all speakers...")
            for i in range(CHANNELS):
                sep_audio = sources[i]
                
                # Normalize to avoid clipping
                sep_audio = sep_audio / (np.max(np.abs(sep_audio)) + 1e-9)
                
                # Calculate audio quality
                rms, snr = calculate_audio_quality(sep_audio)
                audio_qualities.append((rms, snr))
                
                print(f"   🎤 Speaker {i+1} (RMS: {rms:.4f}, SNR: {snr:.2f} dB)...", end=" ")
                text = transcribe_audio_array(sep_audio, RATE)
                transcriptions.append(text)
                
                if text:
                    print(f"✓ ({len(text.split())} words)")
                else:
                    print("(no speech)")
            
            # ==============================
            # DISPLAY ALL TRANSCRIPTIONS
            # ==============================
            print(f"\n{'='*70}")
            print("📄 ALL TRANSCRIPTIONS:")
            print(f"{'='*70}")
            
            for i, text in enumerate(transcriptions):
                print(f"\n🗣️ Speaker {i+1}:")
                print(f"{'-'*70}")
                if text:
                    print(text)
                else:
                    print("   (no speech detected)")
            
            print(f"\n{'='*70}")
            
            # ==============================
            # SELECT BEST TRANSCRIPTION FOR LLM
            # ==============================
            best_text, best_speaker = select_best_transcription(transcriptions)
            
            if best_text:
                print(f"\n{'='*70}")
                print(f"🏆 BEST TRANSCRIPTION FOR LLM: Speaker {best_speaker}")
                print(f"{'='*70}")
                print(f"Selected text: {best_text}")
                print(f"\n{'='*70}")
                
                # Get complete LLM response
                print(f"🤖 Sending to Gemini for detailed response...\n")
                llm_response = ask_llm(best_text)
                
                print(f"💬 Gemini Response:")
                print(f"{'='*70}")
                print(llm_response)
                print(f"{'='*70}")
                
                # Convert response to speech and play
                audio_file = text_to_speech(llm_response, OUTPUT_DIR, play_audio=True)
                
            else:
                print("\n⚠️ No clear speech detected in any channel")

except KeyboardInterrupt:
    print("\n\n⚠️ Interrupted by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Process any remaining data in buffer
    if len(data_buffer) > 0:
        print(f"\n📦 Processing final partial chunk...")
        audio_chunk = np.frombuffer(data_buffer, dtype=np.int16)
        
        # Pad if necessary to avoid reshape issues
        samples_needed = (len(audio_chunk) // CHANNELS) * CHANNELS
        audio_chunk = audio_chunk[:samples_needed]
        
        if len(audio_chunk) > 0:
            audio_chunk = audio_chunk.reshape(-1, CHANNELS).T
            all_audio_chunks.append(audio_chunk)
            
            # FastICA on final chunk
            print("🧪 Running FastICA source separation...")
            audio_f = audio_chunk.astype(np.float32) / 32768.0
            X = audio_f.T
            
            ica = FastICA(
                n_components=CHANNELS,
                random_state=0,
                max_iter=1000,
                tol=1e-4
            )
            
            sources = ica.fit_transform(X)
            sources = sources.T
            
            # Transcribe all and select best
            transcriptions = []
            print("\n📝 Transcribing all speakers...")
            
            for i in range(CHANNELS):
                sep_audio = sources[i]
                sep_audio = sep_audio / (np.max(np.abs(sep_audio)) + 1e-9)
                
                print(f"   🎤 Speaker {i+1}...", end=" ")
                text = transcribe_audio_array(sep_audio, RATE)
                transcriptions.append(text)
                
                if text:
                    print(f"✓ ({len(text.split())} words)")
                else:
                    print("(no speech)")
            
            # Display all transcriptions
            print(f"\n{'='*70}")
            print("📄 ALL TRANSCRIPTIONS:")
            print(f"{'='*70}")
            
            for i, text in enumerate(transcriptions):
                print(f"\n🗣️ Speaker {i+1}:")
                print(f"{'-'*70}")
                if text:
                    print(text)
                else:
                    print("   (no speech detected)")
            
            print(f"\n{'='*70}")
            
            # Select best for LLM
            best_text, best_speaker = select_best_transcription(transcriptions)
            
            if best_text:
                print(f"\n{'='*70}")
                print(f"🏆 BEST TRANSCRIPTION FOR LLM: Speaker {best_speaker}")
                print(f"{'='*70}")
                print(f"Selected text: {best_text}")
                print(f"\n{'='*70}")
                
                print(f"🤖 Sending to Gemini for detailed response...\n")
                llm_response = ask_llm(best_text)
                
                print(f"💬 Gemini Response:")
                print(f"{'='*70}")
                print(llm_response)
                print(f"{'='*70}")
                
                # Convert response to speech and play
                audio_file = text_to_speech(llm_response, OUTPUT_DIR, play_audio=True)
    
    # Optional: Save complete audio
    if all_audio_chunks:
        print("\n💾 Saving complete audio...")
        complete_audio = np.concatenate(all_audio_chunks, axis=1)
        raw_file = os.path.join(OUTPUT_DIR, "raw_4ch_complete.wav")
        sf.write(raw_file, complete_audio.T, RATE)
        print(f"✅ Saved complete audio: {raw_file}")
    
    conn.close()
    sock.close()
    print("\n✅ Connection closed")
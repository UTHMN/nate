import os
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torchaudio
import whisper
from pyannote.core import Segment
from scipy.spatial.distance import cosine
import warnings

import hashlib
from gtts import gTTS
from os.path import exists

from dotenv import load_dotenv
from os.path import dirname, join
from os import getenv

load_dotenv(join(dirname(__file__), ".env"))

# TODO:
# 1. Add more accepted audio types (wav, mp3, aac, flacc)

# Enrollment help:
# Speaking types
#   - Tones: Neutral, energetic, calm
#   - Volumes: Normal, louder, softer
#
# TONE,       VOLUME       SENTENCE
# Neutral	  Normal       The quick brown fox jumps over the lazy dog beside a calm riverbank at sunset
# Neutral     Louder       19 Bold penguins briskly marched into the wind, flapping their wings and honking in protest
# Neutral     Softer       Silent raindrops tapped on the old tin roof as the fire crackled gently in the corner
# Energetic   Normal       Every flaming rocket zigzagged across the night sky as the crowd roared with excitement
# Energetic   Louder       Yes! The lions charged the dusty field as thunder echoed and spectators screamed with joy!
# Calm        Normal       The ocean whispered beneath the moon as soft breezes stirred the seagrass below the cliffs
# Calm        Softer       With each breath, the mountain rested in stillness while snowflakes melted on warm fur

warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")

# Variables
SPEAKER_DB_PATH = "speaker_db.npy"

# Global models
encoder = VoiceEncoder()
whisper_model = whisper.load_model("small")  # Can be "small", "medium", etc.
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=getenv("HUGGING_FACE_TOKEN")  # ← insert token here
)

# Load or initialize speaker DB
if os.path.exists(SPEAKER_DB_PATH):
    speaker_db = np.load(SPEAKER_DB_PATH, allow_pickle=True).item()
else:
    speaker_db = {}

def mp3_to_wav(mp3_path: str, wav_path: str):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")

def enroll_speaker(token: str, mp3_path: str):
    token = token.lower()
    wav_path = mp3_path.replace(".mp3", ".wav")
    mp3_to_wav(mp3_path, wav_path)
    wav = preprocess_wav(wav_path)
    embedding = encoder.embed_utterance(wav)
    if token in speaker_db:
        speaker_db[token].append(embedding)
        print(f"[+] Added another embedding for speaker '{token}'.")
    else:
        speaker_db[token] = [embedding]
        print(f"[✔] Enrolled new speaker '{token}'.")
    
    np.save(SPEAKER_DB_PATH, speaker_db)

def classify_and_transcribe(mp3_path: str):
    wav_path = mp3_path.replace(".mp3", ".wav")
    mp3_to_wav(mp3_path, wav_path)

    print("[🔎] Running diarization...")
    diarization = diarization_pipeline(wav_path)

    print("[🧠] Transcribing full audio...")
    result = whisper_model.transcribe(wav_path, language="en", verbose=False)
    full_transcript = result["text"]

    def get_segment_audio(file_path, segment: Segment):
        waveform, sample_rate = torchaudio.load(file_path)
        start = int(segment.start * sample_rate)
        end = int(segment.end * sample_rate)
        return waveform[:, start:end], sample_rate

    speaker_segments = []

    for turn, _, _ in diarization.itertracks(yield_label=True):
        audio, sr = get_segment_audio(wav_path, turn)
        emb = encoder.embed_utterance(audio.numpy()[0])
        scores = {
            token: max(1 - cosine(emb, ref_emb) for ref_emb in emb_list)
            for token, emb_list in speaker_db.items()
        }

        best_speaker = max(scores, key=scores.get)
        confidence = scores[best_speaker]

        print(f"[🗣️] {best_speaker} [{turn.start:.1f}s → {turn.end:.1f}s] (conf: {confidence:.2f})")
        speaker_segments.append({
            "speaker": best_speaker,
            "start": float(turn.start),
            "end": float(turn.end),
            "confidence": float(confidence)
        })

    return {
        "transcript": full_transcript,
        "speaker_segments": speaker_segments
    }

def transcribe(mp3_path: str):
    wav_path = mp3_path.replace(".mp3", ".wav")
    mp3_to_wav(mp3_path, wav_path)
    
    result = whisper_model.transcribe(wav_path, language="en", verbose=False)
    full_transcript = result["text"]
    
    return full_transcript

def tts(text: str):
    tts = gTTS(text=text, lang="en")
    h = hashlib.sha256(text.encode()).hexdigest()
    print(f"hashed tts file: /tmp/{h}.mp3")
    filepath = f"/tmp/{h}.mp3"
    
    if exists(filepath):
        print(f"tts file already exists: {filepath}")
        return filepath
    
    tts.save(filepath)
    
    return filepath

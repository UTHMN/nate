import os
import json
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torchaudio
import whisper
from pyannote.core import Segment
from scipy.spatial.distance import cosine
import warnings
import pprint

from dotenv import load_dotenv
from os.path import dirname, join
from os import getenv

load_dotenv(join(dirname(__file__), ".env"))

# TODO:
# 1. Add more accepted audio types (wav, mp3, aac, flacc)
# 2. Improve enrollment process for higher access (multiple embeddings per speaker)


# ENROLLMENT EXAMPLE PHRASES
# The quick brown fox jumps over the lazy dog x3 (different tones and volume)
# Pack my box with five dozen liquor jugs x3 (different tones and volume)
# The five boxing wizards jump quickly x3 (different tones and volume)

# TESTING PHRASES
# Multiple random sentences with different tones and volume e.g.
# Twelve quirky zebras jogged briskly through the fog at 6:42 a.m., humming Beethoven's Fifth.

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

def enroll_speaker(name: str, mp3_path: str):
    wav_path = mp3_path.replace(".mp3", ".wav")
    mp3_to_wav(mp3_path, wav_path)
    wav = preprocess_wav(wav_path)
    embedding = encoder.embed_utterance(wav)
    speaker_db[name] = embedding
    np.save(SPEAKER_DB_PATH, speaker_db)
    print(f"[✔] Enrolled speaker '{name}'.")

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
            name: 1 - cosine(emb, ref_emb)
            for name, ref_emb in speaker_db.items()
        }
        best_speaker = max(scores, key=scores.get)
        confidence = scores[best_speaker]

        print(f"[🗣️] {best_speaker} [{turn.start:.1f}s → {turn.end:.1f}s] (conf: {confidence:.2f})")
        speaker_segments.append({
            "speaker": best_speaker,
            "start": turn.start,
            "end": turn.end,
            "confidence": confidence
        })

    return {
        "transcript": full_transcript,
        "speaker_segments": speaker_segments
    }

import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pyttsx3
from sklearn.metrics.pairwise import cosine_similarity
import ollama
import numpy as np

# -----------------------------
# Text-to-Speech
# -----------------------------
engine = pyttsx3.init()

def speak(text):
    print("Agent:", text)
    engine.say(text)
    engine.runAndWait()

# -----------------------------
# Load Whisper model (LOAD ONCE)
# -----------------------------
print("Loading Whisper model...")
model = whisper.load_model("base")

# -----------------------------
# Record + Speech-to-Text
# -----------------------------
def listen():
    fs = 16000
    seconds = 4

    print("🎤 Speak now...")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()

    write("input.wav", fs, recording)

    result = model.transcribe("input.wav")
    text = result["text"].strip()

    print("You:", text)
    return text

# -----------------------------
# Embedding
# -----------------------------
def get_embedding(text):
    return ollama.embed(
        model='nomic-embed-text',
        input=text
    )['embeddings'][0]

# -----------------------------
# Knowledge Base
# -----------------------------
docs = [
    "I like playing cricket",
    "Artificial intelligence includes machine learning",
    "Python is a programming language"
]

print("Generating embeddings...")
doc_embeddings = [get_embedding(d) for d in docs]

# Warm-up (important for speed)
_ = get_embedding("hello")

# -----------------------------
# Voice Agent Loop
# -----------------------------
while True:
    user_text = listen()

    if not user_text:
        speak("I didn't catch that")
        continue

    if "exit" in user_text.lower():
        speak("Goodbye!")
        break

    user_emb = get_embedding(user_text)

    similarities = cosine_similarity([user_emb], doc_embeddings)[0]

    best_match_index = similarities.argmax()
    best_match = docs[best_match_index]

    response = f"Closest match is: {best_match}"
    speak(response)
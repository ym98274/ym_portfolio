import speech_recognition as sr
import numpy as np

SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # Duration of audio chunks in seconds

def detect_audio_presence(audio_chunk):
    """
    Detect if any meaningful audio is present in the chunk.
    Returns True if audio is detected, False otherwise.
    """
    amplitude = np.abs(audio_chunk)
    return np.any(amplitude > 0.02)  # Threshold to detect meaningful audio

def record_and_transcribe():
    """
    Record audio in chunks and yield transcriptions using Google Speech-to-Text.
    If no audio is detected, yield "No audio detected".
    """
    recognizer = sr.Recognizer()
    mic = sr.Microphone(sample_rate=SAMPLE_RATE)

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        print("Listening for audio in chunks...")

        while True:
            try:
                print("Recording audio...")
                audio = recognizer.listen(source, timeout=CHUNK_DURATION, phrase_time_limit=CHUNK_DURATION)
                audio_chunk = np.frombuffer(audio.get_raw_data(), dtype=np.int16)

                # Check for audio presence
                if not detect_audio_presence(audio_chunk):
                    yield "No audio detected"
                    continue

                # Transcribe the audio
                transcription = recognizer.recognize_google(audio)
                if transcription.strip():  # Ensure transcription is not empty
                    yield transcription
                else:
                    yield "No audio detected"

            except sr.UnknownValueError:
                yield "No audio detected"
            except sr.RequestError as e:
                yield f"Google Speech Recognition error: {e}"
            except sr.WaitTimeoutError:
                yield "No audio detected"

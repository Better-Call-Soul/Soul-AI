import speech_recognition as sr

r = sr.Recognizer()

audio_file = sr.AudioFile('./parler_tts_out.wav')

with audio_file as source:

    audio = r.record(source)

    text = r.recognize_google(audio)

    print(text)
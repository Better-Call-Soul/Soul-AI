from gtts import gTTS  
import os
import pygame

def text_to_speech(text):
    # set the language (you can change this to the language you prefer)
    language = 'en'

    # generate the TTS audio
    tts = gTTS(text=text, lang=language, slow=False, tld='com')
    
    # save the audio file
    audio_file = f"output.mp3"
    tts.save(audio_file)

    # initialize the pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)

    # play the audio
    pygame.mixer.music.play()

    # wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # remove the audio file
    os.remove(audio_file)

# example usage:
text = "Hello, how are you feeling today? i wish you a good day and a good night and a good morning. I love you. may be you are sad."
text_to_speech(text)

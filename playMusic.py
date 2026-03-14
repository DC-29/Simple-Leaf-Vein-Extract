import pygame
import time

# Play generated MIDI file
pygame.init()
pygame.mixer.init()

try:
    pygame.mixer.music.load('leaf_music.mid')
    pygame.mixer.music.play()
    print("Playing... press Ctrl+C to stop")
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
except Exception as e:
    print(f"Error: {e}")
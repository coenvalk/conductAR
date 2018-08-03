import pygame
import mido
import time


with mido.open_output('TiMidity port 1') as port:
    F = mido.MidiFile('music_test.mid')

    print F.ticks_per_beat

    total = 0
    
    for msg in F.play():
        total += msg.time
        port.send(msg)

    print total, F.length
        
    port.close()

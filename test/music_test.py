import mido
import time

port = mido.open_output()

mid = mido.MidiFile('music_test.mid')
for msg in mid.play():
    port.send(msg)

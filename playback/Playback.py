import time
import mido
import pygame
from threading import Thread

class Playback:        
    def __init__(self, File=None, port='TiMidity port 1'):
        self.stopped = True

        self.ref_file = File
        self.mid_file = mido.MidiFile(File)
        self.port_name = port
        self.port = mido.open_output(self.port_name)


    def init(self, File, port='TiMidity port 1'):
        self.ref_file = File
        self.mid_file = mido.MidiFile(File)
        self.port_name = port
        self.port = mido.open_output(self.port_name)

    def change_file(self, File):
        self.ref_file = File

    def change_port(self, port):
        self.port_name = port
        self.port = mido.open_output(self.port_name)

    def start(self):
        self.stopped = False
        Thread(target=self._update, args=()).start()

    def stop(self):
        self.stopped = True

    def close(self):
        self.stopped = True
        self.port.close()
    
    def _update(self):
        for msg in self.mid_file.play():
            self.port.send(msg)
            if self.stopped:
                return
        self.close()

    def set_ticks_per_beat(self, ticks):
        self.mid_file.ticks_per_beat = ticks

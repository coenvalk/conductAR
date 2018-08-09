from playback import Playback
import mido
import time

def test_initialization():
    print mido.get_output_names()
    
    P = Playback('test/music_test.mid')
    print P.port

    P.start()
    time.sleep(10)
    P.stop()
    P.close()

test_initialization()

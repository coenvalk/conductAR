from playback import Playback
import time

def test_initialization():
    P = Playback('test/music_test.mid')
    P.start()
    time.sleep(5)
    P.stop()
    P.close()
    

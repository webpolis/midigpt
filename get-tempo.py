#!/home/nico/anaconda3/bin/python

import sys
import math
from mido import MidiFile

files = sys.argv[1:]

def get_tempo(mid):
    for msg in mid:     # Search for tempo
        if msg.type == 'set_tempo':
            return msg.tempo
    return 500000  

try:
	for file in files:
		midi = MidiFile(file)
		bpm = math.floor(60000000/get_tempo(midi))

		print(bpm)
except:
	pass

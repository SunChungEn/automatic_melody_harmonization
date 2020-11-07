import glob
import os
import numpy as np
import pypianoroll as pr
import pickle
import json
import math

beat_resolution = 24
beat_per_chord = 2

melody_data = []
chord_groundtruth = []
symbol_data = []
length = []
tempos = []
downbeats = []
roman_data = []
sec_data = []
borrowed_data = []
mode_data = []
max_melody_len = 0
max_chord_len = 0
max_event_off = 0
error = 0
# os.chdir("./lead-sheet-dataset/datasets/pianoroll")
for root, dirs, files in os.walk("./lead-sheet-dataset/datasets/pianoroll"):
    for file in files:
        if file.endswith(".npz"):
            print(os.path.join(root, file))
            path_to_symbol = "./lead-sheet-dataset/datasets/event/" + os.path.join(root, file)[40:-4] + "_symbol_nokey.json"
            path_to_roman = "./lead-sheet-dataset/datasets/event/" + os.path.join(root, file)[40:-4] + "_roman.json"
            print(path_to_symbol)
            print(path_to_roman)

            temp = pr.Multitrack(os.path.join(root, file))
            if len(temp.tracks) == 2:
                melody = temp.tracks[0]
                # print(melody.pianoroll)
                print(melody.pianoroll.shape)
                if max_melody_len < melody.pianoroll.shape[0]:
                    max_melody_len = melody.pianoroll.shape[0]

                chord = temp.tracks[1]
                chord_list = []
                for i in range(chord.pianoroll.shape[0]):
                    if i%(beat_resolution*beat_per_chord) == 0:
                        chord_list.append(chord.pianoroll[i])

                chord_np = np.asarray(chord_list)
                # print(chord_np)
                print(chord_np.shape)
                if max_chord_len < chord_np.shape[0]:
                    max_chord_len = chord_np.shape[0]

                melody_data.append(melody.pianoroll)
                chord_groundtruth.append(chord_np)
                length.append(chord_np.shape[0])
                print(temp.tempo.shape)
                tempos.append(temp.tempo)
                print(temp.downbeat.shape)
                downbeats.append(temp.downbeat)

                f = open(path_to_symbol)
                event = json.load(f)
                event_on = []
                event_off = []
                symbol = []
                for chord in event['tracks']['chord']:
                    if chord != None:
                        event_on.append(math.ceil(chord['event_on']))
                        event_off.append(math.ceil(chord['event_off']))
                        symbol.append(chord['symbol'])
                symbol_list = [None for i in range(event_off[-1])]
                symbol_len = event_off[-1]
                if (event_off[-1]//beat_per_chord) > max_event_off:
                    max_event_off = event_off[-1]//beat_per_chord
                for i in range(len(symbol)):
                    for j in range(event_on[i], event_off[i]):
                        symbol_list[j] = symbol[i]
                symbol_list = symbol_list[::beat_per_chord]
                symbol_data.append(symbol_list)
                f.close()

                f = open(path_to_roman)
                event = json.load(f)
                mode_data.append(event['metadata']['mode'])
                event_on = []
                event_off = []
                roman = []
                sec = []
                borrowed = []
                for chord in event['tracks']['chord']:
                    if chord != None:
                        event_on.append(math.ceil(chord['event_on']))
                        event_off.append(math.ceil(chord['event_off']))
                        roman.append(chord['sd'])
                        sec.append(chord['sec'])
                        borrowed.append(chord['borrowed'])
                roman_list = [None for i in range(event_off[-1])]
                sec_list = [None for i in range(event_off[-1])]
                borrowed_list = [None for i in range(event_off[-1])]
                romen_len = event_off[-1]
                if (event_off[-1] // beat_per_chord) > max_event_off:
                    max_event_off = event_off[-1] // beat_per_chord
                for i in range(len(roman)):
                    for j in range(event_on[i], event_off[i]):
                        roman_list[j] = roman[i]
                        sec_list[j] = sec[i]
                        borrowed_list[j] = borrowed[i]
                roman_list = roman_list[::beat_per_chord]
                sec_list = sec_list[::beat_per_chord]
                borrowed_list = borrowed_list[::beat_per_chord]
                roman_data.append(roman_list)
                sec_data.append(sec_list)
                borrowed_data.append(borrowed_list)
                f.close()

                if symbol_len != romen_len:
                    error += 1





for i in range(len(melody_data)):
    melody_data[i] = np.pad(melody_data[i], ((0, max_melody_len-melody_data[i].shape[0]), (0, 0)), constant_values = (0, 0))
for i in range(len(chord_groundtruth)):
    chord_groundtruth[i] = np.pad(chord_groundtruth[i], ((0, max_chord_len-chord_groundtruth[i].shape[0]), (0, 0)), constant_values = (0, 0))
melody_data = np.asarray(melody_data)
chord_groundtruth = np.asarray(chord_groundtruth)
length = np.asarray(length)
print(melody_data.shape)
print(chord_groundtruth.shape)
print(length.shape)
np.save('melody_data', melody_data)
np.save('chord_groundtruth', chord_groundtruth)
np.save('length', length)

f = open('tempos', 'wb')
pickle.dump(tempos, f)
f.close()
f = open('downbeats', 'wb')
pickle.dump(downbeats, f)
f.close()


print('max event off:', max_event_off)
print('len of symbol data:', len(symbol_data))
f = open('symbol_data', 'wb')
pickle.dump(symbol_data, f)
f.close()

print('len of roman data:', len(roman_data))
f = open('roman_data', 'wb')
pickle.dump(roman_data, f)
f.close()

print('len of sec data:', len(sec_data))
f = open('sec_data', 'wb')
pickle.dump(sec_data, f)
f.close()

print('len of borrowed data:', len(borrowed_data))
f = open('borrowed_data', 'wb')
pickle.dump(borrowed_data, f)
f.close()

print('len of mode data:', len(mode_data))
f = open('mode_data', 'wb')
pickle.dump(mode_data, f)
f.close()

print('number of len mismatch:', error)
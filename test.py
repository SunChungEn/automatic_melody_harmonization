from tonal import pianoroll2number, joint_prob2pianoroll96
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pypianoroll import Multitrack, Track
import pypianoroll as pr
import pickle
from matplotlib import pyplot as plt
import os
import random
from metrics import CHE_and_CC, CTD, CTnCTR, PCS, MCTD

def random_choice(batch_size=512, p=0.5):
    # p for masking rate
    #p=0 mask everything
    mask = []
    bin_mask = [0 for i in range(batch_size*272)]
    for i in range(batch_size*272):
        if random.random() > p:
            mask.append(i)
            bin_mask[i] = 1
    return np.asarray(mask), np.asarray(bin_mask)

device = torch.device('cuda:0')
val_size = 500

print('loading data...')
melody_data = np.load('./melody_data.npy')
chord_groundtruth = np.load('./chord_groundtruth.npy')

melody = np.load('./melody_baseline.npy')
lengths = np.load('./length.npy')

f = open('tempos', 'rb')
tempos = pickle.load(f)
f.close()
f = open('downbeats', 'rb')
downbeats = pickle.load(f)
f.close()

print('splitting testing set...')
melody_data = melody_data[:val_size]
chord_groundtruth = chord_groundtruth[:val_size]

val_melody = torch.from_numpy(melody[:val_size]).float()
val_length = torch.from_numpy(lengths[:val_size])

tempos = tempos[:val_size]
downbeats = downbeats[:val_size]


class multitask_model(nn.Module):
    def __init__(self, lstm_dim=12*24*2 + 96 + 1, fc_dim=128):
        super(multitask_model, self).__init__()
        self.bilstm = nn.LSTM(input_size=lstm_dim, hidden_size=fc_dim // 2, num_layers=2, batch_first=True, dropout=0.2,
                              bidirectional=True)
        self.chord_fc = nn.Linear(fc_dim + lstm_dim, 96)
        self.drop1 = nn.Dropout(p=0.2)

    def forward(self, x, length, x2, mask):
        x = torch.cat((x, x2, mask), dim=-1)
        packed_x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        packed_x, (ht, ct) = self.bilstm(packed_x)
        lstmout, _ = pad_packed_sequence(packed_x, batch_first=True, total_length=272)
        concat_x = torch.cat((x, lstmout), dim=-1)
        concat_x = self.drop1(concat_x)
        concat_x = F.relu(concat_x)
        chord = self.chord_fc(concat_x)

        return F.softmax(chord, dim=-1)


print('building model...')
model = multitask_model().to(device)
model.load_state_dict(torch.load('./model_gibbs_96_weight.pth'))

model.eval()
melody, length = val_melody.to(device), val_length.to(device)
print('gibbs sampling...')
with torch.no_grad():
    chord_onehot = torch.zeros(val_size, 272, 96).to(device)
    mask = torch.ones(val_size, 272, 1).to(device)
    chord_pred = model(melody, length, chord_onehot, mask)
    p_max = 1
    p_min = 0.05
    n = 10
    for i in range(n):
        alpha = p_min + ((p_max - p_min) / n) * i
        if i < n - 1:
            print('simpling the ', i + 1, 'th round, alpha=', alpha, end='\r')
        else:
            print('simpling the ', i + 1, 'th round, alpha=', alpha)
        chord_mask, bin_chord_mask = random_choice(batch_size=val_size, p=alpha)
        chord_mask, bin_chord_mask = torch.from_numpy(chord_mask).to(device), torch.from_numpy(bin_chord_mask).to(device)
        chord_pred = torch.reshape(chord_pred, (val_size * 272, -1))
        chord_pred[chord_mask, :] = torch.zeros(96).to(device)
        chord_pred = torch.reshape(chord_pred, (val_size, 272, -1))

        mask = torch.reshape(bin_chord_mask, (val_size, 272, 1))

        chord_pred_new = model(melody, length, chord_pred, mask)

        bin_chord_mask = torch.reshape(bin_chord_mask, (val_size, 272)).unsqueeze(2)

        chord_pred = torch.where(bin_chord_mask == 1, chord_pred_new, chord_pred)

# proceed chord decode
print('proceed chord decode...')
chord_pred = chord_pred.cpu().detach().numpy()
melody = melody.reshape(val_size, 272, 48, 12)
melody = melody.cpu().detach().numpy()
length = length.cpu().detach().numpy()

#cal metrics
f = open('gibbs_metrics.txt', 'w')
m = [0 for i in range(6)]
for i in range(val_size):
    chord_pred_part = chord_pred[i][:length[i]]
    melody_part = melody[i][:length[i]]
    che, cc = CHE_and_CC(chord_pred_part, chord_num=96)
    ctd = CTD(chord_pred_part, chord_num=96)
    ctnctr = CTnCTR(melody_part, chord_pred_part, chord_num=96)
    pcs = PCS(melody_part, chord_pred_part, chord_num=96)
    mctd = MCTD(melody_part, chord_pred_part, chord_num=96)
    m[0] += che
    m[1] += cc
    m[2] += ctd
    m[3] += ctnctr
    m[4] += pcs
    m[5] += mctd
    f.write(str(che) + " " + str(cc) + " " + str(ctd) + " " + str(ctnctr) + " " + str(pcs) + " " + str(mctd) + '\n')
f.close()
print('CHE: ', m[0]/val_size)
print('CC: ', m[1]/val_size)
print('CTD: ', m[2]/val_size)
print('CTnCTR: ', m[3]/val_size)
print('PCS: ', m[4]/val_size)
print('MCTD: ', m[5]/val_size)

# joint_prob = chord_pred
#
# chord_pianoroll = []
# for song in joint_prob:
#     pianoroll = []
#     for beat in song:
#         pianoroll.append(joint_prob2pianoroll96(beat))
#     chord_pianoroll.append(pianoroll)
#
# chord_pianoroll = np.asarray(chord_pianoroll)
#
# accompany_pianoroll = chord_pianoroll * 100
# print(chord_pianoroll.shape)
#
# beat_resolution = 24
# beat_per_chord = 2
#
# # augment chord into frame base
# print('augment chord into frame base...')
# accompany_pianoroll_frame = []
# chord_groundtruth_frame = []
# for acc_song, truth_song in zip(accompany_pianoroll, chord_groundtruth):
#     acc_pianoroll = []
#     truth_pianoroll = []
#     for acc_beat, truth_beat in zip(acc_song, truth_song):
#         for i in range(beat_resolution*beat_per_chord):
#             acc_pianoroll.append(acc_beat)
#             truth_pianoroll.append(truth_beat)
#     accompany_pianoroll_frame.append(acc_pianoroll)
#     chord_groundtruth_frame.append(truth_pianoroll)
#
# accompany_pianoroll_frame = np.asarray(accompany_pianoroll_frame).astype(int)
# chord_groundtruth_frame = np.asarray(chord_groundtruth_frame)
# print('accompany_pianoroll shape:', accompany_pianoroll_frame.shape)
# print('groundtruth_pianoroll shape:', chord_groundtruth_frame.shape)
#
# # length into frame base
# length = length * beat_resolution * beat_per_chord
#
# # write pianoroll
# print('write pianoroll...')
# if not os.path.exists('./gibbs_96_weight_result'):
#     os.makedirs('./gibbs_96_weight_result')
# counter = 0
# for melody_roll, chord_roll, truth_roll, l, tempo, downbeat in zip(melody_data, accompany_pianoroll_frame,
#                                                                         chord_groundtruth_frame, length, tempos,
#                                                                         downbeats):
#     melody_roll, chord_roll, truth_roll = melody_roll[:l], chord_roll[:l], truth_roll[:l]
#
#     track1 = Track(pianoroll=melody_roll)
#     track2 = Track(pianoroll=chord_roll)
#     track3 = Track(pianoroll=truth_roll)
#
#     generate = Multitrack(tracks=[track1, track2], tempo=tempo, downbeat=downbeat, beat_resolution=beat_resolution)
#     truth = Multitrack(tracks=[track1, track3], tempo=tempo, downbeat=downbeat, beat_resolution=beat_resolution)
#
#     pr.write(generate, './gibbs_96_weight_result/generate_' + str(counter) + '.mid')
#     pr.write(truth, './gibbs_96_weight_result/groundtruth_' + str(counter) + '.mid')
#
#     fig, axs = generate.plot()
#     plt.savefig('./gibbs_96_weight_result/generate_' + str(counter) + '.png')
#     plt.close()
#     fig, axs = truth.plot()
#     plt.savefig('./gibbs_96_weight_result/groundtruth_' + str(counter) + '.png')
#     plt.close()
#
#     counter += 1
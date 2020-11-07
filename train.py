import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random

def random_choice(batch_size=512, p=random.uniform(0, 1)):
    # p for masking rate
    #p=0 mask everything
    mask = []
    bin_mask = [0 for i in range(batch_size*272)]
    for i in range(batch_size*272):
        if random.random() > p:
            mask.append(i)
            bin_mask[i] = 1
    return np.asarray(mask), np.asarray(bin_mask)

batch_size = 512
epochs = 10
device = torch.device('cuda:0')
val_size = 500

print('loading data...')
melody = np.load('./melody_baseline.npy')
chord = np.load('./number_96.npy')
chord_onehot = np.load('./onehot_96.npy')
length = np.load('./length.npy')
weight_chord = np.load('./weight_chord.npy')

print('splitting validation set...')
train_melody = melody[val_size:]
val_melody = torch.from_numpy(melody[:val_size]).float()
train_chord = chord[val_size:]
val_chord = torch.from_numpy(chord[:val_size]).float()
train_chord_onehot = chord_onehot[val_size:]
val_chord_onehot = torch.from_numpy(chord_onehot[:val_size]).float()
train_length = length[val_size:]
val_length = torch.from_numpy(length[:val_size])
weight_chord = torch.from_numpy(weight_chord).float().to(device)


class ChordGenerDataset(Dataset):
    def __init__(self, melody, chord, length, chord_onehot):
        self.melody = melody
        self.chord = chord
        self.length = np.expand_dims(length, axis=1)
        self.chord_onehot = chord_onehot

    def __getitem__(self, index):
        x = torch.from_numpy(self.melody[index]).float()
        y = torch.from_numpy(self.chord[index]).float()
        l = torch.from_numpy(self.length[index])
        x2 = torch.from_numpy(self.chord_onehot[index]).float()
        return x, y, l, x2

    def __len__(self):
        return (self.melody.shape[0])


print('creating dataloader...')
dataset = ChordGenerDataset(train_melody, train_chord, train_length, train_chord_onehot)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)


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
optimizer = optim.Adam(model.parameters(), lr=0.005)
lambda1 = lambda epoch: 0.995 ** epoch
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
loss_crossentropy1 = nn.CrossEntropyLoss(weight=weight_chord)
# loss_crossentropy1 = nn.CrossEntropyLoss()

print('start training...')
for epoch in range(epochs):
    print('epoch: ', epoch + 1)
    model.train()
    chord_loss = 0
    for melody, chord, length, chord_onehot in dataloader:
        melody, chord, length, chord_onehot = melody.to(device), chord.to(device), length.to(device), chord_onehot.to(device)
        optimizer.zero_grad()

        # random mask
        chord_mask, bin_chord_mask = random_choice()
        chord_mask, bin_chord_mask = torch.from_numpy(chord_mask).to(device), torch.from_numpy(bin_chord_mask).to(device)
        chord_onehot = torch.reshape(chord_onehot, (batch_size * 272, -1))
        chord_onehot[chord_mask, :] = torch.zeros(96).to(device)
        chord_onehot = torch.reshape(chord_onehot, (batch_size, 272, -1))

        mask = torch.reshape(bin_chord_mask, (batch_size, 272, 1))

        chord_pred = model(melody, length.squeeze(), chord_onehot, mask)

        chord_pred_flatten = []
        chord_flatten = []
        length = length.squeeze()

        bin_chord_mask_flatten = []
        bin_chord_mask = torch.reshape(bin_chord_mask, (batch_size, 272))
        for i in range(batch_size):
            chord_pred_flatten.append(chord_pred[i][:length[i]])
            chord_flatten.append(chord[i][:length[i]])

            bin_chord_mask_flatten.append(bin_chord_mask[i][:length[i]])

        chord_pred_flatten = torch.cat(chord_pred_flatten, dim=0)
        chord_flatten = torch.cat(chord_flatten, dim=0)

        bin_chord_mask_flatten = torch.cat(bin_chord_mask_flatten, dim=0)

        # cut the none target index
        chord_pred_flatten_part = chord_pred_flatten[bin_chord_mask_flatten.nonzero(as_tuple=True)[0], :]
        chord_flatten_part = chord_flatten[bin_chord_mask_flatten.nonzero(as_tuple=True)[0], :]

        chord_flatten_part = chord_flatten_part.squeeze().long()
        ce1 = loss_crossentropy1(chord_pred_flatten_part, chord_flatten_part)
        chord_loss += ce1.item()
        ce1.backward()

        optimizer.step()

    print('chord_loss: ', chord_loss / (17505 // batch_size))

    model.eval()
    val_chord_loss = 0
    melody, chord, length, chord_onehot = val_melody.to(device), val_chord.to(device), val_length.to(device), val_chord_onehot.to(device)

    # random mask
    chord_mask, bin_chord_mask = random_choice(batch_size=val_size)
    chord_mask, bin_chord_mask = torch.from_numpy(chord_mask).to(device), torch.from_numpy(bin_chord_mask).to(device)
    chord_onehot = torch.reshape(chord_onehot, (val_size * 272, -1))
    chord_onehot[chord_mask, :] = torch.zeros(96).to(device)
    chord_onehot = torch.reshape(chord_onehot, (val_size, 272, -1))

    mask = torch.reshape(bin_chord_mask, (val_size, 272, 1))

    chord_pred = model(melody, length.squeeze(), chord_onehot, mask)

    chord_pred_flatten = []
    chord_flatten = []
    length = length.squeeze()

    bin_chord_mask_flatten = []
    bin_chord_mask = torch.reshape(bin_chord_mask, (val_size, 272))
    for i in range(val_size):
        chord_pred_flatten.append(chord_pred[i][:length[i]])
        chord_flatten.append(chord[i][:length[i]])

        bin_chord_mask_flatten.append(bin_chord_mask[i][:length[i]])

    chord_pred_flatten = torch.cat(chord_pred_flatten, dim=0)
    chord_flatten = torch.cat(chord_flatten, dim=0)

    bin_chord_mask_flatten = torch.cat(bin_chord_mask_flatten, dim=0)

    # cut the none target index
    chord_pred_flatten_part = chord_pred_flatten[bin_chord_mask_flatten.nonzero(as_tuple=True)[0], :]
    chord_flatten_part = chord_flatten[bin_chord_mask_flatten.nonzero(as_tuple=True)[0], :]

    chord_flatten_part = chord_flatten_part.squeeze().long()
    ce1 = loss_crossentropy1(chord_pred_flatten_part, chord_flatten_part)
    val_chord_loss += ce1.item()

    print('val_chord_loss: ', val_chord_loss)

torch.save(model.state_dict(), 'model_gibbs_96_weight.pth')
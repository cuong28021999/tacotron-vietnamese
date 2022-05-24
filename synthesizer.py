import os
import IPython.display as ipd
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
import waveglow.denoiser as dens
import matplotlib
import matplotlib.pylab as plt

graph_width = 900
graph_height = 360

tacotron2_pretrained_model = 'Ngan_30'
waveglow_pretrained_model = 'waveglow_256channels_universal_v5.pt'

def plot_data(data, figsize=(int(graph_width/100), int(graph_height/100))):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                    interpolation='none', cmap='inferno')
    fig.canvas.draw()
    plt.show()

thisdict = {}
for line in reversed((open('merged.dict.txt', "r").read()).splitlines()):
    thisdict[(line.split(" ", 1))[0]] = (line.split(" ", 1))[1].strip()

def ARPA(text):
    out = ''
    for word_ in text.split(" "):
        word = word_
        end_chars = ''
        while any(elem in word for elem in r"!?,.;") and len(word) > 1:
            if word[-1] == '!':
                end_chars = '!' + end_chars
                word = word[:-1]
            if word[-1] == '?':
                end_chars = '?' + end_chars
                word = word[:-1]
            if word[-1] == ',':
                end_chars = ',' + end_chars
                word = word[:-1]
            if word[-1] == '.':
                end_chars = '.' + end_chars
                word = word[:-1]
            if word[-1] == ';':
                end_chars = ';' + end_chars
                word = word[:-1]
            else:
                break
        try:
            word_arpa = thisdict[word.upper()]
        except:
            word_arpa = ''
        if len(word_arpa) != 0:
            word = "{" + str(word_arpa) + "}"
        out = (out + " " + word + end_chars).strip()
    if out[-1] != ";":
        out = out + ";"
    return out


hparams = create_hparams()

model = Tacotron2(hparams)
model.load_state_dict(torch.load(tacotron2_pretrained_model)['state_dict'])
_ = model.cuda().eval().half()

# Load WaveGlow
waveglow = torch.load(waveglow_pretrained_model)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = dens.Denoiser(waveglow)



text = 'đây là một câu bằng tiếng việt'
sigma = 0.8
denoise_strength = 0.324
raw_input = True # disables automatic ARPAbet conversion, useful for inputting your own ARPAbet pronounciations or just for testing.
                # should be True if synthesizing a non-English language

for i in text.split("\n"):
    if len(i) < 1: continue;
    print(i)
    if raw_input:
        if i[-1] != ";": i=i+";" 
    else: i = ARPA(i)
    print(i)
    with torch.no_grad(): # save VRAM by not including gradients
        sequence = np.array(text_to_sequence(i, ['basic_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        plot_data((mel_outputs_postnet.float().data.cpu().numpy()[0],
                alignments.float().data.cpu().numpy()[0].T))
        audio = waveglow.infer(mel_outputs_postnet, sigma=sigma); print(""); ipd.display(ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate))

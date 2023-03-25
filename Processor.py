
import torch
import torch.nn as nn
import numpy as np
import torchaudio.compliance.kaldi as kaldi

index_map = {}
class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
       with open('.//conf//phonesid.txt','r') as char_map_str:
            char_map_str = char_map_str
            self.char_map = {}
            self.index_map = {}
            for line in char_map_str:
                ch, index = line.split()
                self.char_map[ch] = int(index)
                self.index_map[int(index)] = ch
       with open('.//conf//phonemap.txt','r') as word_map_str:
            word_map_str=word_map_str
            self.word_map={}
            for line in word_map_str:
                word,phones=line.split()
                self.word_map[word]=str(phones)
    def text_to_int(self, text):
        """ Create target label sequence """
        int_sequence = []
        phones = self.word_map.get(text)
        if(phones is None):print(text)
        # Split target word label
        chindex=phones.split('-')
        
  
        for c in chindex:
            ch = self.char_map.get(c)
            if(ch is None):print(c)
            int_sequence.append(int(ch))

        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('>', ' ')

text_transform=TextTransform()

def data_processing(data):
    mfccs = []
    labels = []
    audio_lengths = []
    label_lengths = []

    for (waveform, sample_rate, utterance) in data:
        mfcc=kaldi.fbank(waveform,num_mel_bins=80,
                        dither=0.0,energy_floor=0.0,
                        sample_frequency=sample_rate)   
       
        mfccs.append(mfcc)
    #Get target sequence
        label = torch.IntTensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
    # Audio length after subsampling4
        audio_lengths.append(mfcc.shape[0])
        label_lengths.append(len(label))

    mfccs = nn.utils.rnn.pad_sequence(mfccs, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True,padding_value=-1)
    return mfccs, labels, audio_lengths, label_lengths

def calcmvn(feats,cmvn_mean,cmvn_var):
    m,n=feats.size()
    feats_norm=feats
    for idx in range(m):
        for j in range(n):
            feats_norm[idx][j]=(feats[idx][j]-cmvn_mean[j])/cmvn_var[j]
    return feats_norm



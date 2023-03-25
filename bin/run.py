
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import os

from encoder import *
from CTC import *
from joint import *
from predictor import *
from transducer import *

import numpy as np

def getword_from_id(wordindex,hyps,vocab_size):
    word_seq=[]
    print(hyps)
    for i in hyps[0]:
        if(i>1 and i<vocab_size-1):
            word_seq.append(wordindex[i])
    print(word_seq)


encoder = conformerencoder(input_size=80,num_blocks=2,out_size=80,train_flag=True)
ctc=CTC(74,80)
Predictor=RNNPredictor(voca_size=74,embed_size=80,output_size=80,hidden_size=80,num_layers=2)

joint=TransducerJoint(voca_size=74,enc_output_size=80,pred_output_size=80,join_dim=80)

model=Transducer(vocab_size=74,encoder=encoder,ctc=ctc,predictor=Predictor,joint=joint,transducer_weight=0.5)
checkpoit=torch.load('./rnntmodel.pt',map_location='cpu')
model.load_state_dict(checkpoit,strict=False)
# store word id
word_index={}
# Count decode result

with open('.//conf//phonesid.txt') as char_map:
    for line in char_map:
        word,index=line.split()
        word_index[int(index)]=word
       
wavdir='.//googlecommand//testwav//eight'

file_paths=[]

for root, directories, files in os.walk(wavdir):
    for filename in files:
        filepath = os.path.join(root, filename)
        file_paths.append(filepath)


for wave in file_paths:
    audio_data,samplerate=torchaudio.load(wave)
    print(wave)
   
    mfdata=kaldi.fbank(audio_data,num_mel_bins=80,
                        dither=0.0,energy_floor=0.0,
                        sample_frequency=samplerate)   
    m,n=mfdata.size()
    
    inputdata=mfdata.reshape(1,m,n)
    input_len=torch.zeros((1))
    input_len[0]=m
   # print(maxindex)

    hyp=model.greedy_search(inputdata,input_len)
    getword_from_id(word_index,hyp,74)
    #print(outputdata[0]
   


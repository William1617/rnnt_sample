
from encoder import *

from CTC import *
import torch
import numpy as np
from transducer import *
from joint import *
from predictor import *

#cmvn_mean=torch.Tensor(np.loadtxt('cmvn_mean.txt'))
#cmvn_var=torch.Tensor(np.loadtxt('cmvn_var.txt'))
#istd=1/cmvn_var
encoder = conformerencoder(input_size=80,num_blocks=2,out_size=80,train_flag=True)
ctc=CTC(74,80)
Predictor=RNNPredictor(voca_size=74,embed_size=80,output_size=80,hidden_size=80,num_layers=2)

joint=TransducerJoint(voca_size=74,enc_output_size=80,pred_output_size=80,join_dim=80)

model=Transducer(vocab_size=74,encoder=encoder,ctc=ctc,predictor=Predictor,joint=joint,transducer_weight=0.5)
checkpoit=torch.load('./rnntmodel.pt',map_location='cpu')
model.load_state_dict(checkpoit,strict=False)

quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
print(quantized_model)
script_quant_model = torch.jit.script(quantized_model)
script_quant_model.save('./rnntmodel.zip')



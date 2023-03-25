
import time
import random
import logging
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

from audioloader import audiodataset
from Processor import data_processing

from torch.nn.utils import clip_grad_norm_
from transducer import *
from encoder import conformerencoder
from joint import TransducerJoint
from predictor import *

import numpy as np

def train(train_loader,dev_loader,out_dir,vocab_size):

    logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
    random.seed(1024)
    torch.manual_seed(1024)
    torch.cuda.manual_seed_all(1024)
    patience=20
    #cmvn_mean=torch.Tensor(np.loadtxt('cmvn_mean.txt'))
    # #cmvn_var=torch.Tensor(np.loadtxt('cmvn_var.txt'))
    # #istd=1/cmvn_var

    device = torch.device('cuda')
    encoder = conformerencoder(input_size=80,num_blocks=2,out_size=80,train_flag=True)
    ctc=CTC(vocab_size,80)
    Predictor=RNNPredictor(voca_size=vocab_size,embed_size=80,output_size=80,hidden_size=80,num_layers=2)

    joint=TransducerJoint(voca_size=vocab_size,enc_output_size=80,pred_output_size=80,join_dim=80)

    model=Transducer(vocab_size=vocab_size,encoder=encoder,ctc=ctc,predictor=Predictor,joint=joint,transducer_weight=0.5)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
  #  warmup_step=25000
   
   # scheduler=WarmupLR(optimizer,warmup_step)
    #scheduler.set_step(-1)

    prev_loss = 10000
    lr = optimizer.param_groups[0]['lr']
    model_path=out_dir
   
    
    for epoch in range(1, 500):
        losses = []
        start_time = time.time()
        for i, (xs, ys, xlen, ylen) in enumerate(train_loader):
            if i == 5000:
                break
            x = Variable(torch.FloatTensor(xs))
            x = torch.squeeze(x,1).transpose(1,2)
            x=x.to(device)
           
            x_len=Variable(torch.IntTensor(xlen))
            x_len=x_len.to(device)
          
            y=ys.to(device) 
         
    
            yl=Variable(torch.IntTensor(ylen))
            yl=yl.to(device)
            model.train()
            optimizer.zero_grad()
           

            loss_dict = model(x, x_len,y,yl)
            loss=loss_dict['loss']

            loss.backward()

            losses.append(loss)
            
            optimizer.step()
          

        tol_loss = sum(losses) /len(train_loader)
       
        val_losses = []
        model.eval()
        with torch.no_grad():
            for i, (xs, ys, xlen, ylen) in enumerate(dev_loader):
              
                x = Variable(torch.FloatTensor(xs))
                x = torch.squeeze(x,1).transpose(1,2)
                x=x.to(device)
                x_len=Variable(torch.IntTensor(xlen))
        
                x_len=x_len.to(device)

                yl=Variable(torch.IntTensor(ylen))
                yl=yl.to(device)
                y=ys.to(device)
               
                loss_dict = model(x,x_len,y,yl)
                val_loss = loss_dict['loss']

                val_losses.append(val_loss)

        tol_valoss=sum(val_losses)/len(dev_loader)
        logging.info('[Epoch %d] time cost %.2fs, train loss %.2f; cv loss %.2f,  lr %.3e'%(
            epoch, time.time()-start_time, tol_loss, tol_valoss, lr
        ))
        # Save checkpoint
        checkpoint = {"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
      
        # Early stopping
        if tol_valoss < prev_loss:
            prev_loss = tol_valoss
            torch.save(model.state_dict(), model_path)  
            torch.save(checkpoint,'rnntmodel')   
        else:
            patience=patience-1
        if patience ==0:
            break
   
if __name__ == '__main__':
    train_dir = './/googlecommand//trainwav//'
    dev_dir = './/googlecommand//testwav//'
  
    train_set=audiodataset(train_dir)
    dev_set=audiodataset(dev_dir)
    train_loader = data.DataLoader(dataset=train_set, pin_memory=False,
                                batch_size=8,
                                shuffle=True,
                                num_workers=0,
                                collate_fn=lambda x: data_processing(x))
    dev_loader = data.DataLoader(dataset=dev_set, pin_memory=False,
                                batch_size=8,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=lambda x: data_processing(x))
    train(train_loader,dev_loader,'rnntmodel.pt',74)
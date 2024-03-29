from typing import Dict, List, Optional

import torch
import torchaudio
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from predictor import PredictorBase
from search import *
from mask import *
import torchaudio
import fast_rnnt

class Transducer(nn.Module):
    """Transducer-ctc-attention hybrid Encoder-Predictor-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        encoder: nn.Module,
        predictor: PredictorBase,
        joint: nn.Module,
        
        blank: int =0 ,
        ignore_id: int = -1,
        use_fast_rnnt :bool = False,
    
    ) -> None:
        super().__init__()
    
        self.blank = blank
       
        self.encoder=encoder
        
        self.predictor = predictor
        self.joint = joint
        self.bs = None
        self.ignore_id=ignore_id
        self.fast_rnnt=use_fast_rnnt

        # Note(Mddct): decoder also means predictor in transducer,
        # but here decoder is attention decoder

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + predictor + joint + loss
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)

        # Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1).int()
        # predictor
        ys_in_pad = add_blank(text, self.blank, self.ignore_id)
        predictor_out = self.predictor(ys_in_pad)
        rnnt_text = text.to(torch.int32)
        rnnt_text = torch.where(rnnt_text == self.ignore_id, 0,
                                rnnt_text).to(torch.int32)
        rnnt_text_lengths = text_lengths.to(torch.int32)
        encoder_out_lens = encoder_out_lens.to(torch.int32)
        batchsize=encoder_out_lens.shape[0]

        if(self.fast_rnnt):
            
            boundary = torch.zeros((batchsize, 4), dtype=torch.int64)
            boundary[:, 3] = encoder_out_lens.to(torch.int64)
            boundary[:, 2] = rnnt_text_lengths.to(torch.int64)
            simple_loss, (px_grad, py_grad) = fast_rnnt.rnnt_loss_simple(
                am=encoder_out,lm=predictor_out,symbols=rnnt_text.to(torch.int64),
                termination_symbol=0,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

            s_range = 3  # can be other values
            ranges = fast_rnnt.get_rnnt_prune_ranges(
                px_grad=px_grad,py_grad=py_grad,
                boundary=boundary,s_range=s_range,
            )

            am_pruned, lm_pruned = fast_rnnt.do_rnnt_pruning(am=encoder_out, lm=predictor_out, ranges=ranges)
            logits = self.joint(am_pruned, lm_pruned,fast_rnnt=True)
            pruned_loss = fast_rnnt.rnnt_loss_pruned(
                logits=logits,symbols=rnnt_text, ranges=ranges,
                termination_symbol=0,boundary=boundary,reduction="sum",
            )
            loss=0.5*simple_loss+pruned_loss

        else:
            joint_out = self.joint(encoder_out, predictor_out)
            loss = torchaudio.functional.rnnt_loss(joint_out,
                                                  rnnt_text,
                                                  encoder_out_lens,
                                                  text_lengths.int(),
                                                  blank=self.blank,
                                                  reduction='mean')
       
        # NOTE: 'loss' must be in dict
        return loss
    
    
    
    

    def greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        simulate_streaming: bool = False,
        n_steps: int = 64,
    ) -> List[List[int]]:
        """ greedy search
        Args:
            speech (torch.Tensor): (batch=1, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        # TODO(Mddct): batch decode
        assert speech.size(0) == 1
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        # TODO(Mddct): forward chunk by chunk
        _ = simulate_streaming
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self.encoder(
            speech,
            speech_lengths
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum()
        hyps = basic_greedy_search(self,
                                   encoder_out,
                                   encoder_out_lens,
                                   n_steps=n_steps)

        return hyps

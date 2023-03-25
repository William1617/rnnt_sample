from typing import List, Optional, Tuple

import torch
from torch import nn
from typeguard import check_argument_types



def ApplyPadding(input, padding, pad_value) -> torch.Tensor:
    """
    Args:
        input:   [bs, max_time_step, dim]
        padding: [bs, max_time_step]
    """
    return padding * pad_value + input * (1 - padding)


class PredictorBase(torch.nn.Module):

    # NOTE(Mddct): We can use ABC abstract here, but
    # keep this class simple enough for now
    def __init__(self) -> None:
        super().__init__()

    def init_state(self,
                   batch_size: int,
                   device: torch.device,
                   method: str = "zero") -> List[torch.Tensor]:
        _, _, _ = batch_size, method, device
        raise NotImplementedError("this is a base precictor")

    def batch_to_cache(self,
                       cache: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        _ = cache
        raise NotImplementedError("this is a base precictor")

    def cache_to_batch(self,
                       cache: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        _ = cache
        raise NotImplementedError("this is a base precictor")

    def forward(
        self,
        input: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ):
        _, _, = input, cache
        raise NotImplementedError("this is a base precictor")

    def forward_step(
            self, input: torch.Tensor, padding: torch.Tensor,
            cache: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        _, _, _, = input, padding, cache
        raise NotImplementedError("this is a base precictor")


class RNNPredictor(PredictorBase):

    def __init__(self,
                 voca_size: int,
                 embed_size: int,
                 output_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool = True,
                 embed_dropout: float=0.1,
                 dropout: float = 0.1) -> None:
        assert check_argument_types()
        super().__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        # disable rnn base out projection
        self.embed = nn.Embedding(voca_size, embed_size)
        self.dropout = nn.Dropout(embed_dropout)
        # NOTE(Mddct): rnn base from torch not support layer norm
        # will add layer norm and prune value in cell and layer
        # ref: https://github.com/Mddct/neural-lm/blob/main/models/gru_cell.py
        self.rnn = torch.nn.LSTM(input_size=embed_size,
                                              hidden_size=hidden_size,
                                              num_layers=num_layers,
                                              bias=bias,
                                              batch_first=True,
                                              dropout=dropout)
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        input: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): [batch, max_time).
            padding (torch.Tensor): [batch, max_time]
            cache : rnn predictor cache[0] == state_m
                    cache[1] == state_c
        Returns:
            output: [batch, max_time, output_size]
        """

        # NOTE(Mddct): we don't use pack input format
        embed = self.embed(input)  # [batch, max_time, emb_size]
        embed = self.dropout(embed)
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if cache is None:
            state = self.init_state(batch_size=input.size(0),
                                    device=input.device)
            states = (state[0], state[1])
        else:
            assert len(cache) == 2
            states = (cache[0], cache[1])
        out, (m, c) = self.rnn(embed, states)
        out = self.projection(out)

        # NOTE(Mddct): Although we don't use staate in transducer
        # training forward, we need make it right for padding value
        # so we create forward_step for infering, forward for training
        _, _ = m, c
        return out

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        method: str = "zero",
    ) -> List[torch.Tensor]:
        assert batch_size > 0
        # TODO(Mddct): xavier init method
        _ = method
        return [
            torch.zeros(1 * self.n_layers,
                        batch_size,
                        self.hidden_size,
                        device=device),
            torch.zeros(1 * self.n_layers,
                        batch_size,
                        self.hidden_size,
                        device=device)
        ]

    def forward_step(
            self, input: torch.Tensor, padding: torch.Tensor,
            cache: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            input (torch.Tensor): [batch_size, time_step=1]
            padding (torch.Tensor): [batch_size,1], 1 is padding value
            cache : rnn predictor cache[0] == state_m
                    cache[1] == state_c
        """
        assert len(cache) == 2
        state_m, state_c = cache[0], cache[1]
        embed = self.embed(input)  # [batch, 1, emb_size]
        embed = self.dropout(embed)
        out, (m, c) = self.rnn(embed, (state_m, state_c))

        out = self.projection(out)
        m = ApplyPadding(m, padding, state_m)
        c = ApplyPadding(c, padding, state_c)
        return (out, [m, c])

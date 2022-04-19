import torch
from torch import Tensor

from ct.functional.ct_loss import contrastive_token_loss


class ContrastiveTokenLoss(torch.nn.Module):
    """A Pytorch Module wrapper for the contrastive_token_loss function.

        Args:
            ignore_index (int, optional): Default padding token id. Defaults to -100.
            pad_id (int, optional): Specified padding token id. Used to mask out irrelevant preceding tokens. Defaults to 0.
            ct_length (Union[int, float], optional): When it's a float value and in [0, 1], it's a portion to the original sequence length;
            when it's larger than 1, it specifies the absolute CT length. Defaults to 0.25.
            preced_m_negatives (Union[int, float], optional): When it's a float value and in [0, 1], it's a portion to the CT sequence length;
            when it's larger than 1, it specifies the absolute negative window size. Defaults to 0.5.

        Returns:
            Tensor: Calculated CT loss.
    """
    def __init__(
        self,
        ignore_index=-100,
        pad_id=0,
        ct_length=0.25,
        preced_m_negatives=0.5,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.pad_id = pad_id
        self.ct_length = ct_length
        self.preced_m_negatives = preced_m_negatives
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return contrastive_token_loss(
            input, target, self.ignore_index,
            self.pad_id, self.ct_length,
            self.preced_m_negatives,
        )

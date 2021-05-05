from typing import Dict, Iterator, List,Tuple

import torch
import torch.nn as nn                            
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder


class MatchPyramid(nn.Module):
    '''
    Paper: Text Matching as Image Recognition, Pang et al., AAAI'16
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 conv_output_size: List[int],
                 conv_kernel_size: List[Tuple[int,int]],
                 adaptive_pooling_size: List[Tuple[int,int]]):

        super(MatchPyramid, self).__init__()

        self.word_embeddings = word_embeddings

        if len(conv_output_size) != len(conv_kernel_size) or len(conv_output_size) != len(adaptive_pooling_size):
            raise Exception("conv_output_size, conv_kernel_size, adaptive_pooling_size must have the same length")

        # todo

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"] > 0).float()
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        # todo

        return output

from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataloader import PyTorchDataLoader
prepare_environment(Params({})) # sets the seeds to be fixed

import torch

from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from data_loading import *
from model_knrm import *
from model_conv_knrm import *
from model_tk import *

# change paths to your data directory
config = {
    "vocab_directory": "../data/allen_vocab_lower_10",
    "pre_trained_embedding": "../data/glove.42B.300d.txt",
    "model": "knrm",
    "train_data": "../data/triples.train.tsv",
    "validation_data": "../data/tuples.validation.tsv",
    "test_data":"../data/tuples.test.tsv",
}

#
# data loading
#

vocab = Vocabulary.from_files(config["vocab_directory"])
tokens_embedder = Embedding(vocab=vocab,
                           pretrained_file= config["pre_trained_embedding"],
                           embedding_dim=300,
                           trainable=True,
                           padding_index=0)
word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

# recommended default params for the models (but you may change them if you want)
if config["model"] == "knrm":
    model = KNRM(word_embedder, n_kernels=11)
elif config["model"] == "conv_knrm":
    model = Conv_KNRM(word_embedder, n_grams=3, n_kernels=11, conv_out_dim=128)
elif config["model"] == "tk":
    model = TK(word_embedder, n_kernels=11, n_layers = 2, n_tf_dim = 300, n_tf_heads = 10)


# todo optimizer, loss 

print('Model',config["model"],'total parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Network:', model)

#
# train
#

_triple_reader = IrTripleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
_triple_reader = _triple_reader.read(config["train_data"])
_triple_reader.index_with(vocab)
loader = PyTorchDataLoader(_triple_reader, batch_size=32)

for epoch in range(2):

    for batch in Tqdm.tqdm(loader):
        # todo train loop
        pass


#
# eval (duplicate for validation inside train loop - but rename "loader", since
# otherwise it will overwrite the original train iterator, which is instantiated outside the loop)
#

_tuple_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
_tuple_reader = _tuple_reader.read(config["test_data"])
_tuple_reader.index_with(vocab)
loader = PyTorchDataLoader(_tuple_reader, batch_size=128)

for batch in Tqdm.tqdm(loader):
    # todo test loop 
    # todo evaluation
    pass

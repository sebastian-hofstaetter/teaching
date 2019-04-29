from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
prepare_environment(Params({})) # sets the seeds to be fixed

import torch

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from data_loading import *
from model_knrm import *
from model_conv_knrm import *
from model_match_pyramid import *

# change paths to your data directory
config = {
    "vocab_directory": "../data/allen_vocab_lower_10",
    "pre_trained_embedding": "../data/glove.42B.300d.txt",
    "model": "knrm",
    "train_data":"../data/triples.train.tsv",
    "validation_data":"../data/tuples.validation.tsv",
    "test_data":"../data/tuples.test.tsv",
}

#
# data loading
#

vocab = Vocabulary.from_files(config["vocab_directory"])
tokens_embedder = Embedding.from_params(vocab, Params({"pretrained_file": config["pre_trained_embedding"],
                                                      "embedding_dim": 300,
                                                      "trainable": True,
                                                      "padding_index":0}))

word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

# recommended default params for the models (but you may change them if you want)
if config["model"] == "knrm":
    model = KNRM(word_embedder, n_kernels=11)
elif config["model"] == "conv_knrm":
    model = Conv_KNRM(word_embedder, n_grams=3, n_kernels=11, conv_out_dim=128)
elif config["model"] == "match_pyramid":
    model = MatchPyramid(word_embedder, conv_output_size=[16,16,16,16,16], conv_kernel_size=[[3,3],[3,3],[3,3],[3,3],[3,3]], adaptive_pooling_size=[[36,90],[18,60],[9,30],[6,20],[3,10]])


# todo optimizer, loss 

print('Model',config["model"],'total parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Network:', model)

#
# train
#

_triple_loader = IrTripleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30,tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())) # already spacy tokenized, so that it is faster 

_iterator = BucketIterator(batch_size=64,
                           sorting_keys=[("doc_pos_tokens", "num_tokens"), ("doc_neg_tokens", "num_tokens")])

_iterator.index_with(vocab)

for epoch in range(2):

    for batch in Tqdm.tqdm(_iterator(_triple_loader.read(config["train_data"]), num_epochs=1)):
        # todo train loop
        pass


#
# eval (duplicate for validation inside train loop)
#

_tuple_loader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30) # not spacy tokenized already (default is spacy)
_iterator = BucketIterator(batch_size=128,
                           sorting_keys=[("doc_tokens", "num_tokens"), ("query_tokens", "num_tokens")])
_iterator.index_with(vocab)

for batch in Tqdm.tqdm(_iterator(_tuple_loader.read(config["test_data"]), num_epochs=1)):
    # todo test loop 
    # todo evaluation
    pass
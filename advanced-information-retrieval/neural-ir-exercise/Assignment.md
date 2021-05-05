# Neural IR & Extractive QA Exercise 🔎

In this exercise your group is implementing neural network re-ranking models, using pre-trained extractive QA models, and analyze their behavior with respect to our FiRA data.

We provide some starter code & prepare a subset of the MS MARCO retrieval dataset so that starting for you is easier. We are utilizing PyTorch, AllenNLP, and the Huggingface transformers library (Python 3). PyTorch allows you to be very flexible in the definition of the training process (good for our pairwise training) & debug the code in a natural way - in your IDE on your local PC (with NVIDIA GPU for full training, but developing also works with a CPU only) or Google Colab (which offers a free GPU in the cloud with a jupyter notebook) the same as any other Python code. 

# Goals

In total you need at least 50% of the points. There is no requirement on how you achieve the minimum points for this exercise **other than the report. The report is mandatory** (otherwise we have a hard time getting a good overview of what you did). 

## Part 1 - Neural Re-Ranking **45 points**

Implement 3 neural architectures based on the kernel-pooling paradigm to perform re-ranking in ``src/re_ranking.py`` (KNRM, Conv-KNRM, TK)
	
- Implement: the 3 (KNRM, Conv-KNRM, TK) model classes **25 points**
   - Show that you understood what happens by adding comments to difficult parts of the model (what tensor dimensions represent, what gets summed up, etc..)
- Implement: training process & result evaluation **10 points**
    - Including early stopping based on the validation set
- Evaluate: Compute a test set evaluation at the end
	- MS-MARCO Sparse Labels **5 points**
	- FiRA fine-grained Labels **5 points**
	
## Part 2 - Extractive QA **30 points**

Use the transformers library to download a pre-trained extractive QA model from the HuggingFace model hub and run the extractive QA pipeline on the top-1 neural re-ranking result as well as on the gold-label pairs of FiRA. 

- Get to know the HuggingFace library & model hub
- Select a pre-trained extractive QA model from the model hub to use
- Implement code  in ``src/extractive_qa.py`` to load the model, tokenize query passage pairs, and run inference, store results with the HuggingFace library
	- The goal of extractive QA is to provide one or more text-spans that answers a given (query,passage) pair
- Evaluate both your top-1 (or more) passage results from the best re-ranking model using **fira.qrels.qa-answers.tsv** (only evaluate the overlap of pairs that are in the result and the qrels) + the provided FiRA gold-label pairs **fira.qrels.qa-tuples.tsv** using the provided qa evaluation methods in core_metrics.py with the FiRA QA labels

The data format for the FiRA data is as follows:

**fira.qrels.qa-answers.tsv**

``queryid documentid relevance-grade text-selection (multiple answers possible, split with tab)``

**fira.qrels.qa-tuples.tsv**

``queryid documentid relevance-grade query-text document-text text-selection (multiple answers possible, split with tab)``

**fira.qrels.retrieval.tsv** (same format as msmarco qrels)

``queryid hardcoded-0 documentid relevance-grade``

## Part 3 - Data/Result Analysis & Report **25 points**

- Visualize results or inner-workings of the neural re-ranking or extractive QA models or something interesting based on the FiRA annotations **15 points**

   - First step: Decide what to visualize and how - tell a story (The only things not allowed are plain line & bar chart plots without a good reason for them)
		- The visualization can be for either the re-ranking or the extractive QA part
	- Implement code to prepare the data for it and create the visualizations
	- For FiRA annotations especially, try to find some visualization using text.

- Write a report **10 points**

	- Detail problems & their solutions you encountered
	- Your evaluation results
	- Tell the story of the visualization with output of the visualization
		
# Bonus Points 🎉

* Finding and fixing bugs in the materials we provide (this exercise is new, so there might be bugs in the starter code) - **10 points each (20 max)**

* Try out more variants or other re-ranking models - **up to 15 points**

* Do something cool on top of the exercise goals - **up to 15 points**

# Getting Started

*  The starter pack for 1. includes:
	- Data loading code for re-ranking models (which provides an iterator returning padded, ready-to-go batches of word ids)
	- Neural re-ranking model skeletons (helper code)
	- Evaluation code (core_metrics.py)

* We recommend that you use an Anaconda environment and install the requirements.txt with pip

* You should use an Nvidia GPU, if you don't have one locally, you can use Google Colab (https://colab.research.google.com/), which offers a free GPU in the cloud -> use the train.ipynb for this instead of train.py

## Provided Data

* Provided data: AllenNLP vocabulary (collection specific, in two sizes: use the _10 = min of 10 occurrences in the collection if you have memory problems with the _5), train triples, evaluation tuples (validation & test) with 2.000 queries each and the top 40 BM25 results per query, relevance judgments (qrels, one file covering both validation & test)

* Download a pre-trained glove embedding from: http://nlp.stanford.edu/data/glove.42B.300d.zip

# Notes

* The assignment text is purpusfully kept simple. If you see two diverging paths of doing things, pick the one that makes the most sense for you and write a short note in the report about the choice and why you decided on one way of doing it. In any case, you will receive the points for the part. 

* Use either ``src/re_ranking.py`` or ``src/re_ranking.ipynb`` for the final version and delete the other file (same for extractive QA)

* The final version is the code and report in the main branch of your group's repository

* A few hints and tricks:
    - Use the Adam optimizer (keep in mind that it adds 2x the memory consumption for the model parameters)
    - The iterators do not guarantee a fixed batch size (the last one will probably be smaller)
	- The batch tensors also have no fixed size, the max values in the readers are just to cap outliers (the size will be based on the biggest sample in the batch (per tensor) and the others padded with 0 to the same length)
    - The kernel models need masking after the kernels -> the padding 0's will become non-zero, because of the kernel (but when summed up again will distort the output) 
    - Feel free to change the default parameters in the starter code (in general they should give you good results)

* Make use of the GPU (with .cuda() - see the PyTorch documentation for details, you can switch it based on if a GPU is available). The provided data should keep the models under 5GB of GPU RAM when using the ADAM optimizer

* KNRM should reach ~0.19 MRR@10 and Conv-KNRM should go up to ~ 0.22 MRR@10 and TK should again improve over Conv-KNRM (The values here are lower than the leaderboard/what is shown in the lecture, because we only provide you with a subset of the training & evaluation data)

# References

* KNRM: Chenyan Xiong, Zhuyun Dai, Jamie Callan, Zhiyuan Liu, and Russell Power. 2017. End-to-End Neural Ad-hoc Ranking with Kernel Pooling. In Proc. of SIGIR.
* Conv-KNRM: Zhuyun Dai, Chenyan Xiong, Jamie Callan, and Zhiyuan Liu. 2018. Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search. In Proc. of WSDM
* TK: S. Hofstätter, M. Zlabinger, and A. Hanbury 2020. Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking. In Proc. of ECAI 

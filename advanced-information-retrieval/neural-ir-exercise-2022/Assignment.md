# Neural IR & Extractive QA Exercise 🔎

In this exercise your group is 1) aggregating raw FiRA judgements to usable labels; 2) implementing neural network re-ranking models, 3) using pre-trained extractive QA models and analyze their behavior with respect to our FiRA data.

We provide some starter code & prepare a subset of the MS MARCO retrieval dataset so that starting to train is easier. We are utilizing PyTorch, AllenNLP, and the HuggingFace transformers library (Python 3). PyTorch allows you to be very flexible in the definition of the training process (good for our pairwise training) & debug the code in a natural way - in your IDE on your local PC (with NVIDIA GPU for full training, but developing also works with a CPU only) or Google Colab (which offers a free GPU in the cloud with a jupyter notebook) the same as any other Python code. 

# Goals

In total you need at least 50% of the points. There is no requirement on how you achieve the minimum points for this exercise **other than the report. The report is mandatory** (otherwise we have a hard time getting a good overview of what you did). 

## Part 1 - FiRA Data to Test Collection Preparation **20 points**

This year we annotated out-of-domain data (in comparison to in-domain-training MSMARCO) from three datasets: TREC Robust 04, DBPedia Entity, and TripClick. We already observed that the nature of these three tasks was more noisy to annotate than relatively simple and precise MSMARCO queries from last year. Therefore, we need to first explore robust ways of aggregating the raw judgements (multiple per pair) to labels (1 grade per pair). The judgements are anonymized, but still contain all additional information, such as anonymized-but-trackable user-id, time to annotate, time of day, the judgement value, and the optional selection of text. 

We provide a simple baseline where we used a simple majority voting, only looking at the judgement value, and a heuristic to take the higher grade if we have a tie. The goal of this part is to find a more sophisticated method of aggregating raw judgement to usable and robust labels:

- Define and describe a hypothesis of what you are going to do and why it makes sense (you are free to choose what you do, except for re-using the simple baseline unchanged, it is ok to also drop queries altogether if argued and shown why)
- Implement an algorithm that aggregates raw judgements to labels (either 4-grades or binary) in ``src/judgement_aggregation.py``
- Analyze the results of the aggregation, by randomly(!) picking some examples and conducting a meta-judgement of the raw-judgements + the quality of the aggregations based on the text contents of query and passage
- Describe your results in the report, use your labels for evaluation in part 2


## Part 2 - Neural Re-Ranking **40 points**

Implement 3 neural architectures based on the kernel-pooling paradigm to perform re-ranking in ``src/re_ranking.py`` (KNRM, Conv-KNRM, TK)
	
- Implement: the 3 (KNRM, Conv-KNRM, TK) model classes **20 points**
   - Show that you understood what happens by adding comments to difficult parts of the model (what tensor dimensions represent, what gets summed up, etc..)
- Implement: training process & result evaluation **10 points**
    - Including early stopping based on the validation set
	   - Use the **msmarco_tuples.validation.tsv** input to feed the neural models and **msmarco_qrels.txt** qrels to evaluate the output
- Evaluate: Compute a test set evaluation at the end  **10 points**
	- MS-MARCO sparse labels
	  - Use the **msmarco_tuples.test.tsv** input to feed the neural models and **msmarco_qrels.txt** qrels to evaluate the output
	- FiRA-2022 fine-grained labels on out-of-domain data
	  - Use your created created labels from part 1
	     - Use the **fira-2022.tuples.tsv** input to feed the neural models and your qrels from part 1 to evaluate the output
	  - Compare these results with our baseline label creation method
	     - Use the **fira-2022.tuples.tsv** input to feed the neural models and **fira-2022.baseline-qrels.tsv** qrels to evaluate the output
	  - Explore & describe the differences in metrics between the baseline and your label creation 
	
## Part 3 - Extractive QA **30 points**

Use the transformers library to download a pre-trained extractive QA model from the HuggingFace model hub and run the extractive QA pipeline on the top-1 neural re-ranking result of the MSMARCO test set as well as on the gold-label pairs of MSMARCO-FiRA-2021 (created in 2021). 

- Get to know the HuggingFace library & model hub
- Select a pre-trained extractive QA model from the model hub to use
- Implement code  in ``src/extractive_qa.py`` to load the model, tokenize query passage pairs, and run inference, store results with the HuggingFace library
	- The goal of extractive QA is to provide one or more text-spans that answers a given (query,passage) pair
- Evaluate both your top-1 (or more) MSMARCO passage results from the best re-ranking model using **msmarco-fira-21.qrels.qa-answers.tsv** (only evaluate the overlap of pairs that are in the result and the qrels) + the provided FiRA gold-label pairs **msmarco-fira-21.qrels.qa-tuples.tsv** using the provided qa evaluation methods in core_metrics.py with the MSMARCO-FiRA-2021 QA labels

The data format for the FiRA data is as follows:

**msmarco-fira-21.qrels.qa-answers.tsv**

``queryid documentid relevance-grade text-selection (multiple answers possible, split with tab)``

**msmarco-fira-21.qrels.qa-tuples.tsv**

``queryid documentid relevance-grade query-text document-text text-selection (multiple answers possible, split with tab)``


## Part 4 - Report **10 points**

- Write a report 
	- Explain your data aggregation process from part 1
	- Detail problems & their solutions you encountered
	- Your evaluation results (from parts 2 & 3)
		
# Bonus Points 🎉

* Finding and fixing bugs in the materials we provide - **10 points each (20 max)**

# Getting Started

*  The starter pack for 2. includes:
	- Data loading code for re-ranking models (which provides an iterator returning padded, ready-to-go batches of word ids)
	- Neural re-ranking model skeletons (helper code)
	- Evaluation code (core_metrics.py)

* We recommend that you use an Anaconda environment and install the requirements.txt with pip

* You should use an Nvidia GPU, if you don't have one locally, you can use Google Colab (https://colab.research.google.com/), which offers a free GPU in the cloud -> use the train.ipynb for this instead of train.py

## Provided Data

### Part 1

* Raw judgement FiRA data from 2022 with multiple judgements per judgement pair, simple baseline aggregated qrels (labels), textual data for optional use of improved label generation & analysis

**raw judgements format**

``id	relevanceLevel	relevanceCharacterRanges	durationUsedToJudgeMs	judgedAtUnixTS	documentId	queryId	userId``

**qrels format**

``queryid hardcoded-Q0 documentid relevance-grade``

Text-based files are always: ``id text`` (separated by a tab)

### Part 2

* Provided data: AllenNLP vocabulary (collection specific, in two sizes: use the _10 = min of 10 occurrences in the collection if you have memory problems with the _5), train triples, evaluation tuples (validation & test) with 2.000 queries each and the top 40 BM25 results per query, relevance judgments (qrels, one file covering both validation & test)

* Download a pre-trained glove embedding from: http://nlp.stanford.edu/data/glove.42B.300d.zip

### Part 3

* Question answering tuples and answers from FiRA annotations in 2021 (covering natural queries from MSMARCO)

# Notes

* The assignment text is purposefully kept simple. If you see two diverging paths of doing things, pick the one that makes the most sense for you and write a short note in the report about the choice and why you decided on one way of doing it. In any case, you will receive the points for the part. 

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

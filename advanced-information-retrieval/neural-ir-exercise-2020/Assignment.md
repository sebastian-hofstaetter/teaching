In this exercise your group is implementing neural network re-ranking models and gain an understanding by developing a novel visualization in their context. We provide you with some starter code & prepare a subset of the MS MARCO dataset so that starting for you is easier. We are utilizing PyTorch and AllenNLP (Python 3). PyTorch allows you to be very flexible in the definition of the training process (good for our pairwise training) & debug the code in a natural way - in your IDE the same as any other Python code. The models and further references are described in the first neural IR lecture. 

# Goals

1. Implement 3 Neural Architectures to perform re-ranking (KNRM, MatchPyramid, Conv-KNRM) - **70 points**
	
   	- Implement: the model classes **(15 points each - 45 total)**
       - Show that you understood what happens by adding comments to difficult parts of the model (what tensor dimensions represent, what gets summed up, etc..)
    - Implement: training process & result evaluation **(15 points)**
        - Including early stopping based on the validation set
	- Evaluate: Compute a test set evaluation at the end
		- MS-MARCO Sparse Labels **5 points**
		- FiRA fine-grained Labels **5 points**

2. Visualize results or inner-workings of the models or something interesting based on the FiRA annotations - **20 points**

	- First step: Decide what to visualize and how - tell a story (The only things not allowed are plain line & bar chart plots without a good reason for them)
	- Implement code to prepare the data for it and create the visualizations
	- For FiRA annotations especially, try to find some visualization using text.
	
3. Write a report - **10 points**

	- Detail problems & their solutions you encountered
	- Your evaluation results (validation & test)
    - How fast the models are in ms/sample (average / training + evaluation, add your GPU/CPU model)
	- Tell the story of the visualization with output of the visualization
		
# Bonus Points

* Finding and fixing bugs in the materials we provide (this exercise is new, so there might be bugs in the starter code) - **10 points each (20 max)**

* Try out more variants or other re-ranking models - **up to 15 points**

* Do something cool on top of the exercise goals - **up to 15 points**

# Notes:

*  The starter pack for 1. includes:
	- Data loading code (which provides an iterator returning padded, ready-to-go batches of word ids)
	- Neural IR model skeletons (helper code)
	- Evaluation code (msmarco_eval.py)

* We recommend that you use an Anaconda environment and install the latest PyTorch via conda (https://pytorch.org/get-started/locally/) and AllenNLP subsequently via pip (https://github.com/allenai/allennlp) - you are free to use any modules provided by AllenNLP inside the models 

* You should use an Nvidia GPU, if you don't have one locally, you can use Google Colab (https://colab.research.google.com/), which offers a free GPU in the cloud -> use the train.ipynb for this instead of train.py

* Provided data: AllenNLP vocabulary (collection specific, in two sizes: use the _10 = min of 10 occurrences in the collection if you have memory problems with the _5), train triples, evaluation tuples (validation & test) with 2.000 queries each and the top 40 BM25 results per query, relevance judgments (qrels, one file covering both validation & test)

* Download a pre-trained glove embedding from: http://nlp.stanford.edu/data/glove.42B.300d.zip

* Make use of the GPU (.cuda() - see the PyTorch documentation for details). The provided data should keep the models under 5GB of GPU RAM when using the ADAM optimizer

* A few hints and tricks:
    - Use the Adam optimizer (keep in mind that it adds 2x the memory consumption for the model parameters)
    - The iterators do not guarantee a fixed batch size (the last one will probably be smaller)
	- The batch tensors also have no fixed size, the max values in the readers are just to cap outliers (the size will be based on the biggest sample in the batch (per tensor) and the others padded with 0 to the same length)
    - The kernel models need masking after the kernels -> the padding 0's will become non-zero, because of the kernel (but when summed up again will distort the output) 
    - Feel free to change the default parameters in the starter code (in general they should give you good results)

* KNRM & MatchPyramid should reach ~ 0.19/0.20 MRR@10 and Conv-KNRM should go up to ~ 0.22 MRR@10 (The values here are lower than the leaderboard/what is shown in the lecture, because we only provide you with a subset of the training & evaluation data)

# References

* AIR Lecture 4 & 6
* KNRM: Chenyan Xiong, Zhuyun Dai, Jamie Callan, Zhiyuan Liu, and Russell Power. 2017. End-to-End Neural Ad-hoc Ranking with Kernel Pooling. In Proc. of SIGIR.
* Conv-KNRM: Zhuyun Dai, Chenyan Xiong, Jamie Callan, and Zhiyuan Liu. 2018. Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search. In Proc. of WSDM
* MatchPyramid: Liang Pang, Yanyan Lan, Jiafeng Guo, Jun Xu, Shengxian Wan, and Xueqi Cheng. 2016. Text Matching as Image Recognition. In Proc of. AAAI.

* If you google the models, you will find some implementations on Github - most of them in tensorflow and some of them with a different set of errors in them (we don't discourage you from looking at them, but keep in mind that they might not be correct and you have to show that you understood the models in the exercise interview)
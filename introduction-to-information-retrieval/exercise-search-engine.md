# Exercise 1 - Search Engine

## Goals (Total of 100 points)

The main goal of this exercise is to create, evaluate, and describe your own search engine. To test it we use a subset of Wikipedia articles. More details are in the notes below. The individual goals are the following:

- Implement efficient text processing code to create an inverted index from the given Wikipedia articles, including: tokenization, stemming, stop words **(40 points for evaluation-set, (only 10 points for dev-set))**

- Implement save/load to disk code for your inverted index, including a format to save the index **(10 points)**

- Implement query processing: (re-use tokenization, stemming) including: TF-IDF and BM25 scoring methods **(20 points)**
	- Exploration mode: allow free text input and return ranked documents in full text
	- Evaluation mode: read a topic file and output ranked document ids in the evaluation format 
- Evaluate the given INEX Wiki test collection **(10 points only for evaluation-set)**
	- Report MAP, NDCG@10, Precision@10, Recall@10 for TF-IDF and BM25 configurations (using trec_eval)
- Write a report:  **(20 points)**
	- Describe your architecture briefly 
	- Note 2 or more interesting details of your work (coming up with the interesting things is part of the exercise)
	- List your performance numbers (indexing time, query time for evaluation queries)
	- List the results + observations between the two scoring methods
	
- Exercise Interview: Necessary to get the other points 
	
## Bonus points

- Find a bug in any of Sebastian's open source projects or a factual error in the lecture slides, notes **(5 points)**
- Fix the bug via a GitHub pull request or improve the lecture slides, notes **(10 points)**

- Do something extra -  that we haven't thought of - but it was interesting to you **(10 points)** Note: this extra thing can be some extraordinary efficient code (including, but not limited to: SIMD, GPU acceleration) or some visual analysis of the index, results, etc.. or some inspection of the test collection… Or implement additional features… This category is very relaxed.
	
- Give detailed, constructive 😊 feedback for lectures 1 + 2 and this exercise **(5 points)** 


## Notes

- Teams of 2 people 

- Hand-in: Put everything in the GitHub repository, we will collect your work from there. Use the repository to save your work regularly - don't commit all your changes once before the deadline.  

- You are free to use any programming language/stack you want. But you are not allowed to use high-level libraries (Lucene or any other search engine implementation, which do the work for you). Low-level libraries and methods (string transformations, array buffers, hash-maps, trees, memory mapped files, etc..) are ok. A third-party stemming method is ok (Recommended: Porter https://tartarus.org/martin/PorterStemmer/).
- You are free to choose which architecture and methods you like best to achieve the goals (e.g. type of inverted index creation process, how you structure your data structures, etc..)

- You don't have to store the document text, just save the original filename + start/end bytes and load the text for the exploration query. However you might want to consider saving the preprocessed text (how it is added to the inverted index) per document to explore your tokenization etc.. which gives you an accurate representation of your indexed data.
- You don't have to create a graphic user interface, just use a console based app, that can switch between indexing and the two query modes

- Feel free to look at code samples, search engine implementations, algorithm descriptions - but do not copy code. If you find some good references and ideas add them to the report.
- We use GitHub Classroom for this assignment. You need a GitHub account and register a team with your partner in GitHub. We provide a template repository, which is cloned by GitHub for you and administrated by us. Please commit and push regularly and often.
- Please use the provided directory and file structure for the source code, results and report. 

- Report:

	- Please write the report in markdown (in the report file provided in the repository), so that it looks good and structured on GitHub. 
	- Report length: The instructor should be able to read it in 10 to 15 minutes. We will read it just before the exercise interview to get an overview of your solution.
	- Report language: German or English, whatever is more comfortable for you


## Information on dataset & evaluation

### Dataset: 
- 2 folders with Wikipedia articles (dev-set + evaluation-set)
	- Both contain files with multiple randomly sorted Wikipedia articles in xml format (the body of the articles only contain plaintext)
	- Dev-set: contains only relevant documents, you may use it to develop your implementation and evaluation pipeline, but not for the actual evaluation 
	- Evaluation-set: contains the dev-set and additional documents (total of 2.2 GB)
- "topics" file - with given evaluation queries and their description
- "qrel" file - which contains human relevance judgements for queries in the topics file = allows us to measure the performance of a search engine and compare different configurations (like TF-IDF and BM25)

### Evaluation:

1. Index the articles in evaluation-set
2. Parse the topics file and get query id and query string (you may choose the title only or also text from the description or narrative)
3. For every query search in your index for relevant documents and output 100 documents per topic in the following format (per line):

	   {topic-id} Q0 {document-id} {rank} {score} {run-name} 
	
	- **topic-id** is the id per query from the topics file
	- **document-id** is an identifier for the Wikipedia article
	- **Q0** is just a legacy hardcoded string
	- **rank** is an integer indicating the rank of the doc in the sorted list (normally, this should be ascending from the ﬁrst line to the last line)
	- **score** the similarity score calculated by the BM25 or TF-IDF method
	- **run-name** a name you give to your experiment (free for you to choose)
	
	Put the result in a text file in the "retrieval_results/" folder in your GitHub repository (the name should roughly describe the configuration like: tfidf_title_only.txt)
	
4. Use the trec_eval utility (https://github.com/usnistgov/trec_eval) to compute the performance metrics with the following command:

	```bash
    trec_eval  -m map -m ndcg_cut.10 -m P.10 -m recall.10 path/to/qrel_file path/to/output_from_3 
	```

	Or (if you want to compare more steps)
	
	```bash
    trec_eval  -m map -m ndcg_cut -m P -m recall path/to/qrel_file path/to/output_from_3 
	```
	
	The trec_eval utility has to be compiled with make (on windows: use the Linux subsystem (e.g. bash/ubuntu) for windows)
	
	Put the results for different configurations in separate files (same name as the result file from 3.) in the "eval_results/" folder in your GitHub repository

## References

Wikipedia evaluation collection from: https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/software/inex/

Queries and Ratings (Adhoc Track 2010 Topics) from: https://inex.mmci.uni-saarland.de/data/documentcollection.html
# Lecture 10 - Dense Retrieval and Knowledge Distillation

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Hi and welcome everyone. Today's lecture is about dense retrieval and knowledge distillation. My name is Sebastian, and as always, if you have any questions, please feel free to contact me. Let's get started.  

*16.97 seconds*

### **2** Today 
Today we're talking about a couple of different things, so this lecture is jam packed with information. First, we're going to look at dense retrieval, what that means, which types of model, specifically the BERT DOT retrieval model, is used for that, and how approximate nearest neighbor search brings everything together, and allows for a search engine. Then the second part is about knowledge distillation and specifically about our work in the last couple of months, where we looked at cross architecture, distillation losses such as Margin-MSE or our recently proposed *TAS-Balanced with dual supervision* training approach for dense retrieval models. And then finally, we're going to look at how we can analyze dense retrieval models and what those analyses mean for the future development of those models.

*68.03 seconds*

### **3** Neural Methods for IR Beyond Re-ranking 
Why are we talking about dense retrieval? We want neural methods for information retrieval to work beyond re-ranking models that we saw in previous lectures. We want to bring neural advances in the first stage phase of the ranking pipeline to remove the bottleneck of an inverted index with BM25. And today we look at dense retrieval as one such alternative. Although before we start, I want to acknowledge there are a lot of other very cool techniques how neural approaches can improve first stage retrieval, and some of them are listed here. So in very popular approaches like Doc2query, which actually does document expansion with potential query texts that would semantically match a document. And here you then still use the inverted index with the expanded documents and use a traditional BM25 function. But because the document text has more words in it, it gives you better results. Another option is DeepCT, which assigns term weights based on BERT output during indexing. And there retrieval is still done with and inverted index and BM25 but DeepCT basically changes the statistics of the words in the index. And then COIL, which is a recent approach that fuses contextual vectors into an inverted index structure which gives us faster lookup of semantic matches.  

*111.13 seconds*

### **4** 
Now, let's focus on dense retrieval. The potential future of search. This question is still open, but today I hope we can answer some of the sub-questions that come with it. 

*13.7 seconds*

### **5** Recall: Neural Re-Ranking Models 
Let's recall how a neural re-ranking model is deployed inside the pipeline. So re-rankers change the ranking of a pre-selected list of results. And the interface is you give the ranking model a query text and a document text and then it can do fine-grained interactions between the two sequences. But this all hinges upon this first stage ranker, which in this picture is a traditional inverted index with BM25 statistical matching model. 

*40.4 seconds*

### **6** Dense Retrieval & Re-Ranking 
So now dense retrieval replaces this traditional first stage index and we're using a neural encoder and then nearest neighbor vector index to be able to match the encoded query to all potential passages in our collection. It can be used as part for larger pipeline as visualized here. So here in this case we still have a second stage re-ranking model, but the first stage is now a nearest neighbor index. 

*38.36 seconds*

### **7** Standalone Dense Retrieval 
With a dense retrieval model, if we have a model that is effective enough for our goals and we'll show in a bit how dense can be achieved. Then we can also use it as a standalone solution and just remove the second stage re-ranker, which of course is much faster and comes with less complexity if we remove it. And so our search system reduces only to this first stage retriever. 

*36.18 seconds*

### **8** Dense Retrieval Lifecycle 
Let's look at the lifecycle of such a dense retrieval model. Here, we have three major phases. The first is training. So here we create training batches and we train our dense retrieval BERT DOT model with a lot of repetitions for different training batches. And once we have either a pre trained model from step one, or we train it ourselves from scratch, we can use this trained model in the indexing phase. Here we take every passage in our collection and encode it via the model, which gives us a single vector per passage as the output. And those vectors are indexed in a nearest neighbor index that is potentially an approximated nearest neighbor index, and once this index is built we can take the full index and use it in phase three, the searching. So now a user gives us a query with a sequence of words and those sequence of words are also encoded with our dense retrieval model too similar to the passages get a single vector per query. And this single vector is now matched in the nearest neighbor index against all other vectors. To find the, let's say top ten or top 1000 closest vectors to that query vector, which then get mapped back to the document ID and we can provide the search interface based on that.  

*114.61 seconds*

### **9** BERT DOT Model 
I have mentioned the BERT DOT model before in the previous slides. But now let's look at the heart of dense retrieval, which is the BERT DOT model. And we call it BERT DOT so that we can distinguish it from BERT CAT for example. But others are calling it Siamese BERT, twin BERT or BERT tower or so. Here, passages and queries are compressed into a single vector. They are completely independently encoded, and this means that we completely move the expensive computation of passage encoding into the indexing phase, and we at query time only need to encode a single query once to get the query vector, which is very fast. The relevance is then scored with a dot product between the two CLS vectors, although there are also variants where we use cosine similarities, for example, and this set up allows an easy use of an approximate nearest neighbor vector index. Because this simple dot product or cosine similarity operation is supported by basically every single nearest neighbor library. 

*96.41 seconds*

### **10** BERT DOT 
Formalized is the BERT DOT model as follows. So we have a separate encoding of query and passage. This can be done independently and we feed each sequence with the CLS token into the BERT model and then we pool the first vector output, so the CLS vector gets taken and we ignore the rest of the output. And then in the matching, which can be done outside the model, so inside a nearest neighbor library for example, we have a simple dot product between the query encoded vector and the passage encoded vector. Optionally we can also compress those two vectors with a single linear layer in the encoding stage to reduce the dimensionality of the model after the BERT computation is finished. 

*60.05 seconds*

### **11** Training BERT DOT  
We can train this BERT DOT model like any other re-ranking model and it works OK. But we can do more and considering that we don't actually want to use it as a re-ranking model we have a much stronger requirement on recall and that the model does not return some completely random results. So here there are two main paths for improving the training of BERT DOT. So the first one is because we don't need to compute the score inside the model, we can use the vector output, which allows us to operate on this vector output by reusing the encoding work and we don't need to encode passages multiple times during training if we want to score them with different passages. And then a second, we can use the indexing capabilities of the model also during training and not only after, and with repeated indexing we can use those index results to, for example, sample more passages.  

*80.97 seconds*

### **12** Sampling Strategies: In-Batch Negatives 
To the first point, reusing already encoded passages, we look at a sampling strategy called in-batch negatives. So because we don't need pairwise concatenation in our training batches, we can use our training batches in a so called individual view. That allows us to encode queries, relevant passages and non-relevant passage in individual pieces. So now the training workflow can be adapted. We conduct a forward pass separately for query and the two passages, and then in the pairwise aggregation we can do it row-wise, right? So as before, if we look at one query, enter two associated passages, but we can also crisscross between other samples in the batch or create a loss with different negative samples per query and those in-batch negatives allow us to reuse the encoded vectors and simply repeat the dot product computation, which is very fast, so we can have more signals for the gradient descent to be used once we do the backward pass.  

*96.53 seconds*

### **13** Sampling Strategies: ANCE 
Then another sampling strategy that uses the indexing capabilities of the model is the so called a ANCE approach which stands for approximate nearest neighbor negative contrastive learning, and typically we would train up BERT DOT with BM25 negative passages. But those negative passages are not actually representative of how the model sees negative passages. So ANCE uses the model itself to create new negative samples, and for that it runs continuous indexing on certain checkpoints and inference for every training query. And then once the indexing and inference which runs in parallel to the model training is done, they swap the index and repeat the same process over again, but with new negative samples now coming from the already trained index. Here the indexing costs multiple GPU hours every 10,000 batches which does not scale well and takes a lot of compute to do. 

*88.35 seconds*

### **14** Nearest Neighbor Search 
Now let's look at nearest neighbor search and let's now assume we have a trained dense retrieval model and we can use it to encode every passage in our collection. Then we save passages in an approximate nearest neighbor index. During search, we encode the query on the fly. But now let's look at what that means to search inside the nearest neighbor vector index. 

*33.37 seconds*

### **15** NN Search: GPU Brute-Force 
If we do a GPU-based brute-force search, meaning we compute a dot product for, let's say 9 million vectors in the MSMARCO collection, and we want the top 1000 passages for those nine million. We need to do 9 million dot products which basically is 1 very big matrix multiplication with 700-dimensional vectors. But GPUs are made for this, right? So as long as the vectors fit into the GPU memory, we get an astonishing under 70 millisecond latency for single queries, and when we scale up the batch size, we have an enormous non-linear scale factor, so we can use batch size of up to 2000 queries at a time, which then only takes 5 seconds to compute. Which means it's very easy to run inference on all MSMARCO training queries in a matter of minutes. Using a CPU to do the same task, it would take one second per single query and CPUs of course don't scale as well as GPUs in this case. 

*92.67 seconds*

### **16** Approximate NN Search 
To be able to use CPUs for nearest neighbor search, we can turn to approximate nearest neighbor search. And fortunately, the act of nearest neighbor searching is very common and broadly used technique in machine learning. And therefore a lot of research has been put into making better techniques and faster libraries to speed up search for sub-linear results. In very popular library that I want to mention is FAISS, which offers many algorithms baked into it, such as brute-force or inverted lists based on clusters or each HNSW, which are all approximate nearest neighbor search algorithms. FAISS supports both CPU and GPU, and we have a very good experience using it. And I also want to mention that approximated nearest neighbor search is again another tradeoff between latency and effectiveness that we add to our system. Because if you do approximate nearest neighbor search, you're not guaranteed anymore to find the exact nearest neighbors that you would with a brute-force search, but it is necessary if you want to conduct low latency CPU serving. 

*90.02 seconds*

### **17** Production Support 
And another important point for serving such a dense retrieval model is production support. Luckily, dense retrieval is gaining more and more support in various production systems, and here again the HuggingFace library and its model hub is essential for us to have a common format to share those models. And then search engines must incorporate the indexing and orchestration of the query encoding as well as the nearest neighbor search themselves and some projects that already do that are Vespa.ai, and Pyserini and Vespa is a full fledged search engine which provides deep integration of dense retrieval models with common search functionality such as applying additional features, applying additional filtering on properties or so. Because we want to filter during search and not after search because if we filter after search we might be left with empty result lists or needing to go back to the nearest neighbor search to get us more results back. On the other hand, Pyserini is a project focused on reproducing and comparing as many dense retrieval models as possible. And it is based on Lucene, so it also allows you to easily do hybrid searches combining BM25 and dense retrieval. 

*111.69 seconds*

### **18** Other Uses for the BERT DOT Model 
The BERT DOT model or the abstract architecture of BERT DOT can be used for all sorts of semantic comparisons. So, for example, can compare sentences, passages, or documents to each other, all compressed into a single vector. A lot of recommendation models also work with the same idea in mind. And here a library that I want to mention is the popular S-BERT or sentence Transformer library which provides a lot of models and scenarios built-in based on HuggingFace and they provide us with pre trained sentence similarity models for example. So you can just take a sentence similarity model from HuggingFace, use the library to encode your text and you will get good sentence similarity output. And then finally adaptions on this architecture also make it possible to do multimodal comparison. For example if we want to encode images and text into the same vector space, we can just swap out one part of the BERT DOT model, for, let's say a pre-trained computer vision encoder, and then train this model to move both vectors in the same vector space.  

*95.81 seconds*

### **19** 
Knowledge distillation is a major improvement for the training of BERT DOT. It can be said that it's an almost free lunch, it comes with very little downside if any, and provides us with better trained models. 

*17.9 seconds*

### **20** The Idea of Knowledge Distillation  
So the idea of knowledge distillation is the following: most training data is noisy and not very fine granular. This is true for a lot of machine learning tasks, but especially in the case of MSMARCO. We only have one labeled relevant passage per query, and we might have a lot of false negatives where positives are not labeled as such. And in knowledge distillation we have two types of models. First is a teacher, which is a powerful but slow model which can provide labels for us and for that we can re-run inference on all training samples after the model is fully trained. Now we have fine-grained scores for all examples. And then second, the student model where we can use those new teacher labels to train our more efficient student model and we try to find better weights that produce a higher quality output overall and the teacher helps guide the student to achieve that than the student would on its own. 

*83.31 seconds*

### **21** Different Levels of Supervision 
There are a lot of different levels of supervision that you could apply, so we could use the final output score of the model or class distribution or anything like that as supervision signal and this makes it possible to operate independent of architectures. It also makes it possible to ensemble different teacher models together in one big teacher. But it also leaves information out intermediate information that we could use as well. And if we look at intermediate results for some or all layers as additional supervision signals, we could, for example use activations or attention distributions. From a bigger, better trained teacher, model onto a smaller model, but this basically locks us into a certain architecture with potentially similar or same parameter settings, but it also provides us much more supervision signals than just using a final score. 

*79.64 seconds*

### **22** DistilBERT 
One very important general purpose distillation model that I want to mention right now is the so-called DistilBERT model, which is a distilled and smaller version of the general purpose BERT model. It shares the same vocabulary and also shares the same general purpose nature, so it's ready to be fine tuned by us. It only has six layers and because it is trained with knowledge distillation, it retains something like 97% of effectiveness on a lot of tasks. And when we use DistilBERT as a base model in IR empirically we show that it works very well. We consistently get good results for different architectures including BERT DOT and if we apply knowledge distillation ourselves during the IR training, using DistilBERT as a student, there is hardly a difference to larger BERT instances. 

*68.99 seconds*

### **23** Distillation in IR 
How do we set up distillation in information retrieval? The training setup remains the same, so we're still operating on a query, a relevant passage and a non-assembled non-relevant passage. We first in our first step need to train the teacher model on a binary loss because we don't have fine grained labels yet. And then in the second step we run teacher inference. For example, with the BERT CAT model, because that's the the most powerful model that we generally have available. Then we run this teacher influence and gather all the fine-grained exact scores for all training samples and store them. So now because we stored them, we can reuse those scores without having to re-run inference again. And finally, we can use the result store to train the student models with specialized losses. 

*71.33 seconds*

### **24** Cross-Architecture Issues 
One issue that we notice when looking at different architectures in information retrieval is, that starting from the most powerful architecture which is BERT CAT, we observe that different models converge to vastly different output score ranges during their training because of their architecture. This works because the only thing that matters is a relative difference between each passage that is then is used to sort passages. And what we found is directly optimizing the models to fit the scoring range of BERT CAT while still working does not lead to optimal results. 

*63.99 seconds*

### **25** Margin-MSE Loss 
Therefore, we proposed a new way of knowledge distillation in IR, which we called margin MSE loss. And here we propose to optimize the margin between relevant and non-relevant passages per query so that the scoring range does not matter, but only the relative differences do and we don't need to change anything about the architecture to use this margin-MSE loss, which is defined as the mean squared error of the margin of the student versus the margin of teacher. And the margin of the teacher is used as label in this case and only the student model participates in the gradient updates. We can mix and match different neural ranking models with that loss, so we have absolutely no assumption about an special architecture or attention distributions or so on, and we can precompute the teacher scores exactly once and reuse them, or even share them with the community so that the community can also reuse them. 

*80.04 seconds*

### **26** Cross-Architecture Distillation Re-Ranking 
In our experiments, we looked at what happens with a single teacher and what happens with an ensemble of different BERT CAT based models. And what we found is a single teacher already improves most efficient models most of the time and then adding a teacher ensemble mostly brings additional improvements. So ColBERT, PreTTR and DistilBERT CAT are even better when trained on a teacher ensemble than a single BERT-Base CAT model. And here on the left side, you can see your plot that compares the effectiveness on the Y axis, in this case nDCG at 10 on the track deep learning collection with the query latency of models, when they use passage caching. And here important to note that the query latency in milliseconds is a log scale. So, the models are actually much further apart than they appear here. And with margin MSE, we only shift the effectiveness up, but we don't change the query latency because we don't change the architectures, only the weights of the models.  

*91.37 seconds*

### **27** Margin-MSE Dense Retrieval 
When applying margin-MSE in the dense retrieval task coming from a full collection, we show it's also possible and produces pretty good results, even though we do nothing special for dense retrieval training at the moment. So we train BERT DOT in this case as a re-ranking model, but then use it as a full collection dense retrieval model, and I would say the results are quite respectable and close to much more complex and costly training approaches. And we found that DistilBERT is even better than BERT-Base in this case, although the differences are small and again, the teacher ensemble produces better models than a single teacher.  

*54.15 seconds*

### **28** Dual Supervision 
Because our margin MSE loss doesn't really care which type the teacher model is in which type the student model is, we can also use it in a so-called dual supervision approach specifically. Now we look at training dense retrieval models only. So we only want to train a BERT-DOT model and we are doing it in the following way: we apply more teachers. In addition to our pairwise teacher, which is the BERT CAT model, we also use a trained ColBERT model for in-batch negative signals, and ColBERT can also use the individual view of a training batch which allows us to easily compute in-batch negatives without a big overhead that BERT CAT would give us. So here in this figure you can see we start off with the same training batch, but now for the BERT CAT model we concatenated query and passage, potentially offline, so we can precompute those scores, and then for the individual view, we run both ColBERT and the BERT-DOT model, which gives us results for every single combination of query on all passages. Then for the pairwise loss we used the BERT CAT scores and the scores for the pairs coming from BERT DOT and for the in-batch negative loss we can combine the ColBERT scores with the BERT DOT scores either by using margin MSE on pairs of scores, or we could also use something like KLDiv or LambdaRank and look at the in-batch negatives as a list-wise loss.   

*129.15 seconds*

### **29** Improve In-Batch Negatives 
So now we have a good supervision signal for in batch negatives, but the problem is if we create random in-batch negatives those pairs provide very little information; no relevance signal beyond it has nothing to do with each other, so models don't gain much by doing random in-batch negatives. But our idea was, if in-batch negatives come from queries in a single batch, we can change the way we sample queries. So for that we use a baseline model to cluster query representations. So for example for MSMARCO where we have 400,000 training queries we created 2000 clusters with roughly 200 queries each. And here on the right side you can see a t-SNE plot with two dimensions, which reduces the 700 dimensional query vectors down to two. And you can see when we randomly sample a couple of queries out of randomly sampled clusters, that queries per cluster have something to do with each other, they are topically related. And between clusters we have little relations, so you can see if we randomly sample queries that they easily have nothing to do with each other.  

*108.48 seconds*

### **30** Topic Aware Sampling 
Now, we utilize our query clustering in our novel technique called Topic Aware sampling, or short TAS. And instead of randomly sampling queries out of the whole query pool, we randomly sample topically related queries out of a single cluster per batch. And this technique does not add any additional cost to training process, so we don't need to re-cluster queries, we don't need to reindex anything, we can continue the training as before, but we just change the composition of our batches.  

*49.23 seconds*

### **31** TAS-Balanced 
And in addition to the topic aware sampling, we also propose in the same paper an additional balanced sampling of the pairs per query. So we still operate with our pool of passages per query which have a relevant passage and sample negative passages, and from our pairwise teacher we know the margin between those two pairs. And we observed that we have many more pairs having large margin, so the passages have less to do with each other. And to overcome this skewed distribution, we downsample large margin pairs by binning in a histogram-like fashion the pairs based on there margin and then sampling uniformly through each bin, and this makes the pairwise training harder without discarding any training data. Here in this figure you can now see our full approach for TAS-Balanced. We do random queries out of a single cluster and then for sampling the passages for those queries we look at each margin and the distribution of the margins to get a uniform distribution of margins in a batch. 

*99.33 seconds*

### **32** TAS-Balanced Results 
And this training technique, TAS-Balanced, improves with different teacher types. So here you can see our main ablation study looking at different sampling strategies and different teacher types. And we show that the best results come with dual supervision and TAS-Balanced. Here we have especially strong improvements on the recall at 1000 values, both for TREC as well as the MSMARCO-DEV Collection. 

*37.83 seconds*

### **33** TAS-Balanced Results 
And then the important comparison how does TAS-balanced now compare with other dense retrieval training approaches? And happily, we can say that TAS-balanced represents the current state of the art in dense retrieval training, especially if we also look at the batch size, which defines a lot of the hardware requirements needed for training, so a larger batch size means much more GPU hardware, but our TAS-balanced method already works well with a batch size as low as 32, and we also looked at different larger batch sizes here at the bottom of the table and you can see that yes, if we increase the batch size, the MSMARCO-DEV results go up, but the densely judged TREC deep learning results largely stay the same or are just noisily distributed.  

*76.58 seconds*

### **34** TAS-Balanced Recall Results 
And then let's focus a bit more on the recall results, because task balance is such a good recaller. Previously the densely judged track deep learning query set was very challenging for the recall or dense retrieval models. So here BM25 still had the upper hand, especially compared to standalone or a margin-MSE trained dense retriever where the recall stays roughly equal to BM25. Here in the plot on the right side on the Y axis you have to recall value and on the X axis you have this recall value at different cutoffs. So how many passages do we look at to compute the recall? If, for example, we want to use 500 passages in a re-ranking stage, how good would the recall be? That shows the the maximum potential that the re-ranker could achieve at that point X and we can see that the line in red, which is our TAS-balanced trained BERT DOT model is the first dense retrieve IR to significantly outperform BM25 in the great dotted line and also outperform the docT5query recall in the black dotted line. And then when we combine in a hybrid fashion, the results of task balance and docT5query we get even more of an improvement showing that the sparse and the dense retrieval still benefit from each other. And we observe this and every cutoff and every binarization point of the track deep learning collection. 

*128.87 seconds*

### **35** 
Looking at the overall results, we now want to focus a bit more on analyzing those dense retrieval models and find out on a more fine-grained level what works and what doesn't work.  

*13.51 seconds*

### **36** BEIR Zero-Shot Benchmark 
First up, we want to highlight the BEIR Zero-Shot benchmark, because ultimately what we want to achieve with dense retrieval models is that they are plug-and-play usable. Just as the traditional inverted index and BM25 paradigm and this is referred to as Zero-Shot transfer because in most cases we don't have large training data. We need to transfer the model from the MSMARCO training data for example to the target collection without the model ever seeing what the target collection looks like. And zero-shot is harder than in domain evaluation, because now we solely tests generalization capabilities instead of a mix of memorization, and generalization. So the BEIR benchmark brings many IR collections into a single format, and in includes a framework to run HuggingFace model hub models on all those collections. So now we can evaluate a large number of collections at the same time, and the paper showed that many dense retrieval models struggle in Zero-Shot transfer learning and that BM25 is more consistent. 

*98.08 seconds*

### **37** Why do DR Models Struggle on Zero-Shot? 
So the question now becomes, why do dense retrieval models struggle on Zero-Shot transfer learning. And the answer sadly is not a clear one and there are many nuances to this problem, so possible explanations include where are not limited to generalization. OK, so if we say that dense retrieval models don't generalize to other query distributions that would be pretty bad, because it would say, well, they don't work and we have to go back to the drawing board and try to make them better. That's one potential solution, then the second one is that they training data we're using, such as a MSMARCO contains a lot of specific quirks about the training, collection and to models learn to overfit on those quirks, such as the position biased we looked at in an earlier lecture. And here in this case, we would need adaptions to training data or more or other training data to create dense retrieval models that are robust on zero-shot transfer. And then the third question that is very interesting, and we also touched upon in the evaluation lecture and the test collection lecture, is pool bias, so many older and especially smaller collections are heavily biased towards BM25 results because BM25 was the main retrieval model used to gather candidates for the annotation campaigns. And here the ultimate solution to that is we need more annotations, which of course is very costly and very complicated to do on a lot of different collections. 

*128.43 seconds*

### **38** TAS-Balanced Zero-Shot Results  
Our test balance paper came out roughly at the same time as the BEIR benchmark paper, so it was not yet part of the discussion of the BEIR paper, but the organizers of BEIR took our public HuggingFace checkpoint and ran it through their framework already. And we are happy to answer the question "how good is TAS-Balanced in the Zero-Shot transfer learning?". And the answer is it's very good. It's the first dense retrieval model to have more wins than losses against BM25 as well as any other dense retriever. And it currently sits atop the leaderboard in the BEIR benchmark. 

*52.43 seconds*

### **39** Open Science = Fast Science 
Those results are a great external validation for our dense retrieval model training, and now I want to take a step back and just look at the sheer amount of open tools and platforms are used by multiple groups to make those task balanced at BEIR benchmark results possible. So we use a huge collection of libraries for training those models based on Python, PyTorch, FAISS, Numpy, etc. Then we have this whole ecosystem from HuggingFace, which includes the Transformers library, the Model Hub including free and open hosting of checkpoints. Then we have code hosting on GitHub and a range of tools for dissemination of our results which is archived for the paper, Twitter to publicize this, and GoogleDocs to host the leaderboard. And then the in my opinion, incredible fact about this whole endeavor is the speed of it all. So the timeline is as follows: Our test balance paper was accepted to SIGIR, where the conference is in July and after the acceptance notification, we pushed our preprint to archive in April, including a public model checkpoint on HuggingFace. The BEIR preprint was also published concurrently in April, and then the BEIR team integrated our checkpoint and ran all benchmarks by mid-May, which is even now two months before the conference where the paper is officially presented. Which is just an incredible speed and shows how fast the community as a whole based on a lot of open source tools can create a lot of value in a very short amount of time, and I think that's just incredible. 

*142.83 seconds*

### **40** Lexical Matching 
With that, let's switch gears and look at some of our in depth analysis of dense retrieval models. And first up is lexical matching because dense retrieval models can potentially retrieve any passage out of the collection. We operate in unconstrained vector space, so there are no guardrails from stopping us to retrieve completely irrelevant and random content. So this is a novel failure type that BM25 does not have because with BM25 we need at least one lexical overlap so that a passage is returned from the inverted index. And we now analyze if this problem is something that we have to look out for in dense retrieval models, and for that we looked at the large MSMARCO-DEV set with 50,000 queries. And we gathered all queries for a standalone trained and a TAS-Balanced trained dense retriever that have an overlap between query and passage tokens of the first retrieved passage of either zero, meaning absolutely no overlap or only one subword. And then we went ahead and did an annotation campaign to re-annotate those P@1 passages to get the precision at one. And our findings are in this table right here. We find that absolutely no overlap is very, very uncommon. And even query passage pairs that only have a single token overlap in relative percentage points of all the queries is still in very minor problem. And with TAS-Balanced, those values go down again. If we look at how good those results are, we see that, yes, for the standalone trained query, the precision at one goes down quite a bit for those queries. But then for TAS-Balanced, the precision at one while being lower than average over the deficit as a whole is very negligible, especially considering that the total amount of queries that are in this set is very small. So we can conclude that this is only a small problem, an especially for TAS-Balanced, the problem is even smaller. 

*188.59 seconds*

### **41** Can We Retrieve Unknown Acronyms? 
Our next analysis is a bit more on the qualitative side. So here we want to know if we can retrieve unknown acronyms. And because we want to be a bit manner, we are using information retrieval conference websites as our target passages right here and then queries about information retrieval conferences as our queries. And we have to say that those are completely artificial examples showing lexical matching abilities. We took the first paragraph from each conference website. They are not in the MSMARCO collection and the acronyms our not in the MSMARCO training data. And you can see here if we search for the conference acronyms, we can see that the passage of the respective conference is always matched higher than the other passages. For SIGIR, ECIR and ICTIR. Very interesting here is that SIGIR matches higher than the ICTIR passage on the query SIGIR because SIGIR appears exactly once in both of those passages, but the model somehow figured out that first passage is more about SIGIR than the ICTIR passage. Then also interesting, if we misspell SIGIR, it still matches to the correct passage higher than the others. And also if for example we use a query like *retrieval conference Europe*, we match the ECIR conference higher than the other two, even though in the passage itself only the word *European* appears once. And for the query search engine conference, all three passages roughly get the same score, which is also a good sign. 

*138.46 seconds*

### **42** Are There Still Cases where BM25 is Better? 
And then we want to know if we can still find cases where BM25 is vastly better than a dense retrieval model, even though overall dense retrieval is better, we could still have sub-regions of our queries distribution where BM25 is better. And the answer is yes, there are cases, although its increasingly hard to find them. So what we did is we used sparse judgements as a filter for selecting which queries should be annotated in the fine grained way. So we selected a set of queries where on the sparse judgements BM25 has a perfect MRR score, and the dense retrieval model has the worst MRR score of 0, meaning it did not get the one annotated passage in the top 10 results. And then we went ahead and re-annotated the first retrieved passage from the dense retrieval models to see if the problem is in the sparse judgements, or if those queries are actually harder for dense retrieval than BM25. And it turns out OK, we start off, the number of queries is very small, so we have 2% to 1% of all test queries we looked at match to the case BM25 is vastly better in the sparse judgements, but then in our annotations we found that those results are mostly an artefact of sparse judgements, while for the standalone model appeared one goes down a bit, as well as for TAS-Balanced, most of the queries that are highly relevant are still returned on the first position already, so that means we need to keep looking further to find other ways of trying to get queries where BM25 is significantly and systematically better. 

*148.38 seconds*

### **43** Our Analysis Take Away 
What do we take away from this analysis? Well trained dense retrieval models are very good at lexical matching. And in my opinion, it is a common myth that they are bad at it. But I also want to say that well trained is the key here, because random initialization of course would break it, and this statement applies only to MSMARCO and in domain training, and we might have a bigger problem here in the Zero-Shot transfer scenarios, this needs to be explored further. And then we also showed that unknown acronyms get picked up by the model quite well, even though we only looked at it in a qualitative way. On MSMARCO, dense retrieval models are already so good, that it's quite hard to find large slices of the query distribution where BM25 is much better. And in our analysis we couldn't find systematic patterns that match that. 

*66.94 seconds*

### **44** Summary: Dense Retrieval & Knowledge Distillation 
To conclude this whole talk about dense retrieval and knowledge distillation, dense retrieval is a promising direction for the future of search, and today we already showed a lot of advancements the community made in the last year alone. Knowledge distillation is a key to improve dense retrieval models, whereas strong teacher model helps the dense retrieval model to train better. And then finally, our topic aware sampling approach represents the current state of the art in dense retrieval training, both for in domain results as well as Zero-Shot transfer learning results. 

*50.04 seconds*

### **45** Thank You  
Thank you very much for your attention and I hope you enjoyed this talk and see you next time. Bye. 

*9.87 seconds*

### Stats
Average talking time: 79.1779888888889

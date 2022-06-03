# Lecture 8 - Transformer Contextualized Re-Ranking

*Automatic closed captions generated with the Azure Speech API*

### **1** Transformer Contextualized Re-Ranking
Welcome to another lecture on neural advances in information retrieval. Today, we're going to talk about Transformer Contextualized Re-Ranking. My name is Sebastian and if you have any questions please feel free to contact me. 

*17.72 seconds*

### **2** Today 
Today, on one hand, we're going to talk about re-ranking with BERT in the concatenated query and document way of using BERT. As well as some other techniques for improving upon this as well as long document handling. And then the second major point today is going to be about efficient Transformer-based models. So how can we take the advances from Transformers or BERT and make them faster and more efficient to use? So here we're going to take a look at a few models that try to improve upon the BERT efficiency as well as the Transformer-Kernel family of models that we created to overcome the efficiency problems of BERT. 

*60.51 seconds*

### **3** Now We Reached the State-of-the-art 
We finally did it. Today we are talking about the state of the art. To get here, we had to take a couple of different lectures to build up our foundational knowledge, but today we're going to talk about the best models that are currently out there. And with currently, I mean that most of the contents of this lecture did not exist in the beginning of 2018. And by July of this year we'll probably have a new state of the art technique in our field. So we are very fast moving. There are a lot of open questions that need to be answered and we try to answer a few in this lecture. The general direction is if you have more computation you get better results and this is made possible with better hardware, stronger GPUs and more data.  

*61.39 seconds*

### **4** Context of this Lecture 
We are studying ad-hoc re-ranking models and of course there are many other applications of Transformers in information retrieval. Just to name a few, we can do document or query expansion, we can look at full QA pipelines, conversational search, where the query history is taken into account, dense retrieval or knowledge graph-based search. All of these approaches use Transformers or can use Transformers. Today we're focusing on the efficiency/effectiveness tradeoff from different angles. So how fast or slow is a model compared to the result quality? And of course we can also study many other aspects such as social biases, domain adaption or multilingual models, and many more. 

*61.63 seconds*

### **5** Web Search with BERT 
To get a sense of the impact that BERT has on search here we look at two blog posts by the two major search engine companies, Google and Microsoft, who both started to use BERT re-ranker in production in late 2019. So now if you search on Bing or Google, you will use a BERT re-ranker pretty much all of the time.  

*35.85 seconds*

### **6** Re-Ranking with BERT
Look how BERT can be used as a re-ranking model to drastically improve our search result quality. 

*8.66 seconds*

### **7** Recall: BERT - Workflow 
Let's recall the BERT Workflow from our Transformers lecture. So someone with lots of compute power and time to pre-train a large model creates this pre-trained model for us with Masked Language Modelling (MASK) and Next Sentence Prediction in the case of BERT. OK. So then once this model is trained we can download it for example via the HuggingFace Model Hub and then fine-tune the BERT model on our task based on the pre-trained weights. 

*41.33 seconds*

### **8** BERT Re-Ranking: BERTcat 
The straightforward way of applying BERT in re-ranking is to use the two sequence workflow that BERT provides for us. So we concatenate the query and the passage sequences before importing the complete sequence into the BERT model. And we do so by using the specialized tokens of BERT. We are calling this architecture BERTcat short for conCATenated, but others are calling it monoBERT, vanilla Bert, or just simply BERT, which gets pretty confusing if you only call this architecture BERT. So you concatenate query and passages and then the output from BERT gives you a vector for each token. In our case, we just pool the CLS token, meaning we take only the first output token and ignore the rest, and then we predict the score of relevance with a single linear layer that reduces the 700 dimensional CLS vector to a single dimension: one floating point score. And this needs to be repeated for every passage to get a score for every passage.

*91.93 seconds*

### **9** BERTCAT 
We can formalize BERTcat as follows. We start off by concatenating the special tokens, the query and passage sequences, we run them through BERT and pool the CLS token representation. And then, for the scoring, we run the representation through a single linear layer, which basically multiplies the CLS representation with a weight matrix which starts uninitialized. This we have to train through our retrieval training, but of course we can also and should also train and fine-tune the BERT weights as well. And here we have the choice which BERT model to take and how many layers and how many dimensions this model has? 

*56.77 seconds*

### **10** The Impact of BERTcat
The impact of BERTcat as first shown by Nogueira and Cho jumpstarted the current waive of neural IR. It works pretty awesome out of the box because concatenating the two sequences, fit's BERT's workflow and pre-training regime perfectly. And as long as you have enough time and enough compute it trains quite easily. There are major jumps in effectiveness across collections and domains, but of course it also comes at the cost of performance and virtually no interpretability because everything happens inside the BERT black box. And you have to run BERT for every passage in your re-ranking depth so larger BERT models translate to slight effectiveness gains, but of course at high efficiency costs. 

*65.67 seconds*

### **11** So how good is BERTcat ? 
As a single example of how good is BERTcat architecture actually is, here we show some results from our latest knowledge distillation paper where we use BERTcat as teacher models. And here we really want to get the best possible performance out of them. So for the MSMARCO-Passage dataset we have an MRR@10 of .19 for BM25, but re-ranking those BM25 results with, for example, ALBERT-Large, which is a larger BERT model, we get up to .38. BERT basically doubles the result quality in this case, similar for a MSMARCO-Document, we start off with an MRR of .25 for BM25 and we get up to .38 even with a DistilBERT with a 2000 token depth. Similar results have been shown on the blind evaluation TREC-DL 2019 and 2020. As well as on the recently published TripClick large training collection. So those results are all in the large training data regime, but we also have works that look at the effectiveness of BERT with smaller training data and zero-shot evaluation. And there we also see substantial gains using the BERT re-ranking approach. 

*107.58 seconds*

### **12** The Mono-Duo Pattern 
One way of improving even upon this already great result from BERTcat is the Mono-Duo Pattern where Mono-Duo stands for a multi-stage process which first uses a Mono model similar to BERTcat where we score a single passage and a single query at a time which can be for example done for the top 1000 results. And then the Duo model kicks in which actually is scores triples of two passages at a time which needs to be done for every pair combination in our re-ranking depth. But this improves on the single model stage and it's very good for in leaderboard settings and showing the maximum achievable performance, especially if we use the very very large T5 Transformer-based language model which is even larger than typical BERT models as our base model. But this also means that this whole architecture becomes very slow. 

*74.4 seconds*

### **13** BERTcat for Longer Documents 
And another limitation of BERT is that it is capped at a maximum of 512 input tokens for query and document combined. So a simple solution if you want to re-rank longer documents is just to take 500 tokens in total and ignore the rest of the document, which works surprisingly well, especially for the MSMARCO-Document Connection which contains web documents where at the beginning of a document is already very indicative of the document as a whole, but it might not work well in other domains where documents are really long or in a contain a variety of topics at different document depths. So still a simple but a simple solution that works on full documents is to create a sliding window over the document. So you create a window and each time when you slide the window across the document, you run BERT. And then you can just take the maximum window score as the document score. Here we can also make smaller sliding windows, which might be useful in the user interface (UI)to then highlight this most relevant passage as snippet. 

*91.88 seconds*

### **14** BERTcat In-Efficiency 
And now let's take a look at the inefficiency of BERTcat. So here in this plot on the x-axis we can see the latency, so how long does a query take in milliseconds? And then, on the y-axis, we have the effectiveness. Here in this case the MRR@10 of MSMARCO-Passage. And the colored dots on the lower left corner, those are all our basic re-ranking models we saw in an earlier lecture. They still improve a bit over BM25, and each architecture improves a bit over the next architecture, but then you can see that BERT, this lone gray dot in the upper right corner, is much better in terms of effectiveness, but it is also much, much slower, clocking in at almost 2 seconds latency versus a couple of milliseconds for the more basic re-ranking models. Here we also have to take into account if you require 2 seconds for each re-ranking run that you block your GPU for two seconds at a time, which increases your infrastructure cost if you have a lot of users. So this is not very feasible to put into production. 

*107.35 seconds*

### **15** Efficient Transformer-based Models
Let's take a look at efficient Transformer-based models. We want to adapt BERT or start from scratch with the goal to create the most efficient and most effective model at the same time. 

*17.11 seconds*

### **16** Achieving Efficiency 
There are multiple paths to achieve efficiency, and in our case we look at how to reduce query latency. But of course, you can also think about the full lifecycle efficiency when you include training, indexing and retrieval steps. Very simple way is to just reduce the model size, because of course smaller models run faster. Because they do less computation. This is, although only possible until a certain threshold, after which the quality reduces drastically, so we can use something like DistilBERT which reduces the BERT layers and almost keeps the same effectiveness as larger work models, but reducing layers further than decreases our effectiveness very strongly. Another second way is to change the architecture and move computation away from the query time. For example, move as much computation as possible into the indexing time where we can pre-compute passage representations, so then at query time they become a simple lookup. And for that we need a lightweight aggregation between pre-computed representations and on-the-fly computed query representations. 

*99.17 seconds*

### **17** Splitting BERT for Efficiency 
One way of achieving that is to split BERT for efficiency because the BERTcat model needs to be run for every passage in our re-ranking depth at query time, which is bad for query latency as we showed in the earliest slides. So now, we observed that most of the computational time for BERTcat is spent on passage encoding because we have to do it for every passage, the passages are longer. And if we can move this passage encoding to the indexing phase we can save a lot of time during our query evaluation. We also only need to encode each passage exactly one time and can then reuse this representation. So the more users we have, the more efficient the whole system becomes. And at query time we would only need to encode the query ones with very few words, so this is quite fast actually even on a CPU. Multiple approaches have been proposed lately to basically split BERT, then glue the representations back together and we're going to take a look at some of them now. 

*80.99 seconds*

### **18** Splitting BERT: PreTTR 
The first splitted BERT model is the PreTTR model, which splits BERT at a certain hyperparameter threshold. So you can say OK, I want to split BERT in the middle at layers 6 of 12 or I want to split BERT at layer 11 of 12 and only concatenated my representations at the last layer. So as you can see in this overview plot, we have an independent input of query and passage at the beginning which runs through the first n BERT layers and then we can do the offline computation of passages. And at query time we look up those half encoded BERT representations and concatenate them with our half-encoded query representations in our final BERT layers to, again, create a CLS vector output that we can pull similar to the BERTcat architecture with a single linear layer to create the score. The pro of this is the quality is almost the same as BERTcat. It still gives us quite a low query latency, especially if we only concatenated the last layer or so, but now we need to store all our passages. So here McAvaney et al. actually proposed to compress the passage representations as well. 

*109.97 seconds*

### **19** PreTTR 
We can formalize the PreTTR architecture as follows. So first, in our first part we have an independent computation of query and passage representations, creating half encoded vector sequences, and in the second part we then concatenate them, run them through the rest of the BERT layers, and use a single linear layer to create our output score. Similar to BERTcat, this layer starts uninitialized and has to be trained as part of the re-ranking procedure, but of course in this case especially we have to train also every single BERT layer to get good results. 

*48.39 seconds*

### **20** Splitting BERT: ColBERT 
A similar approach is the ColBERT Model, which also splits BERT, but in this case uses the final representations that come out of all BERT layers before the interaction. So here as you can see, the passages can be fully computed by BERT and are offline computable as well as indexible and then at query time we only have to run the query through BERT, get our per term output, and now the ColBERT model does not use the CLS token from BERT, but it actually creates sort of a matchmatrix between every single query and every single passage term representation. And then uses a simple maxpooling for the document dimension and sum for the query dimension to create the final output score, which is a very lightweight aggregation. It offers very fast query latency but still needs to save all passage term vectors which creates a huge storage cost. 

*81.25 seconds*

### **21** ColBERT 
We can formalize ColBERT as follows. We have our encoding stage where we encode query and passage representations separately. The passages can be done at indexing time and then in the aggregation phase we create a matchmatrix of dot products between every query and every passage representation. We then select the largest score per document term and then sum up those scores for each query term to get our final output score. Optionally, we can compress query and passage encoding's with a single linear layer, for example, to save a lot of the storage requirements that ColBERT would otherwise have in this full setting. And this is very efficient at query time. 

*64.27 seconds*

### **22** TK: Transformer-Kernel Ranking 
Another option is to start from scratch and remove BERT completely. And that's what we did in the TK or Transformer-Kernel model. Here our desired properties were we want to have a lightweight model, it should be interpretable and it should be effective. So therefore we proposed TK, which combines Transformers with kernel-pooling that we saw in a previous lecture. It shows very strong results compared to the basic IR specific models and it also shows state of the art model results for a time budget constrained environment in the query time re-ranking phase. It uses Transformer blocks as a contextualization layer, but those Transformers are uninitialized and we create a hybrid contextualization by merging context jewel and non contextual vectors. Our goal was to limit the number of Transformer layers because each additional layer takes considerable amounts of time to compute. 

*79.99 seconds*

### **23** TK: Transformer-Kernel Ranking 
Conceptually, the TK architecture looks as follows. We again have a query and document as our input, and then we contextualized those two representations independent from each other using word embeddings, Transformers, and the weighted sum. Then after our contextualized encoding, we have query and document representations. For each term, we create a term by term interaction matrix and then use kernel-pooling as a way to aggregate our interaction results. 

*39.08 seconds*

### **24** TK 
The TK architecture can be formalized as follows. We start off with our encoding layer, which uses Transformers and does the encoding independently, so we again can pre-compute our document representations either for storage or for analysis. Then in the matching phase we use cosine similarity for each term by term interaction. This creates a single match-matrix. And then we use kernel-pooling, which is basically the same or very similar to KNRM that counts the match-matrix interactions and then those interactions are summed up and weighted in the final scoring layer. 

*54.68 seconds*

### **25** Designing for a Time-Budget 
And we found that there is, of course large influence on how you employ the model for this efficiency/effectiveness tradeoff. So the main outside factor is of course how many documents to re-rank. If you re-rank fewer documents you will be faster. Even with GPU parallelization, there is an almost linear way of looking at this. So faster models can re-rank more documents in the same time as slower ones. And if you set out a time-budget and say OK, I want to spend this many milliseconds to re-rank something, it allows us to simultaneously evaluate the effectiveness and efficiency in a more realistic setting.  

*68.48 seconds*

### **26** TK on a Time Budget 
The results of the time budget analysis look like that. So if you have more documents in the same time, you get better results. But of course after a while BERT takes over because it has overall better effectiveness results, but at a much higher cost of efficiency. And you can see that three times for MRR Recall and nDCG, our model in red here outperforms traditional retrieval models and is better than BERT in a the given time budget of up to 150, or in the case of recall, where more documents are important, more important over 600 milliseconds. 

*63.25 seconds*

### **27** Comparing the Models 
So let's compare the models that we just saw in the previous slides. We have a lot of different options and dimensions on which to compare the different models, the first of which of course is quality. So how effective are those models with different evaluation measures? Another way of comparing them is by query latency storage and GPU memory requirements or capabilities in general. So here we can see that if we start off with the BERTcat model with an effectiveness of 1 and we scale the other models accordingly, that BERTcat offers a query latency of roughly one second and it needs 10 gigabytes of GPU memory. On the other hand, the ColBERT model basically achieves the same effectiveness with 97%, only takes 28 milliseconds per query and only requires 3.4 gigabytes of GPU memory. The PreTTR model has a similar effectiveness, but in the setting that we evaluated in this paper, it has higher query latency and lower efficiency, and it also needs 10 gigabytes of GPU memory. And then the final model that we looked at is TK, which offers a bit lower effectiveness, so it's not as good as the other two models, but the query latency is even lower in the pre-computed option, and it uses drastically less GPU memory. 

*112.22 seconds*

### **28** Understanding TK  
So now how can we actually understand what our model is doing. I said before we incorporated a bottleneck of a single cosine match-matrix and this cosine match-matrix can be very nicely visualized and so we created a demo application called the Neural IR Explorer to actually visualize what the model sees in terms of the similarity of contextualized words. So you can browse around and you can kind of get a feeling for how a test collection looks, how a model works, what it sees and what works and what doesn't work, and we see it kind of as a complementing thing to metric-based evaluation. 

*65.2 seconds*

### **29** Demo
Right, so here is the demo. You can see the MsMARCO collection as you know from the exercise has a lot of queries. So to overcome this complex field we clustered the queries based on their contextualized representations. Another side output of our model if you will. Then for each cluster you can sort the queries by how much better the neural model improves over the initial BM25 baseline, you can filter out queries to find specific words and then you can look at how a single query looks in terms of a result list compared to a couple of documents at the same time. So you can for example, only focus on some words that you're interested in, and then you can look up why the model scored some documents higher than others. And of course, you can also compare two documents side-by-side to get a more detailed look at how this whole thing works, and if you like to look at shiny colorful things, please check out the life demo that's available at neural-ir-explorer.ec.tuwien.ac.at. 

*109.6 seconds*

### **30** TK-Sparse with Contextualized Stopwords 
And to this end we propose TK-Sparse, an adaption of the TK passage ranking model, and this adoption is possible because the TK model, which uses Transformers and kernels to score query and passage pairs contextualizes out of the box the query and passage independent from each other, which means that we can offload the passage vector computation, the contextualization, which is the most expensive part of the model, we can offload that to the index time. And again we just have to save every single vector, and our contextualized stopwords actually learn a sparsity module after the Transformer contextualization to remove words that the model seems not going to be relevant to any given query. And this sparsity module is trained end-to-end with an L1 norm augmented loss function that minimizes a sparsity vector that decides that if it's zero, it will remove the given term at this position. And it removes the time from the ranking decision of the kernels, which then allows us to not save those removed terms.  

*110.43 seconds*

### **31** Effectiveness Results (MSMARCO-Passage) 
The effectiveness results on the MsMARCO-Passage collection are quite interesting if we look at the baseline where we remove common stopword lists before actually imputing the passage text into the model, and if we remove those stopwords before the neural model, we can see an overall drop in performance across both sparse and dense judgments from the deep learning TREC.

*42.28 seconds*

### **32** Effectiveness Results (MSMARCO-Passage) 
When we then look at different variants of our novel TK-Sparse model, on the other hand, we can see that we remove something around 20 to 40% of the words that we now don't need to save in our index, and still improve the effectiveness of the model overall.  

*31.44 seconds*

### **33** Stopword Analysis 
Furthermore, we also investigated which words are removed by our end-to-end trained stopword component. And for that, here in this plot we compare the most common stopwords that appear most often in our collection where every occurrence of a term is removed and we compare it with different variants of our TK-Sparse model. And the contextualized stopwords decide on an occurrence basis if we should remove terms. And here you can see: yes we have an overlap in remove terms, but also substantial differences in the terms removed. 

*54.71 seconds*

### **34** TKL: Transformer-Kernel for Long Documents 
Right, so with our TK model, we now have a very efficient and effective model for passage re-ranking. But in information retrieval we also encounter long documents that have thousands of tokens where we can't just or we shouldn't just cut off the first 200 and be done with it. Most models currently out there don't contain a notion of region importance or a way of handling the long document text well. So the current best approach that has been done is we split a document and we score individual sentences or paragraphs and then externally outside of the model we combine those scores, which is not an end-to-end approach. And we proposed an extension for TK for long documents called TKL, which just recently has been accepted to a conference, where we look at an end-to-end approach that you can actually train based on full documents at once. 

*93.83 seconds*

### **35** TKL: Transformer-Kernel for Long Documents 
And the schematic overview looks like that. So you have again a query and a document, but this time our document is quite long and to overcome the quadratic runtime requirement of Transformers, we actually contextualize in overlapping chunks to come to a linear runtime requirement. The document, again is contextualized independent from the query, so we could pre-compute document representations to speed up the production system. And then we apply the same term interaction and kernels. But for the scoring part, we actually don't want to look at the document as a whole, but we want to scan across the document and find regions of importance. Because as it turns out, most of the time only a couple of regions, multiple sentences, etc are relevant to the query, and so we want to detect those regions, both for interpretability and to get a better scoring representation out of it. So we do that by scanning the saturation function. The KNRM style log-based saturation function or log-like saturation function across the documents fine the top relevance regions and scored them together in a weighted way.  

*121.95 seconds*

### **36** TKL: Why long documents? 
So why do we actually need long documents? We evaluated our model on different document lengths, and here again in red, you can see that only our TKL model consistently improves when presented with longer document input. And that's also actually the main idea behind exercise 1. We wanted to find out if the model was correct in this assumption, because right now we can only say yeah, the model works better, but do we actually know via human annotations that documents at the end also contain relevant information and how, thus the structure of relevant information look like. So that's what we did with exercise one. We annotated documents in a very fine-grained and thorough fashion, so now we can analyze all those things.  

*79.46 seconds*

### **37** TKL: Where is the relevance? 
And the TKL model also gives us a way to look at where it found the relevance. So in our configuration we said, OK, let's look at the top three relevant regions per document. And we found out that the occurrence pattern of those regions across the document length actually follows a soul called Zipfian-Distribution that represents a tailed distribution, so if you plot it via a log/log scale like here, it looks like a linear way. But in reality, if you wouldn't plot it in a log/log plot, it would very much look like a tailed distribution. And this relevance region detection could also be used as a snippet generation, right? So you can have better user interfaces. 

*76.32 seconds*

### **38** IDCM: A Hybrid Approach 
Building atop the findings of the TKL model, we recently introduced the IDCM model, which is a hybrid approach between kernel-pooling and BERT-based ranking for long document re-ranking. So here we combine slow and fast module in a single model to provide an intra document cascade. Here we start off with a query and along document separated in different passages or sliding windows. So first in our fast stage we use an efficient student model to select k passages which are then slowly scored by the effective teacher model or BERT to get our final and good score for each passage which is then aggregated in a very simple way to create a final document score. 

*64.51 seconds*

### **39** IDCM: Distilled Training 
The cool thing about the IDCM model is how we actually train the different modules. So here we proposed a three step training procedure. First we train our BERT model, the efficient teacher model on a passage task to get it primed up on the passage ranking. Then we train the same model in the document setting with the different distribution of the new document collection. As well as the passage score aggregation module and then to train our efficient student model that should do the selection of passages, we use knowledge distillation from the BERT model to the efficient student model without using any external labels for the passages. And we want the student model to select the same passages in the top spots as the BERT model so that we would get theoretically the same score once we score these passages with BERT. And we found that this is easier, the more relevant a document is, and it gets harder, the less relevant a document is for the efficient student model to select the right passages. 

*89.26 seconds*

### **40** IDCM: Improving Throughput 
Here you can see how we improve the throughput with different cascade settings. So the main setting in the IDCM model is how many passages we choose to score with the slow model. Because the fast model scores every passage of course to make the selection. And the more passages we score with the slow model, the better the results, but of course at the cost of efficiency. And here on the left you can see on the x-axis the effectiveness score in terms of MRR@10 and on the y-axis the throughput of how many documents per second can we re-rank from scratch. And you can see that with a setting of four to five passages out of up to 40, we can achieve almost the same effectiveness as with the full model, but at up to four times lower query latency as if we would run BERT on every single passage in the old BERT setting. 

*76.88 seconds*

### **41** More Resources  
If you want to know more about Transformer contextualized re-ranking there is a great survey buyer Jimmy Lin et al. where they look at most papers that have been published in the field in the last two years called "Pretrained Transformers for Text Ranking" as well as a pretty cool ECIR 2021 Tutorial that uses the Terrier academic search engine to give you a hands on tutorial of how to use that in different stages and how to use BERT and other re-ranking models on top of that. And of course if you want to keep up with new papers in the field, I can only recommend the archive list for the IR subfield, which produces around 10 new papers a day and it's definitely the fastest source to get information about preprints. 

*63.95 seconds*

### **42** Summary: Transformer Contextualized Re-Ranking 
This talk was packed with information and I want to only pick out a few major points to take away for you. So first the concatenated BERTcat opened a new era of information retrieval. Then second it does so because it provides enormous effectiveness jumps, although at the cost of speed. But we saw different approaches to make this effectiveness/cost tradeoff much better. One of those approaches is to use Transformers and Kernels in a way that leads to a good compromise between cost and effectiveness.  

*47.76 seconds*

### **43** Thank You  
With that I thank you very much for your attention and I hope you tune in next time. Bye. 

*6.59 seconds*

### Stats
Average talking time: 68.27207267441858

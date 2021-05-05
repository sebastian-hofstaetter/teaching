# Lecture 8 - Transformer Contextualized Re-Ranking

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Welcome to another lecture on neural advances in information retrieval. Today, we're going to talk about Transformer. Contextualized re ranking my name is Sebastian and if you have any questions please feel free to contact me. 

*17.72 seconds*

### **2** Today 
Today, on one hand, we're going to talk about re ranking with Bert in the concatenated query and document way of using Bert, as well as some other techniques for improving upon this as well as long document handling. And then the second major point today is going to be about efficient transformer based models. So how can we take the advances from Transformers? Or Bert and make them faster and more efficient to use. So here we're going to take a look at a few models that try to improve upon the Bert efficiency as well as. The Transformer kernel family of models that we created to overcome the efficiency problems of bird. 

*60.51 seconds*

### **3** Now We Reached the State-of-the-art 
We finally did it today. We are talking about the state of the art. To get here, we had to take a couple of different lectures to build up our foundational knowledge, but today we're going to talk about the best models that are currently out there and with currently, I mean that most of the contents of this lecture did not exist in the beginning of 2018. And by July of this year we probably have a new state of the art technique. In our field, so we are very fast moving. There are a lot of open questions that need to be answered and we try to answer a few in this lecture. The general direction is if you have more computation you get better results and this is made possible with better hardware, stronger cheap use and more data.  

*61.39 seconds*

### **4** Context of this Lecture 
We are studying at Hawk Re ranking models and of course there are many other applications of Transformers in information retrieval, just to name a few. We can do document or query expansion. We can look at full QA pipelines, conversational search, where the query history is taken into account, dense retrieval or knowledge graph based search. All of these approaches. Used Transformers or can use Transformers? Today we're focusing on the efficiency effectiveness tradeoff from different angles. So how fast or slow is a model compared to the result quality? And of course we can also study many other aspects such as social biases, domain adoption or multilingual models, and many more. 

*61.63 seconds*

### **5** Web Search with BERT 
To get a sense of the impact that Bert has on search here we look at two blog posts by the two major search engine companies, Google and Microsoft, who both started to use Bert rear anchors in production in late 2019. So now if you search on Bing or Google, you will use a Bert rear anchor. Pretty much all of the time.  

*35.85 seconds*

### **6** 
Look how bird can be used as a re ranking model to drastically improve our search result quality. 

*8.66 seconds*

### **7** Recall: BERT - Workflow 
Let's recall the Bert Workflow from our Transformers lecture. So someone with lots of compute power and time to pre train a large model creates this pre trained model for us with masked language modelling and next sentence prediction in the case of bird. OK. So then once this model is trained we can download it for example the hugging face Model Hub. And then fine tune the bird model on our task based on the pre trained weights. 

*41.33 seconds*

### **8** BERT Re-Ranking: BERTCAT 
The straightforward way of applying Bert in re ranking is to use the two sequence workflow that Bert provides for us. So we concatenating the query and the passage sequences before importing the complete sequence into the Bert model. And we do so by using the specialized tokens of Bert. We are calling this architecture BERTCAT short for concatenated, but others are calling it Mona Bird, Vanilla Bert, or just simply Bert, which gets pretty confusing if you only call this architecture work. So you concatenating query and passengers and then the output from bird. Gives you a vector for each token. In our case, we just pool the seal as token, meaning we take only the first output token and ignore the rest, and then we predict the score of relevance with a single linear layer that reduces the. 700 dimensional CLS vector to a single dimension. One floating point score and this needs to be repeated for every passage to get a score for every passage.  

*91.93 seconds*

### **9** BERTCAT 
We can formalize BERTCAT as follows. We start off by concatenating the special tokens. The query and passage sequences, we run them through bird and pool. The CLS token representation, and then. For the scoring, we run the representation through a single linear layer, which basically multiplies the CLS representation with a weight matrix which starts uninitialized. This we have to train through our retrieval training, but of course we can also and should also train and fine tune the bird weights as well. And here we have the choice. Which Bert model to take and how many layers and how many dimensions this model has? 

*56.77 seconds*

### **10** The Impact of BERTCAT 
The impact of BERTCAT as first shown by Nogueira and show jumpstarted the current waive of neural IR in works pre awesome out of the box because concatenating the two sequences, Fit's birds workflow and pre training regime perfectly. And as long as you have enough time and enough compute it trains quite easily. There are major jumps in effectiveness across collections and domains, but of course it also comes at the cost of performance and virtually no interpretability because everything happens inside the bird black box and you have to run Bert. For every passage in your re ranking depth so larger Bert models. Translate to slight effectiveness gains, but of course at high efficiency costs. 

*65.67 seconds*

### **11** So how good is BERTCAT ? 
As a single example of how good is BERTCAT architecture actually is, here we show some results from our latest knowledge distillation paper where we use BERTCAT as teacher models and here we really want to get the best possible performance out of them. So for the MSMARCO passage data set we have an Mr and 10 of .19. Four PM 25, but re ranking those being 25 results with, for example, Albert Large, which is a larger Bert model. We get up to .38. Bert basically doubles the result quality in this case, similar for a MSMARCO document. We start off with an Mr of .25 for beam 25 and we get up two point 38 even with a distilled bird with a 2000 token depth. Similar results have been shown on the blind evaluation track. Deep Learning Tracks 2019 and 2020. As well as on the recently published TripClick large training collection. So those results are all in the large training data regime, but we also have works that look at. The effectiveness of bird with smaller training data and 0 short evaluation. And there we also see substantial gains using the Bert Re ranking approach. 

*107.58 seconds*

### **12** The Mono-Duo Pattern 
One way of improving even upon this already great result from Burkett. Is the mono duo pattern where? Mono do stands for a multi stage process which first uses a mono model similar to Bert Cat where we score a single passage and a single query at a time which can be for example done for the top 1000 results and then the two model kicks in which actually is scores triples of two passages at a time which needs to be done for every pair combination. In our re ranking depth? But this improves on the single model stage and it's very good for in leaderboard settings and showing the maximum achievable performance, especially if we use the very very large T5 Transformer based language model which is even larger than typical Bert models as our base model. But this also means that this whole architecture becomes very slow. 

*74.4 seconds*

### **13** BERTCAT for Longer Documents 
And another limitation of bird. Is that it is capped at a maximum of 512 input tokens. For query and document combined, so a simple solution if you want to re rank longer documents is just to take. 500 tokens in total and ignore the rest of the document. Which works surprisingly well, especially for the Ms Market Document Connection which contains web documents where at the beginning of a document is already very indicative of the document as a whole, but it might not work well in other domains where documents are really long or in a contain a variety of topics at different document depths. So still a simple but. A simple solution that works on full documents is to create a sliding window over the document. So you create a window and each time. When you slide the window across the document, you run bird and then you can just take the maximum window score as the document score. Here we can also make smaller sliding windows, which might be useful in the user interface to then highlight this most relevant passage as snippet. 

*91.88 seconds*

### **14** BERTCAT In-Efficiency 
And now let's take a look at the inefficiency of BERTCAT. So here in this plot. On the X axis we can see the latency. So how long does it query take in milliseconds and then? On the Y axis, we have the effectiveness. Here in this case the MRR at 10 of Ms Marco passage. And the. Colored dots on the lower left corner. Those are all our basic re ranking models we saw in an earlier lecture. They still improve a bit over being 25, and each one improves a bit over. Each architecture improves a bit over the next architecture, but then you can see that Bert this known Gray dot in the upper right corner is much better in terms of effectiveness, but it is also much, much slower, clocking in at almost 2 seconds latency. Versus a couple of milliseconds for the more basic re ranking models. Here we also have to take into account if you require 2 seconds for each re ranking run that you block your GPU for two seconds at a time, which increases your infrastructure cost. If you have a lot of users so this is not very feasible to put into production. 

*107.35 seconds*

### **15** 
Let's take a look at efficient transformer based models. We want to adapt Bird or start from scratch with the goal to create the most efficient and most effective model at the same time. 

*17.11 seconds*

### **16** Achieving Efficiency 
There are multiple paths to achieve efficiency, and in our case we look at how to reduce query latency. But of course, you can also think about the full lifecycle efficiency when you include training, indexing and retrieval steps. Very simple way is to just reduce the model size, because of course smaller models run faster because they do less computation. This is, although only possible until a certain threshold, after which the quality reduces drastically, so we can use something like this still bird. Which reduces the birth layers and. Almost keeps the same effectiveness as larger work models, but reducing layers further than. Decreases our effectiveness very strongly. Another second way is to change the architecture and move computation away from the query time. For example, move as much computation as possible into the indexing time where we can precompute passage representations. So then at query time they become a simple look up, and for that we need a lightweight. Aggregation between precomputed representations and on the fly computed query representations. 

*99.17 seconds*

### **17** Splitting BERT for Efficiency 
One way of achieving that is to split Bert for efficiency because the BERTCAT model needs to be run. For every passage in our ranking depth at query time, which is bad for query latency as we showed in the earliest line, so now. We observed that most of the computational time for BERTCAT is spent on passage encoding because we have to do it for every passage to passengers are longer. And if we can move this passage encoding to the indexing phase week and save a lot of time during our query evaluation. We also only need to encode each passage exactly one time and can then reuse this representation. So the more users we have, the more efficient the whole system becomes. And at query time we would only need to encode. The query ones with very few words, so this is quite fast actually even on a CPU. Multiple approaches have been proposed lately to basically splitboard then glue the representations back together and we're going to take a look at some of them now. 

*80.99 seconds*

### **18** Splitting BERT: PreTTR 
The first splitted bird model is the PreTTR model, which. Splits Bert. At a certain hyperparameter threshold, so you can say OK, I want to split bird in the middle at layers 6 of 12. Or I want to split bird at layers 11 of 12 and only. Concatenated my. Representations at the last layer. So as you can see in this overview plot, we have an independent input of query and passage at the beginning. Which runs through the first end Bert layers and then we can do the offline computation of passages and it's query time. We look up those half encoded Bert representations and concatenated them with our half encoded query representations. In our final Bert layers to again create a CLS vector output that we can pull similar to the BERTCAT architecture with a single linear layer to create the score. The pro of this is the quality is almost the same as BERTCAT in still. Gives us quite a low query latency, especially if we only concatenated the last layer or so. But now we need to store all our passengers, so here mcaveney at all actually proposed to compress the passage representations as well. 

*109.97 seconds*

### **19** PreTTR 
We can formalize the PreTTR architecture as follows. So first, in our first part we have an independent computation of query, an passage representations, creating half encoded vector sequences, and in the second part we then concatenate them, run them through the rest of the Bert layers, and use a single linear layer to create our output score. Similar to Bert Cat, this layer starts uninitialized and has to be trained as part of the RE ranking procedure, but of course in this case especially we have to train also every single Bert layer to get good results. 

*48.39 seconds*

### **20** Splitting BERT: ColBERT 
A similar approach is the Colbert Model, which also splits bird, but in this case uses. Final representations that come out of all Bert layers before the interaction. So here as you can see, the passages can be fully computed by bird and are offline, computable as well as indexible and then. Add query time. We only have to run the query through Bert, get our per term output and Now the Colbert model does not use the CLS token from bird, but it actually creates sort of a match matrix between every single query and every single passage representation term representation. And then uses a simple Max pooling for the document dimension and sum for the query dimension to create the final output score, which is a very lightweight aggregation. It offers very fast query latency but still needs to save all passage term vectors which creates a huge storage cost. 

*81.25 seconds*

### **21** ColBERT 
We can formalize Colbert as follows. We have our encoding stage where we encode query and passage representations separately. The passages can be done at indexing time and then in the aggregation phase we create a match matrix of dot products between every query and every passage representation. We then select the largest. Score per document term and then some of those scores for each query term to get our final output score. Optionally, we can compress query and passage encoding's with a single linear layer, for example. To save alot of the storage requirements that Colbert would otherwise have in this full setting. And this is very efficient at query time. 

*64.27 seconds*

### **22** TK: Transformer-Kernel Ranking 
Another option is to start from scratch and. Remove bird completely and that's what we did in the TK or Transformer kernel model. Here our desired properties were we want to have a lightweight model. It should be interpretable and it should be effective. So therefore we proposed TK, which combines Transformers with Colonel pooling that we saw in a previous lecture. It shows very strong results compared to the basic IR specific models and in Arsenal also shows state of the art model results for a time budget constrained environment in the query time RE ranking phase. It uses transformer blocks as a contextualization layer, but those Transformers are uninitialized and we create, say, hybrid contextualization. By merging context jewel and non contextual vectors. Our goal was to limit the number of transformer layers because each additional layer takes considerable amounts of time to compute. 

*79.99 seconds*

### **23** TK: Transformer-Kernel Ranking 
Conceptually, the TK architecture looks as follows. We again have a query in document as our input, and then we contextualized those two representations independent from each other using word embeddings, Transformers, and the weighted sum. Then after our contextualized encoding, we have query and document representations for each term. We create a term by term interaction matrix and then use kernel pooling. As a way to aggregate our interaction results. 

*39.08 seconds*

### **24** TK 
The TK architecture can be formalized as follows. We start off with our encoding layer, which uses Transformers and thus the encoding independently, so we again can precompute our document representations either for storage or for analysis. Then in the matching phase we use cosine similarity for each term by term interaction. This creates a single match matrix. And then. We use kernel pooling, which is basically the same or very similar to KNRM that counts the match matrix. Directions, and then those interactions are summed up and weighted in the final scoring layer. 

*54.68 seconds*

### **25** Designing for a Time-Budget 
And we found. That there is, of course large influence on how you employ the model for this efficiency effectiveness tradeoff. So the main outside factor is of course how many documents to re rank. If you re rank. Fewer documents you will be faster. Even with GPU parallelization, there is an almost linear hum. Linear way of looking at this. So faster models can rerank more documents in the same time as slower ones. And if you set out a time budget and say OK, I want to spend this many milliseconds to rerank something. It allows us to simultaneously evaluate the effectiveness and efficiency in a more realistic setting.  

*68.48 seconds*

### **26** TK on a Time Budget 
The results of the time budget analysis look like that, so the more. If you have more documents in the same time. You get better results, but of course after awhile. Birth takes over because it has overall better effectiveness results, but at a much higher cost of efficiency. And you can see that both four or four three times for MRR recall and NCG. Our model in red here. Outperforms traditional retrieval models and is better. Did Burt in a the given time budget of up to 150, or in the case of recall, where more documents are important, more important over 600 milliseconds. 

*63.25 seconds*

### **27** Comparing the Models 
So let's compare the models that we just saw in the previous slides. We have a lot of different options and dimensions on which to compare the different models, the first of which of course is quality. So how effective are those models with different evaluation measures? Another way of comparing them is by query latency storage and GPU memory requirements or capabilities in general. So here we can see that. If we start off with the bird cat models with and effectiveness of 1 and we scale the other models accordingly, that Bird Cat offers a query latency of roughly one second and it needs 10 gigabytes of GPU memory. On the other hand, the Colbert model. Basically achieves the same effectiveness with 97%. Only takes 28 milliseconds upper query and only requires 3.4 gigabytes of GPU memory. The Predator model has a similar effectiveness, but in the. Setting that we evaluated in this paper, it has a lower as higher query latency and lower efficiency, and it also needs 10 gigabytes of GPU memory and then. The final model that we looked at is TK, which offers a bit lower effectiveness, so it's not as good as the other two models, but the query latency is even lower in the precomputed option, and it uses drastically less GPU memory. 

*112.22 seconds*

### **28** Understanding TK  
So now how can we actually understand what our model is doing and? I. Set before we incorporated a bottleneck of a single cosine match matrix and this cosine match matrix can be very nicely. Visualized and then we created a. Demo application called the Neural IR Explorer. To actually visualize what the model seas in terms of the similarity of contextualized words so you can browse around and you can kind of get a feeling for how it test collection looks, how it model works, and what it sees and what does, what works and what doesn't work, and we see it kind of as a complementing thing to metric. Based evaluation. 

*65.2 seconds*

### **29** 
Right, so here is the demo. You can see the Ms Marker collection as you know from the exercise has a lot of queries. So to overcome this complex field we clustered the queries based on their contextualized representations. Another side output of our model if you will. Then for each cluster you can sort the queries by how much better the neural model improves over the initial beam 25 baseline you can filter out queries to find specific words. And then you can look at how a single query looks in terms of a result list compared to a couple of documents. At the same time. So you can for example, only focused on some words that you're interested in, and then you can look up why the model scored documents. Some documents higher than others. And of course, you can also compare two documents side-by-side to get a more detailed look at how this whole thing works, and if you like to look at shiny colorful things, please check out the life demo that's available at neuralirexplorer.ec.tv dot AC dot. 

*109.6 seconds*

### **30** TK-Sparse with Contextualized Stopwords 
And to this end we propose TK sparse a, an adaption of the TK passage ranking model, and this adoption is possible because the TK model, which uses Transformers and kernels to score query and passage pairs contextualized is out of the box. The query and passage independent from each other, which means that we can. Offload the passage. Vector computation, the contextualization, which is the most expensive part of the model. We can offload that to the index time, and again we just have to save every single vector. And our contextualized stopwords actually learn a sparsity module after the transformer contextualization to remove words that are not. But the model theme is not going to be relevant to any given query. And this sparsity module is trained end to end with an L1 norm augmented loss function. That minimizes a sparsity vector that decides at if it's zero, it will remove. The given term at this position. And it removes the time from the ranking decision of the kernels, which then allows us to save to not save those removed terms.  

*110.43 seconds*

### **31** Effectiveness Results (MSMARCO-Passage) 
The effectiveness results on the Ms Marco Passage collection are quite interesting. If we look at the baseline where we remove stop, common stop word lists before actually imputing the passage text into the model, and if we remove those stop words before the neural model, we can see a an overall drop in performance. Across both sparse and dense judgments from the from the track deep learning track. When we then. 

*42.28 seconds*

### **32** Effectiveness Results (MSMARCO-Passage) 
Look at different variants of our novelty, K. Sparse model, on the other hand, we can see that we remove something around 20 to 40% of the words that we now don't need to save in our index, and still. Improve the effectiveness of the model overall.  

*31.44 seconds*

### **33** Stopword Analysis 
Furthermore, we also investigated which words are removed by our end to end trains, stopword component, and for that here in this plot we compare the most common stopwords that appear most often in our collection where every occurrence of a term is removed and we compare it with different variants of our TK sparse model. And the contextualized stopwords decide on an occurrence basis if we should remove terms. And here you can see. Yes, yes we have an overlap in remove terms, but also substantial differences in the terms removed. 

*54.71 seconds*

### **34** TKL: Transformer-Kernel for Long Documents 
Right, so with our TK model, we now have a very efficient and effective model for Passage Re ranking. But in information retrieval we also encounter long documents that have thousands of tokens where we can't just or we shouldn't. Just cut off the 1st 200 and be done with it. Uhm? An most models currently out there, they they don't contain any notion of region importance or. A way of handling the long document text well so the current best approach that has been done is we split a document and we score individual sentences or paragraphs and then externally outside of the model we combine those scores, which is not an end to end approach. And we proposed an extension for TK for long documents called TKL, which just recently has been accepted to a conference. Where we look at an end to end approach that you can actually. Train based on full documents at once. 

*93.83 seconds*

### **35** TKL: Transformer-Kernel for Long Documents 
And the schematic overview looks like that. So you have again a query and a document, but this time our document is quite long and to overcome the quadratic runtime requirement of Transformers, we actually contextualized in overlapping chunks. Home to only be. Do to come to a linear runtime requirement. The document, again is contextualized independent from the query, so we could precompute document representations to speed up production system. And then we apply the same term interaction and kernels. But for the scoring part, we actually don't want to. Look at the document as a whole, but we want to scan across the document and find regions of importance, because as it turns out, most of the time only a couple of regions, multiple sentences, etc are relevant to the query, and so we want to detect those regions, both for interpretability and. To get a better scoring representation out of it. So we do that. By scanning the saturation function. The key and RM style log base saturation function or long like saturation function across the documents fine. The top relevance regions and scored them together in a weighted way.  

*121.95 seconds*

### **36** TKL: Why long documents? 
So why do we actually need long documents? We evaluated our model. On different document lengths, and here again in red, you can see that only our model consistently. Our TKL model consistently improves when presented with longer document input. And that's also actually the main idea behind exercise. One we wanted to find out if the model was correct in this assumption, because right now we can only say yeah, the model works better, but. Do we actually? No, via human annotations that. Documents at. The end also contain relevant information and how, thus the structure of relevant information look like, so that's what we did with exercise one we annotated. Documents in a very fine grain and thorough fashion, so now we can analyze all those things.  

*79.46 seconds*

### **37** TKL: Where is the relevance? 
And the TKL model also gives us a way to look at where it found the relevance. So in our configuration we said, OK, let's look at the top three relevant regions per document. And we found out that the occurrence pattern of those regions across the document length actually follows a soul called zippy and distribution. That represents a tailed distribution, so if you plot it via a log log scale like here, it looks like a linear. I'm in linear away. But in reality, if you wouldn't plot it in a log log plot, it would very much look like a tailed distribution. And. This relevant region detection could also be used as a snippet generation, right? So you can have better user interfaces. 

*76.32 seconds*

### **38** IDCM: A Hybrid Approach 
Building atop the findings of the TKL model, we recently introduced the IDC M model, which is a hybrid approach between kernel pooling and bird based ranking for long Document re ranking. So here we combine slow and fast module in a single model to provide an intra document cascade. Here we start off with a query and along document separated in different passages. Or sliding windows. So first in our fast stage we use and efficient student model to select K passages which are then slowly scored by the effective teacher model or bird to get our final and good score for each passage which is then aggregated in a very simple way to create a final document score. 

*64.51 seconds*

### **39** IDCM: Distilled Training 
The cool thing about the ICM model is how we actually train the different modules. So here we proposed a three step training procedure. First we train our Bert model, the efficient teacher model on a passages task to get it primed up on the passage ranking. Then we train the same model in the document setting with the different distribution of the new document collection. As well as the passage score aggregation module and then to train our efficient student model that should do the selection of passengers. We use knowledge distillation from the Bert model to the efficient student model without using any external labels for the passengers, and we want the student model to select the same passages in the top spots. As the Bert model so that we would get theoretically the same score. Once we score these passengers with Burke and we found that this is easier, the more relevant a document is, and it gets harder, the less relevant a document is for the efficient student model to select the right passengers. 

*89.26 seconds*

### **40** IDCM: Improving Throughput 
Here you can see how we improve the throughput with different cascade settings. So the main setting in the IDC M model is how many passages we choose to score with the slow model because the fast model scores every passage. Of course to make the selection. And the more. Passages we score with the slow model, the better the results, but of course at the cost of efficiency. And here on the left you can see on the X axis the effectiveness score in terms of MRR at 10 and on the Y axis. The throughput of how many documents per second can we re rank from scratch? And you can see that with a setting of four to five passengers out of up to 40, we can. Achieve almost the same effectiveness as with the full model, but at up to four times lower query latency as if we would run Bert on every single passage in the old bird setting. 

*76.88 seconds*

### **41** More Resources  
If you want to know more about Transformer contextualized re ranking there is a great survey buyer Jimmy Lin at all where they look at most papers that have been published in the field in the last two years called Pretrained Transformers for text ranking. As well as in pretty cool ECIR R 2021 tutorial that uses the Terrier. Our academic search engine to give you a hands on tutorial of how to use that in different stages and how to use Bert and other re ranking models. On top of that and of course if you want to keep up with new papers in the field, I can only recommend the archive list for the IR subfield, which produces around 10 new papers today and it's definitely. The fastest source to get information about preprints. 

*63.95 seconds*

### **42** Summary: Transformer Contextualized Re-Ranking 
This talk was packed with information and I want to only pick out a few major points to take away for you. So first the concatenated BERTCAT opened a new era of information retrieval, then second. It does so because it provides enormous effectiveness jumps, although at the cost of speed. But we saw a different approaches to make this effectiveness cost tradeoff much better. One of those approaches is to use Transformers and kernels in a way that leads to a good compromise between cost and effectiveness.  

*47.76 seconds*

### **43** Thank You  
With that I thank you very much for your attention and I hope you tune in next time. Bye. 

*6.59 seconds*

### Stats
Average talking time: 68.27207267441858
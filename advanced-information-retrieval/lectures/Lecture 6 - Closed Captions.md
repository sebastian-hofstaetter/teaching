# Lecture 6 - Neural Re-Ranking

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Welcome back everyone. Today is today we're going to start talking about how to use neural networks for Information Retrieval re ranking. In my opinion, a very interesting topic. Well, because I research in this area, so let's get started. 

*25.5 seconds*

### **2** Today 
Today is actually the first of two lectures devoted to Neural re ranking. So today we're going to start with the basics, meaning the General Workflow of a neural re ranking system. And then we're going to take a look at little bit more simpler models that are not state of the art anymore, but they paved the way for Neural re ranking to become what it is now. And they also aren't the building blocks of your exercise 2 because there are much faster than state of the art models. And we have constraint. We all have constrained resources, so we are going to take a look at those more efficient models. 

*53.8 seconds*

### **3** Disclaimer: Content based Neural IR models 
Alright, so learning-to-rank systems in production use external features and not neural IR models. Well, that's what I told the students last year in April 2019.  

*23.01 seconds*

### **4** Disclaimer: Content based Neural IR models 
Um, wait a second. We now have April 2020. And a lot has changed since then.  

*11.38 seconds*

### **5** Behold … BERT! 
Neural IR models and especially variants of the BERT model are used in production by the two largest web search engine companies. Both Google and Microsoft announced in late 2019 that they started using neural re ranking systems as part of their ranking pipeline. There is a small disclaimer here, so both companies only put out block posts, Hum that are very marketing heavy and did not describe their systems in academic papers. But still, I think it is a great sign for the potential of Neural re ranking models that both of those companies are actually using them in production right now. 

*58.93 seconds*

### **6** Content based Neural IR models 
So we adapt our slide from before and now we can say that content-based neural re-ranking systems are used in production at least by the major search engine companies who have the resources and know how to currently deploy those systems. And of course they are used in addition to other signals like click counts, recency, authors and quality of the publications, personalization, etc. But it remains that deploying a re-ranker in production is a huge effort, so you have to spend a lot of time for performance and quality control, version management, changing indices, so there's a lot of stuff going on, multiple languages etc. And in this lecture. We are not looking at production systems, but we're looking at kind of the core of those systems and this is neural IR models, which in this lecture mean content based ad-hoc retrieval, where we only use the content of the query and the document to produce a ranking score. And of course they can be thought of as another signal for a learning to rank-system as a whole. 

*92.39 seconds*

### **7** Desired properties of Neural IR models* 
Desired properties of Neural IR models are pretty much in line with any other machine learning task. They should be effective, right? So they if they don't work then we just wouldn't use them. They need to be fast and scalable, so that's something that maybe differentiates the information retrieval task more from other machine learning tasks, is that if you do online re-ranking meaning you re-rank when the user searches for something, you basically have a time budget of 10 to 100 milliseconds for the full query. And potentially your indices are terabytes large. Of course this can vary a lot, right? So we have also we're also working with a lot of indices that are smaller for more specialized products and projects, so not everyone is working on a web search engine. And then, in my opinion, also a very important point is that those models should be interpretable because search engines filter information so they act as a gate to access certain informations for a large number or potentially large number of people. That means that the reasons for including or excluding certain results should be explainable and it should be analyzable, if you see that something is not right. 

*101.89 seconds*

### **8** Beware! Neural Networks fail silently … 
And with that, maybe let's do a small warning for the exercise, especially. So Neural Networks fail silently. Which means that you can train a neural network for hours and hours and it will produce some results, you will get some numbers out of it. But that doesn't mean that the network learns correctly, or that the network produces any meaningful outputs. And, even worse, sometimes you have small mistakes in there that just cost you a couple of percentage points of difference in your evaluation metrics, but those couple of percentage points could mean the difference between a new state of the art model and something that's kind of in the middle of everything. Right so the Neural Re-ranking field is very much a wild west, at the moment. We have a lot of new developments and an enormous amount of problems is not solved at the moment, and we don't know the answers to it and we don't have best practices yet. And kind of every research group has their own best practices. So it's a little bit different than other research fields that are more mature already, but I think that also makes the research field more exciting. So if you want to know more information about what could go wrong with neural network training and how to potentially fix that and find those errors, you might find that link quite handy. 

*116.22 seconds*

### **9** 
Let's talk about the workflow. So what is actually different for neural re ranking models in comparison to other machine learning tasks?

*13.42 seconds*

### **10** Neural IR Models 
Neural IR Models re-rank a candidate list of results. So they don't actually create their own index at the moment. We're going to look at indexing features and techniques in a later lecture. In this lecture, we only focus on the re-ranking part, which means that we still depend on an inverted index and the first stage ranker such as BM25 that we learned about in the crash course to produce our candidate list. Let's say we have a candidate list of 1000 potential documents that comes out of our first stage ranker here. Then for those thousand documents, we look up the full text in our full text storage. And together with the query text each document gets paired with the query text, so the neural IR model actually has the same interface as classical ranking methods. You get the score of a pair of one query and one document. And then after each pair is scored, you just sort the pairs by the return score to, for example, take the top ten documents out of the pool of 1000. And only present those top ten documents to the user. 

*102.49 seconds*

### **11** Context 
And we also have to be aware of how a search engine is interacted with. So search engines have a lot of users potentially right? And those users interact with the search engine in two ways. So one way the search engine delivers uses the query results. But users also deliver search engines click data. So they know where a user clicked on. And this can be saved in logs. And then overtime, those logs can be accumulated and used to improve the search engine. And here in this picture we see. we start with a couple of users that are sometimes happy and sometimes they're not happy, but overtime we use the clicks as a relevant signal so we can make more users happy down here. And the way we do is, is at certain intervals neural IR models are trained so called offline and are then swapped out with the actual running model. So it's not like the neural IR model continuously learns, but it is a loop that is somewhat asynchronous with the whole process, because of course, you want to do quality control before actually deploying a neural IR model. 

*100.31 seconds*

### **12** Inside Neural Re-ranking Models 
Alright, let's take a look inside in Neural ranking model and how most of the neural ranking models that we're looking at our build up. So the core part is a so-called matching module. Here we operate on word on the word interaction level. So each word in the query somehow interacts with each word in the document. And in this example here, you can see we have two input sequences - the question and the document. Both get encoded to, for example, with word embeddings, then maybe we have a feature extraction stage here that does some sort of feature extraction or interaction in one of the sequences and then we match the two sequences together. And after that we probably have some sort of module that takes this matrix of interactions between every term and Condenses it to the relevance score. So as output we get a single score. 

*82.64 seconds*

### **13** Training 
As I said, the training is independent from the rest of the search engine. And the training itself is also not dependent on a continuous retrieval of something so we can do this all in patches. Neural IR models are typically trained with triples, where you have a pairwise relevant or positive and one non relevant document or negative document. And the way this works is we have two forward passes in our model, so one the pair with the relevant document and one to pair with the non relevant document. And then in our loss function ,we want to maximize the margin of the score between relevant and the non relevant document. In Pytorch, yhis is done with the margin ranking loss, where we just use that loss function and provided with both results and PYTorch, then generates the computational graph for both paths, so to speak. So that means that all model components are trained end to end. And of course there are other ways to train those models, for example, with a listwise loss, but in this lecture we're going to keep it simple and only look at the pairwise loss. 

*98.55 seconds*

### **14** Evaluation 
From the training, let's now come to the evaluation part. So we score a single tuple of query and document. Although we're not evaluating those tuples on their own. No, we are doing a list based evaluation after we scored all query document pairs, sorted them and then we apply ranking metrics. Those same ranking metrics we talked about in the evaluation lecture such as MRR or NDCP@10. Just a reminder, MRR means Mean Reciprocal Rank where you only look at the first relevant document and stop looking at position 10. So this brings us to a mismatch. We can't really compare in training loss that's based on triples or a combination of two pairs and list based IR evaluation metrics. And in practice this looks the following. So the training loss is only good for checking at the beginning if the loss goes down. But then the training loss quickly converges to a relatively small number such as 0.02 or something and stays roughly at that point, although the IR evaluation metrics continue to improve during and continuous validation of our model, which is very interesting and you should keep that in mind for the exercise. 

*109.02 seconds*

### **15** MS MARCO Microsoft MAchine Reading COmprehension Dataset 
Then the next ingredient for good neural deep neural re ranking models is having a lot of training data. And one more major dataset that came out in the end of 2018. And that major dataset that makes possible all the things we're talking about today is the Ms Marco data set which stands for Microsoft machine reading comprehension dataset. And it's the first re ranking data set with actually too much training data, so we're too much means that at some point we have to stop because we don't benefit anymore from having more training data. And yeah, so now we have a luxury problem that the scale of training and evaluation data and the number of queries we have to evaluate becomes an issue. The Ms Marco dataser was released by Microsoft research and it's based on real Bing web search queries that have been sampled from the Bing search log and passage-level answers that have been annotated by human annotators. And the way they annotated those queries is that the human annotator picked one relevant document out of a list of 10. And only judged that one relevant document as relevant, so those sparse judgments cover a broad range in terms of vocabulary, although we have to be careful that of course the evaluation becomes noisy because we only have one judged document. But to some extent, the large number of queries that is comfortably in the 10s of thousands solves or mitigates that noise problem of it, and makes the evaluation quite stable if we use enough queries. 

*146.15 seconds*

### **16** MS MARCO 
Ms Marco Training. So we have training triples where we have queries and one relevant and one non relevant document. And here in this example you kind of see that those training triples are truly of good quality because the relevant is human judged and the non relevant can be sampled randomly from the collection. And that means that relevant document definitely is relevant and the non relevant is very probable non relevant, although of course again we could introduce noise at that stage, but it doesn't seem to be that much of an issue. And then for the evaluation, we worked the following: so we have an document ID and a query ID and that way we can then map the texts of the query and document back to the relevance labels. 

*70.75 seconds*

### **17** Actual input & output 
Now that we've seen examples of how the text looks like, so how humans look at the text, we now look at the actual input and output of our neural re ranking model. And to input, we have tensors of word IDs containing each a batch of samples, so a tensor is a multidimensional array abstraction. And batching means that for efficiency, we're not scoring a single pair at a time, but we actually pack multiple queries and multiple documents together in a tensor so that on our GPU, we can compute them in parallel. And this means that the same dimension, for all entries, needs to be with the same length. So each dimension you can see that as if you look at it in the simplest example, we have a 2 dimensional array, right? That means that every row and column have to have the same number of entries, otherwise you could not simply allocate one block of memory and you could not simply index on every position that's allowed by  the boundaries of the array. So to do that, we so called "pad" shorter sequences with zeros or a special padding vector. Um, that even though we have all almost exclusively we have sequences of different lengths. We pad them to be the same length, so that the sequence with the maximum length defines the size of our tensors, and every sequence that's shorter than that gets padded. And for the query tokens, the shape is simple, so we have first the first index defines our batch size and our different batch samples, and then the second indexible dimension is the query. The maximum query sequence length and for the document tokens it's the same, but with the document sequence length. 

*161.7 seconds*

### **18** 
Alright, so how do our neural re ranking models work with this input that we give them? Let's take a look. 

*12.68 seconds*

### **19** The encoding layer 
We start with the Encoding layer, so it's the starting point for our text process, for any text processing task. And here we take our inputs that we just saw. So the word IDs or word-piece IDs and we convert them to a dense representation. And in Information Retrieval it's very important to have word boundaries in your model so that you know what and when word starts and when a word ends. Even if we operate on word pieces, it can be quite useful to know which word pieces are in which word. And we can know that if we also have an offset vector that tells us when words start and end based on parts of a word. Right, and so the actual implementation or the pre train data of an encoding layer is pretty easily swappable. So you can evaluate the same information Retrieval Model with different encoding layers and of course you can swap in some more complex methods of encoding your words. And the encoding layer is usually shared between the query and the document encoding. So if you get the same word, it also is signed the same, or at least in very similar vector and similar is assigned the same vector. If you for example have a simple word embedding, then you will get the same vector. But if you have contextualized representations, you will probably not get exactly the same vector, but at least if the model works correctly, you will get a very similar vector that still closer to the own word than to others, even though it contains contextual information. 

*135.93 seconds*

### **20** The encoding layer 
So typically before 2019, this encoding layer indeed was a word embedding. Simple word embedding, nothing else. You took a pre-trained word2Vec or Glove word embedding and you fine tuned it with the rest of the model. But since 2019, this encoding layer and simple encoding layer is not state of the art anymore. And state of the art encoding layers are based on transformer self attention models which show very strong results but also quite complex, so we're going to look at them in our next lecture and in this lecture we cover the basics and start with simple word embeddings only. And even though they're not state of the art anymore, in my opinion they still have a lot of strong benefits. So first and foremost is speed. A simple word embedding is much, much faster than any contextualization, 'cause it's just a simple memory look up on the GPU. If you load the word embedding on the GPU. And they are also somewhat interpretable. You have a lot of analysis tools for them and you can reuse them, etc. 

*86.89 seconds*

### **21** The match matrix 
Good, after the encoding layer what many neural models build upon or the core of them is the so-called match matrix, where you go from query and document representations and each representation interacts with each other. For example, by using a cosine similarity. So the cosine similarity takes 2 vectors and produces for two vectors, one output value. So when you have, two lists of vectors that interact with each other, what you get out is an interaction matrix. Then for each interaction contains exactly 1 value and based on those similarities, you can then do more. 

*65.22 seconds*

### **22** Cosine similarity 
The cosine similarity, too formalize it a bit, looks like that. So you take the dot product of normalized vectors. And with that you measure the direction of vectors, but not the magnitude. And, technically, you shouldn't say that it is a distance, but it is equivalent to the Euclidean distance of unit vectors, say cosine similarity.

*39.85 seconds*

### **23** Cosine similarity in PyTorch 
All right, uhm? And the cosine similarity in Pytorch can be implemented as a quite efficient batched matrix multiplication, because of course in our pictured example here we only have one sample. But in reality you need to visualize another layer then we have multiple examples at the time and this looks like that. So you have an input shape of batch dimensions, query sequence length and then embedding dimensions, the same for the document. So you now have 3 dimensional tensors. And as an output shape, you want to have the batch dimension, then the query sequence length, and the document sequence length. Hum. And. Which basically says for each combination of query and document you get one value and in pytorch you can do it like that. 

*62.02 seconds*

### **24** MatchPyramid 
Right, and with that, we're going to take a look at our first neural IR model called Match Pyramid. And Match Pyramid is very much influenced from computer vision models. So what you do is you compute the match matrix based on word embedding interactions. And on top of that match matrix, you apply a set of 2D convolution layers. And the set of convolution layer builds up on top of each other. So you have, CNN's and then you have dynamic pooling, what we saw in the last lecture as well to take care of the variable length input of the match matrix and create a fixed sized output. What we found is that the architecture and the effectiveness very strongly depends on the configuration. So, the model is defined as that the number of layers, the sizes, etc, they are all hyper parameters. Nut the model and very strongly depends on a good selection of those hyperparameters. But generally, you said it so, that the pooling output becomes gradually smaller, so, and it has a pyramid sized shape, and that's where the name match pyramid comes from. 

*90.6 seconds*

### **25** MatchPyramid 
Now, let's take a look at the architecture or the flow of the architecture in a visual way. So we start off with our word embeddings, the encoding. Then we have a match matrix followed by our set of CNN layers, and each CNN layer produces a number of output matrices with multiple channels. So the convolutional layer extract local interaction features and by using a Max pooling layer, we only keep the strongest interaction or match signals. And by having different channels in our CNN's, we can learn different interaction patterns. And what match pyramid very often learns is to have n-n-grams such as Bigrams and trigrams. Those create stronger interaction signals than single word matches. And finally, after our last CNN layer, a multilayer feedforward module scores the extracted feature vectors. 

*84.52 seconds*

### **26** MatchPyramid 
This is formalized as the following. I'm not going to go through the formulas right now, but of course you can pause the video and look at the formula yourself. 

*22.69 seconds*

### **27** KNRM 
The next model we're going to look at is the so called KNRM Model, Kernel based neural ranking model. And here we are not using any sort of convolution, but a very efficient and very smart way of counting the amount of different similarities between a query and a document. The kernel part itself does not have much knowledgeable parameters, only the embedding layer here learns a lot of parameters. And by being very simple in the computational complexity, the KNRM model is very fast. So it's definitely by far the fastest model we're talking about today and in this course as a whole. And it has, on its own, it has roughly the same effectiveness as match pyramid. 

*68.41 seconds*

### **28** KNRM 
Right, so again, we have our query and document representations from our encoding layer. Each representation gets matched in the match matrix and then on top of the match matrix we apply a set of RBF kernel functions. And this set of RBF kernel functions then creates a set of matrices that contain the activation of the kernel function, which are then summed up per document dimension and for query dimension. Which results in a single value per kernel function, which is then waited to become the final score and we just learn the weightings of the kernels, but we don't actually learn the kernel functions here,  we take fixed functions.

*71.77 seconds*

### **29** KNRM 
This can be formalized the following. So, first we start with the match matrix. And then for each entry in the match matrix, Mij we apply this kernel function K here. And the kernel function takes the, Gaussian kernel function, which takes the exponential of the following formula. And then, after we get, and it results in an activation between zero and one. And after that each document term gets summed up, so we're left with value per query term. Then, those query terms are log normalized, which is very important to form a score per kernel and then the kernel is summed up.  

*77.83 seconds*

### **30** KNRM – Inside a Kernel 
Alright, so. I talked a lot about kernels and activation functions and it might it's very hard to just visualize that intuitively without seeing how this Gaussian kernel function looks like. So here is one. The thing is, we start off with cosine similarities and those cosine similarities are in the range of one to minus one. And they can't be outside that range, so we only have to look at the range from one to minus one and you can view this plot here as the input of the function. The cosine similarity is on the X axis, so if we have a cosine similarity of let's say one, we get roughly an activation of 0.6. And if we have a cosine similarity of 0.9 in this kernel function, we get a full activation of one. So we can never get a higher activation in one. And if the cosine similarity is not in our range around 0.9 here, this particular case will not be activated then. 

*87.31 seconds*

### **31** KNRM – Inside all Kernels 
That's why we have multiple Kernels, so here typically for KNRM we have something like 10 kernels that overlap each other. And they are able to count he number of times a cosine similarity value is inside their kernel range. If you sum up the activations after the kernels.  And as you can see, that's also why we don't learn the kernel activations on their own because they are evenly spaced out and they cover the whole range of our potential cosign similarities. Additionally, KNRM also defines an exact match kernel that only looks at cosine similarities of one. And it can do that because it only uses word embeddings that produced the same vector for each same occurrence of a bird.  

*76.81 seconds*

### **32** And now ... A short tale about reproducibility 
Last year I actually re implemented all those neural IR models I'm showing you today. And so, I can report you some interesting findings that I had so, so to speak, a short tail about reproducibility. And how hard it can be to actually achieve what the original paper laid out because of missing details or confusion in what to do exactly. 

*38.31 seconds*

### **33** KNRM – Implementation Details Matter 
And that's why details matter. So here you can see two heatmaps of the output of our evaluation of two KNRM runs. So on the X axis, horizontally, you can see the evaluation at different cut offs. So how much documents we actually evaluate until so at K from one to 100. And then vertically, we can see the training time coming from top down. So lower means we are later in the training time. And on the left side you can clearly see that the model does not learn anything because the more documents you add, the worse it gets. And the best MRR score is very poor, so it stays the same as our initial BM25, which of course is not good. So if we can't have a neural model that produces better scores than our simplest baseline, that's a bad sign. But on the right side, we see a run where we actually learn to incorporate more and more documents and get better results from that the longer we train. So we can see that the model actually starts to learn. Which is also visible with the best MRR score that we reach in the end. So what's the difference well? Let's have a look. 

*124.37 seconds*

### **34** KNRM – Implementation Details Matter 
The difference is quite small. The difference is a single added one. Namely, we just saw that the document activations before they get some up to the query on the query dimension are normalized with a log. And if you now take a log of one plus the value instead of just the log, it doesn't learn anything. And the problem now is there is open source code available for both versions, which of course makes it hard to see what to do here. And at first it might seem counter intuitive, because if you only use the log, you get negative values, if your soft term frequency here is smaller than one. But our best educated guess is that the log on its own acts as a regularization, and if you don't have a lot of occurrences of a cosine similarity, ot actually affects the score negatively. And you can basically ignore single occurrences as noise and only score documents that are have more matches than that. 

*103.72 seconds*

### **35** Conv-KNRM 
The next model that I want to show you today is the convocation KNR model which is a great extension of the KNRN model and introduces convolutional neural networks into the process. Namely, it cross matches n-gram representation and then applies the kernel pooling from KNRM. And which in theory allows to match convolutional neural networks with deep learning. And the place with the fact that N grams and term proximity are very important in Information Retrieval. And it's not feasible to create a vocabulary with all possible n-grams. The Conv-KNRM model is the most effective model highlighted today. And only next week we will see, we will basically take the next step towards the current state of the art. 

*74.44 seconds*

### **36** Recall: Word N-Grams with 1D CNNs (Lecture 5) 
Let's recall from the previous lecture. On sequence modeling in NLP, how we can create word n-grams with one dimensional CNN's? Right, so we start again with one word embedding per word and then we run a one dimensional CNN with a sliding window across our sequence to create n-gram representations. 

*33.55 seconds*

### **37** Conv-KNRM 
And this is exactly what the container and model is doing. So, here in the encoding layer, you can see that before we do any sort of matching, each sequence is run through CNN's and we have CNN's of multiple windows sizes. So we have CNN with size 2, CNN with size 3 and also CNN with size 1 to basically filter single matches down to the same dimension so that we can cross match them altogether. And KNRM creates a single match matrix for every combination. So you match single words with bigrams with trigrams. You match bigrams with each other. You match Bigrams and trigrams, and so on. And then the configuring applies KNRM on top of the match matrix and concatenates all the different n-gram combination results, weights them together to form the final score. 

*81.87 seconds*

### **38** Conv-KNRM 
Formalized Conv-KNRM looks like this. So we basically apply in 1D_CNN on our encoding. Different 1D_CNN's on our encoding and what we do is we add another dimension to our kernel tensors right? So we operate on more dimensions here. But the basic layout stays the same. We apply the matching for each combination. And as I just said, KNRM Kernels basically stay the same. 

*47.14 seconds*

### **39** Other models 
Other models, other early neural IR Models include Packer, which applies multiple 2D convolutional layers on top of the match matrix. Which is a little bit different than match pyramid does it. Then we have that the DUET model which models individual word matches and also creates a single vector per document and query and takes the similarity there and then combines both paths at the end. And the, I would say foundational yet not effective neural re ranking model, is the DRMM model, so it uses hard histograms of similarities and because of that, the histograms are not differentiable anymore. And the embedding is not updated which makes this model absolutely not effective, but the general ideas it presented resonate through all our models until today. 

*75.58 seconds*

### **40** 
Alright, we're almost done with this lecture, but I'm not letting you off the hook just now. Now we're going to take a look at some deeper evaluation, especially on the effect of low frequency terms on the neural IR models I just showed you. 

*18.76 seconds*

### **41** On the Effect of Low-Frequency Terms 
So what is the general context of this work? Infrequent terms carry a lot of relevance inforamtion in information Retrieval. And if you search for an infrequent term, you're very likely that this term contains the most impact on your search query. If you search for multiple words plus the infrequent term. But if we use neural IR models with a fixed vocabulary then those infrequent terms are removed mainly as a concession to efficiency and memory requirements, but that also means that the neural IR model doesn't see removed terms. And a fixed vocabulary for all terms wouldn't scale, and even so, even if you would have a fixed vocabulary for all terms, you could still have very little training data for those terms or unseen query terms that are again out of your vocabulary. 2019 Uh, and cigar. 

*82.66 seconds*

### **42** On the Effect of Low-Frequency Terms 
We have two contributions here. We show that the importance of covering all terms of a collection is paramount to good model results. And we are the first to analyze this re ranking threshold as a great tool for diagnostics to see if a model gets better than more documents you actually re-rank. And if a model doesn't perform well, well then the more re ranking documents decrease the effectiveness. And then the second part is that we also used fast text to strongly increase the effectiveness, especially for queries that contain low frequency terms. And just to recap, fast text as I told you in the word embedding lecture is a subword embedding made of character in n-gram compositions, so low frequency terms get better representations. 

*70.19 seconds*

### **43** Effect of the Fixed Vocabulary Size 
And the results for the three models that we looked at today are as follows. So especially match pyramid and KNRM suffer if they have small vocabularies. And even can get worse than the baseline of BM25 if those vocabularies are too small and don't cover enough of the collection. And especially, if you then use fast text, you get better results overall, for all models except for KNRM where the results are quite on par. 

*43.42 seconds*

### **44** Handling of Low-Frequency Terms 
But if you look at low frequency terms, you can see that the difference between a full vocabulary and fast text is very strong for infrequent terms that appear less than 20 times in your collection. So in this plot you can see everything that's red means it's better with fast text. And we plot on the X axis results based on the minimum collection frequency of terms in the query. But the differences become less with higher frequency terms which I would say is to be expected because fast text and Word2Vec for example are very similar for high frequency terms. 

*61.53 seconds*

### **45** Handling of Low-Frequency Terms 
And we can even look at the collection frequency in more details. And now we're only focusing on Conv-KRNM for queries with the collection frequency of lower than 20. And we can see that here fast text is the only method that consistently improves over the BM25 baseline, even for very low frequency terms. 

*31.86 seconds*

### **46** Interested? Here is more… 
Right, so if that all sounds interesting, here are some more pointers. So for example, the very good survey article by Mitra at al. on Neural. As well as a conference tutorial and of course the next lecture where we talk about Transformers And Bert based re ranking to look at all the advances that have been made in 2019 that cumulated in Google and Bing actually using the very costly BERT based ranking in their production systems.  

*48.02 seconds*

### **47** Summary: Neural Networks for IR 
To summarize, so what I want you to take away from this lecture is that we score pairs of query and document full text, but we train with triples and we evaluate listwise. There is quite a lot of things going on. Then word level match matrices are the core building block of early neural IR models. And the environment in which you use this neural IR model, namely the vocabulary and the re ranking depth mater lot. 

*44.72 seconds*

### **48** Thank You  
With that I thank you for your attention that you watch the video through to the end and I hope I inspired you for the second exercise and to come back for the next talk when we talk about self attention. See you. 

*20.85 seconds*

### Stats
Average talking time: 69.82676041666666

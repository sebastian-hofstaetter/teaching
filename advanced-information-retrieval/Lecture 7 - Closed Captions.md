# Lecture 7 - Introduction to Neural Re-Ranking

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Hi everyone, I'm especially excited today because this is the first lecture of our neural IR block and we're going to take a look at some of the workflows and basic neural re ranking models. My name is Sebastian and if you have any questions please feel free to contact me. 

*25.82 seconds*

### **2** Today 
Today we're looking at two things, specifically. First, is the re ranking workflow and it's connection to the first stage ranker, including training and evaluation of neural re ranking models and then in the second part we're going to take a look at Basic re ranking models which don't use Transformers. And here we're going to take a look at some of the models stages and then some of the models such as MatchPyramid and kernel pooling based models that have been developed before birth. As well as the influence of fixed sized vocabularies. 

*50.27 seconds*

### **3** Disclaimer: Text Based Neural Models 
Before we start, I think we need to set this lecture in the right context. So today we look at text based neural models in learning to rank systems. Usually when you run them in production you can have many different features such as click, count, recency, source quality, etc and the learning to rank field that works with classical. Machine learning techniques such as regression, SVM or tree based. Techniques can use those features as well. Plus increasingly in production at larger companies run those text based neural models that we also look at. I also have to stress that deploying a rear anchor in production is a huge effort on performance, quality control, version management, changing indices, etc. And we don't look at them today, but what we're going to look at today is neural models for content based ad hoc retrieval, which means that we only use the text of the query and the document, and this can be thought of. It's another signal in a bigger learning to rank system. And we mainly use this content based and talk retrieval without other features because the open test collections that we have available don't provide us with more features. 

*104.79 seconds*

### **4** Disclaimer: The Basics are not SOTA Anymore 
And the second disclaimer that the basics that we look at today are not state of the art anymore. So Transformer based models surpassed those initial initially proposed simple models in terms of quality. However, we believe that it's still interesting and informative to look at them and also many of the intuitions now guide our work with Transformers. End the basic models R much faster and resource efficient to train, so that's why we also use them in our exercise, which could be helpful to have a look at the slides today. And of course the re ranking workflow is still used with transformer based models if they are used as rear anchors. More about transformer. Re ranking Bert models and dense retrieval is coming in the next lectures. 

*67.9 seconds*

### **5** Desired Properties of Neural IR Models* 
The desired properties of neural IR models are pretty much in line with any other machine learning tasks. They should be effective, right? So they if they don't work then we just wouldn't use them. They need to be fast and scalable, so that's something that maybe differentiates the information retrieval task more from other machine learning tasks. Is that if you do online re ranking meaning you rerank when the user searches for something, you basically have a time budget of 10 to 100 milliseconds for the full query. And potentially your indices are terabytes large. Of course this can vary a lot, right? So we have also we're also working with a lot of indexes that are smaller for more specialized products and projects, so not everyone is working on a web search engine. And then, in my opinion, also a very important point is that those models should be interpretable because search engines filter information so they act as a gate. To access certain informations for a large number or potentially large number of people. That means that the reasons for including or excluding certain results should be explainable and it should be analyzable. If you see that something is not right. 

*101.89 seconds*

### **6** Beware! Neural Networks Fail Silently … 
And with that, maybe let's do a small warning for the exercise, especially so. Neural networks fail silently, which means that you can train in neural network for hours and hours and it will produce some results. You will get some numbers out of it. But that doesn't mean that the network learns correctly, or that the network. Produces any meaningful outputs. And. Even worse, sometimes you have small mistakes in there that just cost you a couple of percentage points of difference in your evaluation metrics, but those couple of percentage points could mean the difference between a new state of the art model and something that's kind of in the in the middle of everything. Right so the. Neural re ranking field is very much a Wild West at the moment we have a lot of new developments. And a lot of. An enormous amount of problems is not solved at the moment, and we don't know the answers to it and we don't have best practices yet and kind of every research group has their own best practices. So it's a little bit different than other research fields that are more mature already, but I think that also makes the research field more exciting. So if you want to know more information about what could go wrong with neural network training and how to potentially fix that and find those errors, you might find deadlink quite handy. 

*116.22 seconds*

### **7** 
Let's talk about the workflow. So what is actually different for neural re ranking models? In comparison to other machine learning tasks. 

*13.42 seconds*

### **8** Neural Re-Ranking Models 
Neural IR models Re rank A. Candidate List of results so they don't actually create their own index at the moment. We're going to look at indexing features and techniques in a later lecture. In this lecture, we only focus on the RE ranking part, which means. That we still depend on an inverted index and the first stage rancor such as being 25 that we learned about in the crash course to produce our candidate list. Let's say we have a candidate list of 1000 potential documents that come comes out of our first stage rancor here. Then for those thousand documents, we look up the full text in our full text storage and together with the query text each. Document gets paired with the query text, so the neural IR models actually. As the same interface as classical ranking methods, you get the score of a pair of 1 query and one document. And then after each pair is scored, you just sort the pairs by the return score to. For example, take the top ten documents out of the pool of 1000. And only present those top ten documents to the user. 

*102.49 seconds*

### **9** Context 
And we also have to be aware of how a search engine. It's interacted with so search engines have a lot of users potentially right, and those users interact with the search engine in two ways. So one way the search engine delivers users the query results. But users also deliver search engine's click data. So they know where a user clicked on. And this can be saved in logs. And then over time those logs can be accumulated and used to improve the search engine. And here in this picture we see we start with a couple of users that are sometimes happy and sometimes they're not happy but overtime we use the clicks as a relevant signal so we can make more users happy down here. And the way we do is is at certain intervals neural IR model's are trained so called offline. And. Are then swapped out with the actual running model, so it's not like the neural IR model continuously learns, but it is a loop that is somewhat asynchronous with the whole process, because of course. You want to do quality control before actually deploying a neural IR model. 

*100.31 seconds*

### **10** Inside Neural Re-ranking Models 
Alright, let's take a look inside. In neural ranking model and how most of the neural ranking models that were looking at our build up. So the core part is a so called matching module. Here we operate on word on the word interaction level. So each word in the query somehow interacts with each word in the document and. In this example here you can see we have two input sequences. The question and the document. Both get encoded. Two, for example with word embeddings. Then maybe we have a feature extraction stage here that does some sort of feature extraction. Or interaction in one of the sequences and then we match the two sequences together. And after that we probably have some sort of module that takes this matrix of interactions between every term. And condenses it to the relevance score. So as output we get a single score. 

*82.64 seconds*

### **11** Training 
As I said, the training is independent from the rest of the search engine. And the training itself is also not dependent on. A continuous retrieval of something so we can do this all in batches. Neural IR models are typically trained with triples where you have a pairwise relevant or positive and. One non relevant document or negative document. And the way this works is we have two forward passes in our model, so once the pair with the relevant document and once the pair with the non relevant document and then in our loss function we want to maximize the margin of the score between relevant and the non relevant document. In Pytorch, this is done with the margin ranking loss, where we. Just use that loss function and provide it with both. Results and Pytorch then generates the computational graph for both paths, so to speak. So it means that all model components are trained end to end. And of course there are. Other ways to train those models, for example with a list wise loss, but in this lecture we're going to keep it simple and only look at the pairwise loss. 

*98.55 seconds*

### **12** Creating Batches 
To best utilize modern CPUs, we train with batches of multiple examples per gradient update, and usually we do it by forming a batch as large as possible that fits into our so that everything fits into our GPU memory. Typically this is somewhere between 16 and 128 triples. And here we mix different queries together so we don't have the same query, but. A diverse set of queries and depending on the model, we need to create query passage pairs or we can run each of the three sequences individually through the model. As can be seen here on the left side. We run a backward pass and gradient update per batch and the sequence. Inputs, they come as a single matrix as a single tensor, so we need to pad different length inputs to make them all fit into the same tensor, because the tensor always needs the same sequence length for very efficient. Matrix multiplications on top of that. 

*92.46 seconds*

### **13** Sampling Non-Relevant Passages 
Virtually all collections that we're working with in academia only come with judgements of relevant or false positive selections from other models, but not judgements. For truly non relevant cases, because it doesn't really make sense to spend a notation resources on annotating completely random pairs, 'cause the probability that those random pairs are non relevant is quite high. So we somehow we need to tell the model what is none relevant so and we don't have information about which passages are non relevant, but. This simple procedure that is very often used to sample non relevant passengers uses the more simpler beam 25 retrieval model with an inverted index to get the, for example, top 1000 results per training query and then randomly select a few of those results as non relevant. Which of course we. Could make mistakes in there, but empirically it has been shown that this procedure works very well and produces non relevant passages that are not completely random because they still provide some signal as there must be at least some lexical overlap. Because 25 works on lexical overlap but their mostly non relevant passages. And a bit of noise is also good, right? So we. We don't mind if there's a bit of noise in our training data. More on how we compose batches and how we sample from the index and. What that means for dense retrieval comes in future lectures. 

*128.8 seconds*

### **14** How to Break the Non-Relevant Sampling 
Even though most of the time non relevant sampling works pretty straightforward out of the box. We can easily break non relevant sampling if we make a mistake in our sampling procedure. So too many false negatives which are actually relevant confuse the model during training, and one such example is if we have click data as relevance signals. We also potentially know about non click passengers, so passages which have been in the result list, but the users skipped over them and did not click on them. And if we now go ahead and sample those non click passengers. As negatives, we actually break the training. As we can see here on the right hand side. As the validation, as the Orange line of our validation results for the trip click collection where we even use a bird based model which should train very well. But in the Orange case did not train very well and once we removed those non click passengers from our negative sampling procedure and only rely I'd on be in 25 negatives. Shown in the Green line, we can see that the model trains much, much better and in actions only the first validation result actually starts after the 1st 4000 patches actually basically starts at the best point of the previous Orange Run, but of course continues much higher. Then that. 

*123.95 seconds*

### **15** Loss Functions 
And for the loss we also have multiple choices that we can take, but in practice usually the choice of the binary relevance loss is not very critical, so here we give you 2 examples. First, we can use a plain margin loss, which is in Pytorch called margin ranking loss, which just pushes the two scores from the relevant and non relevant document away from each other without any activation without any transformation, just the scores need to get away from each other. And then. We can also use the so-called rank net loss when she uses a binary cross entropy on the margin of two scores, and both losses assume binary relevance. The deep dive on non binary loss functions comes in the knowledge distillation lecture. 

*73.0 seconds*

### **16** Re-Ranking Evaluation 
From the training, let's now come to the evaluation part. So we score a single tuple of query and document. Although we're not evaluating. Those are tuples on their own. Now we are doing a list based evaluation after we scored all query document pairs, sorted them and then we apply ranking metric. Those same ranking metrics we talked about in the evaluation lecture such as MRR or NDC G at 10. Just a reminder, MRR means mean reciprocal rank where you only look at. The first relevant document. And stop looking at position 10. So this brings us to a mismatch. We can't really compare eat training loss. That's based on triples or a combination of two pairs and list based IR evaluation metrics. And in practice this looks the following so that training loss is only good for checking at the beginning if the loss goes down. But then the training loss quickly converges to a. Relatively small number such as 0.02 or something and stays roughly at that. Point, although the IR evaluation metrics continue to improve during a continuous validation of our model, which is very interesting and you should keep that in mind for the exercise. 

*109.02 seconds*

### **17** Recall How MS MARCO Looks 
Ms Marco data kind of looks like that. So we have training triples where we have queries an you and one relevant and one non relevant document. And here in this example you kind of see that. Those training triples are truly awful. Good quality because the relevant is human judged and the non relevant can be sampled randomly from the collection. And that means that relevant document definitely is relevant and the non relevant is very probable non relevant, although of course again we could introduce noise at that stage, but it doesn't seem to be that much of an issue. And then for an evaluation. We work the following so we have an document ID and a query ID and. That way we can then map the texts of the current document back to the relevance labels. 

*70.75 seconds*

### **18** Actual input & output 
Now that we've seen examples of how the text looks like, so how humans look at the text, we now look at the actual input and output of our neural RE ranking model. And. Two input. We have tensors of word IDs containing each a batch of samples, so a tensor is a multidimensional array abstraction. And patching means that for efficiency, we're not scoring a single pair at a time, but we actually pack multiple queries and multiple documents together in a tensor, so that on are GPU. We can compute them in parallel. And this means that the same dimension. Home for all entries. Needs to be. With the same length, so each dimension you can see that as if you look at it in the simplest example, we have a 2 dimensional array, right? That means that every row and column have to have the same number of entries, otherwise you could not simply allocate one block of memory and you could not simply index on every position that's. Allowed by by the boundaries of the array. So to do that, we so called pet shorter sequences with zeros or a special padding vector. That even though we have. All almost exclusively we have sequences of different lengths. We pet them to be the same length so that the sequence with the maximum length defines the size of our tensors, and every sequence that's shorter than that gets padded. And for the query tokens, the shape is simple, so we have first the first index defines our batch size and our different batch samples, and then the second indexible dimension is the query. The maximum query sequence length and for the document tokens it's the same, but with the document sequence length. 

*161.7 seconds*

### **19** 
Alright, so how do our neural re ranking models? Work with this input that we give them. Let's take a look. 

*12.68 seconds*

### **20** The encoding layer 
We start with the encoding layer so it's the starting point for our text process. For any text processing task. And here we take our inputs that we just saw. So the word IDs or a word piece IDs and we convert them to a dense representation and information we will. It's very important to have word boundaries in your model so that you know. What? When word starts and when it word ends, even if we operate on word pieces. It can be quite useful to know which word pieces are in which word, and we can. Know that if we also have an offset vector that tells us when word start and end based on parts of a word. Right, and so the actual implementation or the pre train data. Often encoding layer is pretty easily swappable so you can. Evaluate. The same information retrieval model with different encoding layers and of course. You can swap in. Some more complex methods of encoding your words. And the encoding layers usually shared between the query and the document encoding. So if you get the same word, it also is signed the same, or at least in very similar vector and similar. So in this assigned the same vector, if you for example have a simple word embedding, then you will get the same vector. But if you have contextualized representations, you will probably not get exactly the same vector, but at least. I'll Mr Models works correctly. You will get a very similar vector that's still closer to the own word than to others, even though it contains contextual information. 

*135.93 seconds*

### **21** The encoding layer 
So typically before 2019, this encoding layer indeed was a word embedding. Simple word embedding. Nothing else. You took a pre trained work back or glove word embedding and you fine tuned it with the rest of the model. But since 2019. This encoding layer simple encoding layer is not state of the art anymore and state of the art is state of the art. Encoding layers are based on transformer self attention models which show very strong results. But also quite complex, so we're going to look at them in our next lecture and in this lecture we cover the basics and start with simple word embedding's only. And even though they're not state of the art anymore, in my opinion they still have a lot of strong benefits. So first and foremost is speed. A simple word embedding is much, much faster than any contextualization, 'cause it's just a. Simple memory look up on the cheap you if you load the word embedding on the GPU. And they are also somewhat interpretable. You have a lot of analysis tools for them and you can reuse them, etc. 

*86.89 seconds*

### **22** The match matrix 
Good after the encoding layer. What many neural models? Billdip on or the core of them is the so called Match matrix where you. Go from query and document representations and each representation interacts with. Each other, for example by using a cosine similarity so the cosine similarity takes 2 vectors and produces for two vectors, one output value. So when you have. Two lists of vectors that interact with each other. What you get out is an interaction matrix that for each interaction contains exactly 1 value and. Based on those similarities, you can then do more. 

*65.22 seconds*

### **23** Cosine similarity 
And the cosine similarity to formalize it a bit looks like that. So you. Take the dot product of normalized vectors. And with that you measure the direction of vectors, but not the magnitude and. Technically, you shouldn't say that it is a distance, but it is equivalent to the Euclidean distance of unit vectors. So say cosine similarity. Which is. 

*39.85 seconds*

### **24** Cosine similarity in PyTorch 
Alright, and the cosine similarity in PyTorch. Can be implemented as a quite efficient batched matrix multiplication, because of course in our pictured example. Here we only have 1 sample, but in reality. You need to visualize them. Another layer that we have multiple examples at the time and this looks like that. So you have an input shape of batch dimensions, query sequence length and then embedding dimensions. The same for the document. So you now have free dimensional tensors, Ann's and output shape. You want to have the. Bets dimension, then the query sequence length and the document sequence length. I'm. And. Which basically says for each combination of query and document you get one value and in PyTorch you can do it like that. 

*62.02 seconds*

### **25** MatchPyramid 
Right, and with that, we're going to take a look at our first neural IR model. Called MatchPyramid and MatchPyramid is very much influenced. From computer vision models. So what you do is. You compute the match matrix based on. Embedding interactions and on top of that met match matrix. You apply a set of 2D convolution layers and the set of convolutional layer builds up on top of each other, so you have. CNN's and then you have dynamic pooling. What we saw in the last lecture as well to take care of the variable length input of the match matrix and create a fixed sized output. What we found is that the architecture and the effectiveness very strongly depends on the configuration. So the model is defined as that the number of layers, the sizes, etc. They are all hyperparameters, but the modern very strongly depends on a good selection of those hyperparameters, but generally you said it so that the pooling output becomes gradually smaller. So it has a permit size shape and that's where the name MatchPyramid comes from. 

*90.6 seconds*

### **26** MatchPyramid 
No, let's take a look at the architecture. More the flow of the architecture in a visual way. So we start off with our morning beddings, the encoding. Then we have a match matrix followed by our set of CNN layers and each CNN layer produces a number of output matrices with multiple channels. So the convolutional layer extract local interaction features and by using a Max pooling layer we only keep the strongest interaction or match signals. And by having different channels in our CNN's we can learn different interaction patterns. And what MatchPyramid? Mary often learns is to have engrams such as bigrams and trigrams. Those create stronger interaction signals than single word matches. And finally, after our last CNN layer. A multilayer feed forward module scores the extracted feature vectors. 

*84.52 seconds*

### **27** MatchPyramid 
This is formalized as. The following. I'm not going to go through the formulas right now, but of course you can pause the video and look at the formula yourself. 

*22.69 seconds*

### **28** KNRM 
The next model we're going to look at is the so called KNRM model, kernel based neural ranking model. And here we are not using any sort of convolution, but a very efficient and very smart way of counting the amount of different similarities between the query and a document and the kernel part itself does not have much nonverbal parameters, only the embedding layer here learns a lot of parameters. And by being very simple in the computational complexity, the cannon model is very fast, so it's definitely by far the fastest model we're talking about today, and in this course as a whole. And it has on its own. It has roughly the same effectiveness as MatchPyramid. 

*68.41 seconds*

### **29** KNRM 
Right, so again, we have our query and document representations from our encoding layer. Each representation gets matched in the match matrix and then on top of the match matrix we apply a set of RBF kernel functions and this set of RBF kernel functions then creates. A set of matrices that contain the activation of the kernel function, which are then summed up pair query. Now per document dimension and per query dimension. Which results in a single value per kernel function, which is then waited to become the final score and we just learn the. Waitings of the kernels. But we don't actually learn the kernel functions here. We take fixed functions and. 

*71.77 seconds*

### **30** KNRM 
This can be formalized in the following so. First we start with the match matrix. And then for each entry in the match matrix, MIJ we apply this kernel function key here and the kernel function takes the. Gaussian kernel function, which takes the exponential of the following. Formula and. Then after we get and it results in an activation between zero and one. And after that each. Each query gets each document term gets summed up, so we're left with value per query term then. Those query terms are log normalized, which is very important to form a score per kernel and then the kernel is summed up.  

*77.83 seconds*

### **31** KNRM – Inside a Kernel 
Alright, so. I talked a lot about kernels and activation functions and it might it's very hard to just visualize that intuitively without seeing how this caution kernel function looks like. So here is 1. The thing is, we start off with cosine similarities and those cosine similarities are in the range of 1 to minus one. And. They can't be outside that range, so we only have to look at the range from one to minus one and you can view this plot here as the input of the function. The cosine similarity is on the X axis, so if we have a cosine similarity of let's say one, we get roughly an activation of 0.6 and if we have equals and similarity of 0.9 in this kernel function. We get a full activation of 1 so we can never get a higher activation in one. And if the cosine similarity is not. In our range around 0.9 here. This particular case will not be activated then. 

*87.31 seconds*

### **32** KNRM – Inside all Kernels 
That's why we have multiple kernels, so here typically for KNRM we have something like 10 kernels that overlap each other. And they are able to count. The number of times a cosine similarity value is inside their kernel range. If you sum up the activations after the kernels and. As you can see, that's also why we don't learn the kernel activations on their own because they are evenly spaced out and they cover the whole range of our potential cosine similarities. Additionally, can a RAM also defines and exact match kernel that only? Looks at cosine similarities of 1 and it can do that because it only uses word embeddings that produce the same vector for each same occurrence of a bird.  

*76.81 seconds*

### **33** And now ... A short tale about reproducibility 
Last year I actually reimplemented all those neural IR models. I'm showing you today. And so I can report you some interesting findings that I had, so it's, so to speak, a short tale about reproducibility and how hard it can be to actually achieve what the original paper laid out. 'cause of missing details or confusion in what to do exactly. 

*38.31 seconds*

### **34** KNRM – Implementation Details Matter 
And that's why details matter. So here you can see. Two heat Maps of the output of. Some of the output of our evaluation of a KNRM of two KNRM runs, so on the. X axis horizontally. You can see the evaluation at different cut offs. So how much documents we actually evaluate until so at K from one to 100 and then vertically we can see the training time coming from top down. So lower means we are later in the training time. And on the left side you can clearly see that the model does not learn anything. Because the more documents you add, the worse it gets. And the best MRR score is very poor, so it stays the same as our initial PM 25, which of course is not good. So if we can't have a newer model that produces better scores than our simplest baseline. That's a bad sign, but on the right side we see a run where we actually learn to incorporate more and more documents and get better results from that. The longer we train. So we can see that the model actually starts to learn. Which is also visible with the best MRR score that we reach in the end. So what's the difference well? Let's have a look. 

*124.37 seconds*

### **35** KNRM – Implementation Details Matter 
The difference is quite small. The difference is a single added one. Namely, we just saw that. The. Document. Activations before the. They get some up to the query and on the query dimension are normalized with a log and if you now take. A log of 1 plus the value instead of just the log. It doesn't learn anything, and the problem now is there is open source code available for both versions, which of course makes it hard to see what to do here. And at first it might seem counter intuitive, because if you only use the log, you get negative values. If your soft term frequency here is smaller than one. But our best educated guess is that the log on its own acts as a regularization, and if you don't. Have a lot of occurrences of a cosine similarity. It actually affects the score negatively. And. You can basically ignore single occurrences as noise and only. Score documents that are have more matches than that. 

*103.72 seconds*

### **36** Conv-KNRM 
The next model that I want to show you. Today is the conflict in Aran model, which is a. Great extension of the KNRM model and introduces convolutional neural networks into the process. Namely, it crossmatches engram representation, and then applies the kernel pulling from KNRM, and which in theory allows to match convolutional neural networks with deep learning. Home. And. The place with the fact that engrams and term proximity are very important in information retrieval. And it's not feasible to create a vocabulary with all possible engrams. The Concannon model is the most effective model highlighted today. And only next week we will see and we will basically take the next step towards the current state of the art. 

*74.44 seconds*

### **37** Recall: Word N-Grams with 1D CNNs 
Let's recall from the previous lecture. On sequence modelling in NLP, how we can create word engrams with one dimensional CNNs? Right, so we start again with one word embedding poor word and then we run. In one dimensional CNN with a sliding window across our sequence to create engram representations. 

*33.55 seconds*

### **38** Conv-KNRM 
And this is exactly what the conflict in Aran model is doing so. Here in the encoding layer. You can see that. Before we do any sort of matching, each sequences run through CNN's and we have CNN's of multiple window sizes. So we have CNN with size 2, CNN with size 3 and also CNN with size 1 to basically filter single matches down to the same dimension so that we can cross match them altogether. And. Camera M creates a single match matrix for every combination. So you match single words with bigrams with trigrams. You match bigrams with each other. You match bigrams and trigrams and so on. And then the. Then configuration applies KNRM on top of the match matrix and concatenates all the different end grain combination results. Weights them together to form the final score. 

*81.87 seconds*

### **39** Conv-KNRM 
Formalized Hannah confirm looks like this. So we basically apply. In one the CL on our encoding. Different one DCNS on our encoding and what we do is we add another dimension to our kernel tensors, right? So we operate on more dimensions here. But the basic layout stays the same. We apply the matching for each combination. And as I just said, Kerem kernels basically stayed the same. 

*47.14 seconds*

### **40** Other models 
Other models other early neural IR models include Packer, which applies multiple 2D convolutional layers on top of the match matrix. Which is a little bit different than match permit does it? Then we have the threat model, which models individual word matches and also creates a single vector per document in query and takes a similarity there and then combines both paths at the end and the. I would say foundational yet not effective. They will re ranking model is the DRMM model, so it use is hard histogram's of similarities and because of that the histograms are not differentiable anymore and the embedding is not updated which makes this model. Absolutely not effective, but the general ideas it presented resonate through all our models until today. 

*75.58 seconds*

### **41** 
Alright, we're almost done with this lecture, but I'm not letting you off the hook just now. Now we're going to take a look at some deeper evaluation, especially on the effect of low frequency terms on the neural IR models. I just showed you. 

*18.76 seconds*

### **42** On the Effect of Low-Frequency Terms 
So what is the general context of this work? Infrequent terms carry a lot of relevance information information retrieval, and if you search for in infrequent term, you're very likely that this term contains the most impact on your search query. If you search for multiple words plus the infrequent term. But if we use neural IR models with a fixed vocabulary then. Those infrequent terms are removed mainly as a concession to efficiency and memory requirements, but that also means that the neural IR model doesn't see removed terms. And a fixed vocabulary for all terms wouldn't scale, and Even so, even if you would have a fixed vocabulary for all terms. You could still have very little training data for those terms or. Unseen query terms that are again out of your vocabulary and we presented this work in 2019. And cigar. 

*82.66 seconds*

### **43** On the Effect of Low-Frequency Terms 
We have two contributions. Here. We show that the importance of covering all terms of a collection. Is a paramount to good model results. And we are the first two to analyze this re ranking threshold as a great tool for diagnostics to see if a model gets better. The more documents you actually rank and if the model doesn't perform well. Well, the more re ranking documents decrease the effectiveness. And then the second part is that we also used FastText to strongly increase the effectiveness, especially for queries that contain low frequency terms. And. Just to recap, FastText as I told you in the morning, bending lecture is a subword embedding. Made of character in Graham composition's so low frequency terms get better representations. 

*70.19 seconds*

### **44** Effect of the Fixed Vocabulary Size 
And the results for the three models that we looked at today are as follows, so. Especially match permit and kenaram. Suffer if they have small vocabularies. And even can get worse than the baseline of PM. 25 If those vocabularies are too small and don't cover enough of the collection, and especially. If you then use fast text, you get better results overall. For all models except for Caner M Warner results are quite on par. 

*43.42 seconds*

### **45** Handling of Low-Frequency Terms 
But if you look at low frequency terms. You can see that the difference between a full vocabulary and FastText is very strong. For infrequent terms that appear less than 20 times in your collection, so in this plot you can see everything that's red means it's better with FastText and we plot on the X axis. Results based on the minimum collection frequency of terms in the query. But the differences become less with higher frequency terms which. I would say is to be expected because FastText and work to back for example. Are very similar for high frequency terms. 

*61.53 seconds*

### **46** Handling of Low-Frequency Terms 
And we can even look at the collection frequency in more details. So now we're only focusing on conflict in or M4 queries with the collection frequency of lower than 20. And we can see that here FastText is the only method that consistently improves over the beam 25 baseline, even for very low frequency terms. 

*31.86 seconds*

### **47** Summary: Neural Networks for IR 
To summarize, So what I want you to take away from this lecture is that. We score pairs of query and document full text, but we train with triples and we evaluate listwise. There's quite a lot of things going on. Then word level match matrices are the core building block of early neural IR models. And the environment in which you use this neural IR model, namely the vocabulary and the re ranking depth matter lot. 

*44.72 seconds*

### **48** Thank You  
With that I thank you for your attention that you watched the video through to the end and I hope I inspired you for the second exercise and to come back for the next talk when we talk about self attention. See ya. 

*20.85 seconds*

### Stats
Average talking time: 74.15631640624999
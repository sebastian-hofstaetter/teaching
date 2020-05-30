# Lecture 7 - Transformer Contextualization and Re-ranking

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Hi everyone, welcome to this lecture on transformer contextualization and re ranking. 

*10.66 seconds*

### **2** Today 
Today we're going to take a look at contextualized representation learning, with Transformers and the so-called multihead self attention. Then we're also taking a look at pre trained models such as BERT that have been all the rage lately. And then, following our introduction to the contextualization, we look at how to employ those techniques in the re ranking scenario. First, how to use BERT for re-ranking and then how to use an efficient transformer kernel architecture. That is actually my proposed model. So I'm very happy to talk about that and I hope you enjoy this lecture.  

*57.26 seconds*

### **3** Now we Reached the State-of-the-art 
Finally, we reached the state of the art techniques. In general, this is a very fast moving field, so most of this lecture didn't exist in the beginning of 2018. And by july this year, we probably have new state of the art techniques that need to make those slides updated. And then we also have to note that many questions in this field are very much open. And we try to answer them. But the more questions we answer, the more questions come up. But the general direction is because we have now more computational resources combined with more training data, we get better results.  

*58.35 seconds*

### **4** 
So how can we contextualize a sequence when it only attends by itself?  

*11.87 seconds*

### **5** Contextualization via Self-Attention 
The general idea of contextualization is we have term representations, so we start again with some sort of word or subword embedding, and each word gets assigned an embedding, just as in the previous lectures. But now we have a second layer underneath that and that's the contextualized representation, where each vector corresponds to a combination of all the vectors in the sequence, and now the problem is how to learn this combination for our words. So in this example here the word "step", if you only look at term representation of "step" on its own, it can have many different meanings and you probably are not going to pick this one. But with the contextualization the word "step" here should correspond to the correct meaning. Of course, this is a very simplified example, but I hope you get the general idea. Context in this context means also only the local sequence, so we're not looking at external knowledge bases or anything like that. And then we also have to start right off the bat and say the general idea here is very computationally intensive, because for every contextyalized output representation,you need to combine all other term representations, which means that you get an O(N2) complexity. 

*128.66 seconds*

### **6** Transformer 
The transformer architecture contextualizes with multi head self attention and commonly just as with RNNs, we stack multiple layers of Transformers on top of each other. The transformer can be utilized as an encoder, but also in the encoder-decoder combination we saw in the previous lecture to generate natural language output. The large improvement over RNNs comes from the fact that Transformers do not require any recurrence. So the attention is done via is series of matrix multiplications over the sequence as one tensor, and we don't have to have any recurrence to go through the sequence, which of course makes the computation easier. There's a lot of boilerplate code that falls off. And it's much better to paralyze on modern GPUs. And the transformer has initially been proposed in translation, but now it's the backbone of basically any NLP Advancement in the last year and last couple of years. 

*89.95 seconds*

### **7** Transformer – Architecture  
So before we run our embeddings through the self attention mechanism of the transformer architecture, we have to do one crucial step. And this is we add a positional encoding to every vector in our embedding, because now we don't have recurrence, we need some other mechanism for the model to know about the relative and absolute positions of the word in the sequence. And therefore we add this positional encoding as an element wise addition to our embeddings and how that looks will see in a minute. And after we added the positional encoding, each transformer layer has three projection layers, the query, key and values. So for the attention itself were looking at the query and key projection. Each of those projections is basically a fancy way of saying linear layer. And after we ran every element in our sequence through those linear layers, we matrix multiply the query and key by transposing one of them. So now, we have an interaction between the elements, between each word inside the transformer. But we have a twist here. Now is the fun part where the multihead aspect comes in. So by convention we say that each head of the query and key projections starts at a certain index. So basically if we have 10 heads, and our dimensionality is 100 every 10th index we start with a new head. And we can do this by just reshaping the tensor inside the model. So we're not changing the data, but we're just having a convention that yes, every 10th index we start a new head, so now we reshape our tenses to add this additional head dimension, and now everything runs basically in parallel and each head can focus on different things. After we did that, we receive our attention via a softmax for each head, do you run interactions through a softmax. And then we receive a single attention value for each input word for every other output word. Basically a matrix of attentions and those are multiplied with the vectors that come out of the value projection. Then again, we also need to project it back to the output dimensionality because we can have different total dimensions between the input and the sum of the multi head dimensions. And this output is then fed into the next transformer layer, and so on and so on, until we get our contextualized representations back. That, in general, have the same dimensions as the input. And each word corresponds to a contextualized word of the output.  

*259.85 seconds*

### **8** Self-Attention Definition 
In the transformer paper, this attention mechanism is formally defined as follows. So we have, again our QKV values, that are those projections so this formula hides quite a bit of complexity. But let's just assume we have our projections and then we matrix multiply, query and key. And divided by a scaling factor and the scaling factor is the square root of the key embedding dimensionality. After we ran it through the Softmax, we also multiply it with the value. So now we get out our attended values. The thing here is we also have another part of complexity and that is sequence length that are uneven. So if we have uneven sequences, we actually apply a masked softmax that masks out padding values so they receive a zero attention, so that in the end the sum of attention values, sums up to one.

*94.6 seconds*

### **9** Transformer – Positional Encoding 
Coming back to the positional encoding that I mentioned earlier. The initial transformer used sinusoid curves and those look like that. So each dimension receives as an addition, the value of one of those curves at its current location in the sequence, which means that, relatively speaking, the distance, if you measure the distance, for example with a cosine similarity, the distance of the relative distance is always the same between positions, but of course the absolute value is different.  

*60.16 seconds*

### **10** Transformer - Variations 
Oh well, so it would be kind of easy if we could say yeah, the one transformer paper solved all problems and we're all happy with that. Well, even though it's an incredible work and spurred alot of new ideas, there is a lot we can fix or improve upon this initial architecture, and as you can see, a lot of people did so. So I would say this is a very non exhaustive list of different transformer variants and improvements. There's a lot of focus on efficiency and how to handle long input to break this quadratic runtime and especially memory requirement. And if you look at the. year values in each citation here, you can see that the speed of innovation is incredible and if you work on an improvement through transformer architecture, the chances are very high that someone else is doing almost the exact same thing and will publish before you. So there is an incredible rush going on at the moment.  

*89.56 seconds*

### **11** Masked Language Modelling 
OK, so how do we train those models? Well, we can of course train them supervised in their target domain. But one thing that is becoming increasingly popular or the most popular usage of Transformers is actually to pre train them in an unsupervised way, similar to how you would pre train word embeddings and then take the pre trained word embeddings and build upon it. If we come back to our example here, so we want contextualized representations for the word "steps" here and the unsupervised pretraining task, called mask language modeling takes this text and it masks random words and then it forces to model to predict the original word. And then you can update your weights based on the loss of prediction versus the actual word 

*76.0 seconds*

### **12** Masked Language Modelling 
And this prediction runs over as a probability over the full vocabulary. That means that you need a small vocabulary to make this prediction, because for the probability, again you have to use softmax activation to get the probability out of your model. And the Softmax is very costly for large vocabularies. So what most people here do is we don't take a large vocabulary that only contains single words, but we use different tokenizers to split infrequent terms in their parts so that they become multiple tokens. And with this the vocabularies become much smaller, so popular techniques are called WorldPiece or BytePair encoding. But the general idea is the same that they split infrequent words by some algorithm. And then you can map those word parts back together. 

*89.72 seconds*

### **13** BERT 
Bert standard big pre train transformer model that is out there right now. Very similar to how Word2Vec a couple of years ago over run everything, Bert is now doing the same. Coincidentally both come from Google. So BERT stands for bidirectional encoder representations from Transformers. It showed large effectiveness gains out of the gate and the more people work with it, the bigger the improvements get. So the general ingredients that make all those gains possible are WordPiece Tokenization. So, again we have a small vocabulary that covers infrequent terms by splitting them up into pieces. But very frequent terms get their own representation. So it's not like every word get split up, but only the infrequent ones get split up. Then, BERT models are very large. So we're looking at the base model alone has 12 transformer layers with 700 dimensions each for each vector. So you require a lot of memory to train and also to infer. Then Bert kind of uses a multi task approach where some parts of the model are reused between tasks and shared. And BERT achieves that not by creating a special network architecture and connections inside the network, no.  BERT achieves that by using specialized tokens. And those specialized tokens are just part of the vocabulary and only how they are employed in the pre training and fine tuning stages, their meaning comes to life, so to speak. So the most important one is the CLS token, the classification token. It is prepended to every sequence that's fed into BERT. And it is used as a pooling operator to get a single vector per sequence. And when you have the single vector, you can just add a single linear layer that reduces this vector to score, or a class probability for any prediction classification Multilabel Task you might have based on your encoded words. Then the second token is the MASK token. So BERT is pre trained with mask language modeling and for that it replaces some words with mask tokens and then the model is forced to predict those words. Next is separated token. So to allow for more  tasks where you also need to compare two sentences together. BERT is actually trained always with two sentences together. So we have one sentence then a separate the token and then the other sentence. This is also augmented with additional sequence encoding's that are learned. Hey, quick side note, BERT also learns positional encodings and does not use the fixed sinusoid curves as the original Transformers. And if you have the two sentences, you can do stuff like question answering for example, where the question is the first sentence in the fine tuning, and the answer is the 2nd sentence. But it is pre trained on random sentence taken out of documents. And BERT is pre trained very very long. So if you would do it on one GPU alone it would take weeks on end to do that. But of course the authors of BERT didn't do that. They basically used a whole server rack full of GPUs to train BERT.  

*315.12 seconds*

### **14** BERT - Input 
Let's take a closer look at the input that gets fed into the BERT model. We have either one or two sentences. So we can also omit the second sentence if we want. BERT adds positional and segment embeddings on top of the token embeddings. And you can see that the position embeddings differs based on position and the segment  based on if it's part of the first or second sequence. And also in this example you can see if you look closely at the end of the input, you can see the effects of the WordPiece tokenization where the word playing is actually split up into play and ing. And ING is prepended with this special double hashtag, that's WordPiece tokenization way of telling us that this is a connection to the previous word. No other connection than that exists, so the model has to learn that on its own.  

*74.77 seconds*

### **15** BERT - Model 
The actual model is quite simple, so I don't think we need a formal definition for it. It's basically n layers of stacked Transformers with some special things like layer normalization, GeLu activations which are like relu activations but with as, I would say like a grace swing under zero that allows for a meaningful gradient if you have values that are negative but you don't really want to activate based on those values. Basically you want to have a way of pushing them into the positive range if needed. With ReLus you can't do that. So if a ReLu is zero it basically, is dead because the flat line of the negative equals 0 output does not allow for a gradient. Then BERT uses the so called task specific heads on top of the stack transformers to pool the CLS  or individual token representations and this pooling most of the time means we have single linear layer that takes in the dimension of the CLS token and Outputs a single score or a multilabel score. Right, every transformer layer receives input the output of the previous one, just as you would in other transformer architectures. And as I said before, and I want to re emphasize this point the CLS token is only special because we train it to be special in there is no mechanism inside the model that differentiates it from other tokens. And most in my opinion, most of the novel contributions in the BERT model center around pre training and especially the workflow of how other people then build up and interact with the model. 

*152.97 seconds*

### **16** BERT - Workflow 
The Workflow is pretty simple. Someone who has a lot of hardware pre trains the model on a combination of mask language modeling and next sentence prediction. So this is something knew that I haven't told you before. The next sentence prediction actually uses the CLS token to try to predict if the second sentence is actually the next sentence in the document of the first sentence, or if it is randomly picked from another document. And once Bert is trained for a very long time, we can just download it and fine-tune it on our task by switching the input and maybe switching the head on top of it. And to do that, there is a really awesome library called Transformers from huggingface, which incorporates numerous model variants. It incorporates a lot of pre trained models and a lot of simplicity to just get started with pre train transformer models. 

*83.94 seconds*

### **17** Beyond BERT 
And of course, just as with the transformer variance, there are now many BERT variants, so we have pre trained bird models for many languages, we have pre trained word models for domains like biomedical publications, etc. And, we also have different architectures that have a similar workflow, but maybe at different pre training regime, a little bit different architecture inside the Transformers to allow for bigger models, more efficient models and especially to allow for longer sequences as Bert is kept at 512 tokens in total for both sentences. Which of course, especially in information retrieval, tends to become a problem if you want to score full documents with thousands of tokens, but you don't actually want to run BERT with longer sequences. As with Transformers, BERT also has a quadratic runtime in the sequence length.  

*83.2 seconds*

### **18** 
And now we're going to take a look at how to use BERT and transformer architectures for re ranking. And I would say like the general theme here is that, well, they can be slow and very effective, or they can be fast and almost as effective.  

*25.75 seconds*

### **19** The Case for Contextualization in IR 
Our field of research with re ranking and transformer models is actually not disconnected to the real world because both the two biggest web search companies Google and Microsoft announced in the end of 2019 that they're both using BERT based or BERT style re ranking models for their web search engines in production.  

*36.9 seconds*

### **20** BERT Re-Ranking 
So how do you use BERT for re ranking? Well, it's actually quite simple. You score one query and one passage, just as with all the other neural re ranking models we saw in previous lectures. This means you need a candidate set of documents of passages. Then you concatenate the two sequences to fit BERTs workflow so you have a CLS token, the query text, a separated token, and the passage text. To get a score, you pull the CLS token with a single linear layer and you train with Pairwise Ranking loss or a fine tune, the pre trained bird model with the pairwise ranking loss. And it works awesome. Out of the box you fine tune it for a couple of hours, well maybe a day, well, I mean you have to find unit for like a week on a strong GPU, but then it works awesome. It provides major jumps in effectiveness across collections and domains. But of course it's slow, and one thing that's often overlooked. Yes, BERT works awesome and incredible, but we don't actually know why. We can't see inside the model because it's very much a black box.  

*104.79 seconds*

### **22** BERT In-Efficiency 
And Bert is like really inefficient for inference. And that's what in the end really counts. We want fast query responses. And so last year we did a comparison study between those original first generation neural re ranking models that I showed you in the previous lecture KNRM, C-KNRM, match pyramid, PACRR and DUET as well as BERT. And here in this plot you can see the run time in milliseconds for 250 passages per query that you re rank. And you can see that those IR specific networks are really fast. Right, so you are in the couple of milliseconds per query range for 200 documents. But they're not so good. And then you have BERT which is on top in the upper right corner there. But please note the jump in the X axis, so BERT does not take a couple of milliseconds, BERT actually takes a couple of seconds to do the same task. But it gives you much better results. So one thing to note here as well, is that, well, you might say OK, then the user has to wait 2 seconds. If it's a specific re ranking task, fine. But it's also infrastructure cost that comes with blocking a single GPU for two seconds at a time. And the trade off or in in classical learning to rank models between efficiency and effectiveness is well studied, but in Neural re ranking models were just getting started to understand what makes BERT fast or how can we improve, how can we make something like BERT but much faster?  

*142.57 seconds*

### **23** Efficiency – Effectiveness Tradeoff 
And we found that there is, of course large influence on how you employ the model for this efficiency effectiveness trade off. So the main outside factor is of course how many documents to re rank. If you re rank, fewer documents you will be faster. Even with GPU parallelization, there is an almost linear way of looking at this. So faster models in the same time as slower ones. And if you set out a time budget and say OK, I want to spend this many milliseconds to re rank something. It allows us to simultaneously evaluate the effectiveness and efficiency in a more realistic setting.  

*68.48 seconds*

### **24** TK: Transformer-Kernel Ranking 
And that's what we did by proposing the transformer kernel model. So last lecture we heard about kernel pooling and we bring kernel pooling in Transformers together to form a lightweight, interpretable and effective model. It has, the TK model, has very strong results compared to IR specific models. And if you apply a time budget constraint environment where you actually care about the speed of your model, it is the state of the art model and it beats BERT, in the same time frame. It uses transformer blocks as contextualization layer and we have a trick there that we create hybrid contextualization by merging the context and non context vectors after the contextualization. And one design goal of ours was that we want to limit the number of transformer layers 'cause each additional layer takes considerable amount of time and you get diminishing returns by adding more layers.  

*79.53 seconds*

### **25** TK: Transformer-Kernel Ranking 
The model looks like that, so we have as input on the left side here a query and a document full text of both. Then, very importantly, we run the contextualization independent. So that's a strong difference to how BERT works for BERT you always have to use query and documents together. In our model you can actually precompute document representations and save them or analyze them. In the contextualization phase we again add positional encoding, run our terms through the transformer layers. And then combine them with the original non contextualized representations, with a weighted sum to improve or make the lexical matching more important. We do the same for the query, but again, independently and then only then when we have the contextual as representations, we actually combine the two via image matrix and cosine similarity. So this is a bottleneck inside the model where each query document term representation is represented by one value. We can use this value for interpretation analysis and visualizations. And then after that we basically run KNRM style kernel pooling that counts how many matches we have in certain similarity ranges. And then we score them, the individual kernel scores we sum them up and we wait them to receive our final score. 

*131.1 seconds*

### **26** TK: Transformer-Kernel Ranking 
The results of the time budget analysis look like that, so the more. If you have more documents in the same time, you get better results, but of course after a while BERT takes over because it has overall better effectiveness results, but in a much higher cost of efficiency. And you can see that both were or 4 three times for MRR, recall and NDCG, our model in red here  outperforms traditional retrieval models and is better than BERT in a given time budget of up to 150 or in the case of recall where more documents are important, more important, over 600 milliseconds. 

*63.25 seconds*

### **27** Understanding TK  
So now how can we actually understand what our model is doing? And I saud before we incorporated a bottleneck of a single cosine match matrix and this cosine match matrix can be very nicely visualized. And so we created a demo application called Neural IR Explorer to actually visualize what the model seed in terms of the similarity of contextualized words. So you can browse around and you can kind of get a feeling for how a test collection looks, how a model works, what sees, what works, and what doesn't work, and we see it kind of as a complementing thing to metric based evaluation. 

*65.2 seconds*

### **28** 
Right, so here is a demo. You can see the MSMARCO as you know from the exercise has a lot of queries. So to overcome this complex view we clustered the queries based on their contextualized representations. Another side output of our model, if you will. Then for each cluster you can sort the queries by how much better the Neural model improves over the initial beam 25 baseline you can filter out queries to find specific words. And then you can look at how a single query looks in terms of a result list compared to a couple of documents at the same time. So you can for example, only focused on some words that you're interested in, and then you can look up why the model scored some documents higher than others. And of course, you can also compare two documents side by side to get a more detailed look at how this whole thing works and if you like to look at shiny colorful things, please check out the live demo that's available at Neural IR Explorer dot EC dot TOV dot AC .80. 

*109.6 seconds*

### **29** TKL: Transformer-Kernel for Long Documents 
Right, so with our TK model, we now have a very efficient and effective model for passage re ranking. But in Information Retrieval  we will we also encounter long documents that have thousands of tokens where we can't just or we shouldn't just cut off the 1st 200 and be done with it. Um? And most models currently out there, they don't contain any notion of region importance or a way of handling the long document text well. So the current best approach that has been done is we split document and we score individual sentences or paragraphs and then externally outside of the model we combine those scores, which is not an end to end approach. We proposed an extension for TK for long Documents called TKL which just recently has been accepted to a conference. Where we look at an end to end approach that you can actually train based on full documents at once. 

*93.83 seconds*

### **30** TKL: Transformer-Kernel for Long Documents 
And the Schematic overview looks like that. So you have again a query and a document, but this time our document is quite long and to overcome the quadratic runtime requirement of Transformers, we actually contextualize in overlapping chunks to come to a linear runtime requirement. The document, again is contextualized independent from the query, so we could precompute document representations to speed up our production system. And then we apply the same term interaction and kernels. But for the scoring part, we actually don't want to look at the document as a whole, but we want to scan across the document and find regions of important because as it turns out, most of the time only a couple of Regions, multiple sentences, etc are relevant to the query, and so we want to detect those regions both for interpretability and to get a better scoring representation out of it. So we do that by scanning the saturation function. The KNRM style log based situation function or lock like saturation function across the document find the top relevance regions and scored them together in a weighted way.  

*121.95 seconds*

### **31** TKL: Why long documents? 
So why do we actually need long documents? We evaluated our model on different document length and here again in red you can see that only our model consistently, our TKL model, consistantly improves when presented with longer document input. And that's also actually the main idea behind exercise one. We wanted to find out if the model was correct in this assumption, because right now we can only say Yeah, the model works better, but do we actually via human annotations that documents at the end, also contain relevant information and how does the structure of relevant information look like. So that's what we did with exercise one we annotated documents in a very fine grain and thorough fashion, so now we can analyze all those things.  

*79.46 seconds*

### **32** TKL: Where is the relevance? 
And the TKL model also gives us a way to look at where it found the relevance. So in our configuration we said OK, let's look at the top three relevant regions per document. And we found out that the occurrence pattern of those regions across the document length actually follows a so-called zipfian distribution that represents a tailed distribution. So if you plot it via a log log scale like here, it looks like a linear way. But in reality, if you wouldn't plot it in a log log plot, it would very much look like a tailed distribution. And  this relevance region detection could also be used as a sniper generation, right? So you can have better user interfaces. 

*76.32 seconds*

### **33** Summary: Contextualization & Re-Ranking 
And with that I'm at the end of my talk. I hope you like it. I liked it and what I really want you to take away from this talk is how Transformers work so that they apply self attention to contextualize words. Then that BERT provides enormous effectiveness jumps, but at the cost of speed, so you always have to keep that in mind. And I hope you liked my work on combining Transformers and Kernels which if deployed correctly can lead to a very nice compromise between efficiency and effectiveness.  

*49.32 seconds*

### **34** Thank You  
So until next time, see ya. 

*6.48 seconds*

### Stats
Average talking time: 90.6415625

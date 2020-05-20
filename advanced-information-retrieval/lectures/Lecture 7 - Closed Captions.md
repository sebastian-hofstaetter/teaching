# Lecture 7 - Transformer Contextualization and Re-ranking

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Hi everyone, welcome to this lecture on transformer contextualization and re ranking. 

*10.66 seconds*

### **2** Today 
Today we're going to take a look at contextualized representation, running with Transformers and the so-called multihead self attention. Then we're also taking a look at pre pre trained models such as bird that have been all the rage lately. And then. Following our introduction to the contextualization. We look at how to employ those techniques in the re ranking scenario. First, how to use bird for re ranking and then how to use an efficient transformer? Kernel architecture is actually my proposed model. So I'm very happy to talk about that and I hope you enjoy this lecture.  

*57.26 seconds*

### **3** Now we Reached the State-of-the-art 
Finally, we reached the state of the art techniques. In general, this is a very fast moving field, so most of this lecture didn't exist in the beginning of 2018. And by two live this year, we probably have new state of the art techniques that need to make those slides updated. And then we also have to note that many questions in this field are very much open. And we try to answer them. But the more questions we answer, the more questions come up. But the general direction is because we have now more computational resources combined with more training data. We get better results.  

*58.35 seconds*

### **4** 
So how can we? Contextualize a sequence when it only attends by itself.  

*11.87 seconds*

### **5** Contextualization via Self-Attention 
The general idea of contextualization is we have term representations, so we start again with some sort of word or subword embedding, and each word gets assigned an embedding, just as in the previous lectures, but now. We have a second layer underneath that and that's the contextualized representation, where each vector corresponds to a combination of all the vectors in the sequence, and now the problem is how to learn this combination for our words. So that in this example here the word step if you only look at term representation of step on its own, it can have many different meanings and you probably are not going to pick this one. But with the contextualization? The word step here should correspond to the correct meaning. Of course, this is a very simplified example, but I hope you get the general idea. Context. Text means also only the local sequence, so we're not looking at external knowledge bases or anything like that. And then. We also have to start right off the bat and say. The general idea here is very computationally intensive because for every. Contextualization. You need to combine all other. Tom representations, which means that you get an O of N squared complexity. 

*128.66 seconds*

### **6** Transformer 
The transformer architecture contextualize is with multi head self attention and commonly just as with R and ends. We stack multiple layers of Transformers on top of each other. The transformer can be utilized as an encoder, but also in the encoder decoder combination we saw in the previous lecture to generate natural language output. The. Um, large improvement over R and ends comes from the fact that Transformers do not require any recurrence, so the attention. Is done via is serious of matrix multiplications over the sequence as one tensor, and we don't have to have any recurrence to go through the sequence, which of course makes the computation easier. There's a lot of boilerplate code that falls off. And it's much better to paralyze on modern Tribulus. And the transformer has initially been proposed in translation, but now it's the backbone of basically any. LP Advancement in the last year and last couple of years. 

*89.95 seconds*

### **7** Transformer – Architecture  
So. Before we run our embeddings through the self attention mechanism of the transformer architecture, we have to do one crucial step and this is we add a positional encoding. To every vector in our embedding, because now we don't have recurrence, we need some other mechanism for the model to know about the relative and absolute positions of the word in the sequence. And therefore we add this positional encoding as an element wise addition to our embeddings and how that looks will see in a minute. And after we added the positional encoding, each transformer layer. Has three projection layers, the query key and values. So for the attention itself were looking at the query and key projection, each of those projections. Is basically a fancy way of saying linear layer and after we ran every element in our sequence through those linear layers. We matrix multiply the query and key by transposing one of them. So now. We have an interaction between the elements between each word. Inside the transformer. But we have A twist here. Now is the fun part where the multi hat aspect comes in. So by convention we say that each. Mount each head of the query and key projections. Starts at a certain index. So basically if we have 10 heads, an hour dimensionality is 100 every 10th. Every 10th index we start with a new head and we can. Do this purchased, reshaping the tensor inside the model so we're not changing the data, but we're just having a convention that yes, every 10th index we start a new hat, so now we reshape our tenses to add this additional head dimension, and now everything runs basically in parallel and each had can focus on different things. After we did that, we receive our attention via a softmax for each head. So when you run interactions through a softmax and. Then we receive a single attention value for each. Change input word for every other output word. Basically a matrix of attentions and those are multiplied with the vectors that come out of the value projection. Then again, we also need to project it back to the output dimensionality because we can have different total dimensions between the input and the sum of the multi head dimensions. And this output is then fed into the next transformer layer, and so on and so on until we get our contextualized representations back. That, in general, have the same dimensions as the input. And each word corresponds to a contextualized word. Of the output.  

*259.85 seconds*

### **8** Self-Attention Definition 
In the transformer paper, this attention mechanism is formally defined as follows, so we have. A again our QKNV values. And are those projections so this formula Heights quite a bit of complexity. But let's just assume we have our projections and then. Um, we matrix multiply, query and key. And divided by a scaling factor and the scaling factor is the square root of the key embedding dimensionality. After we ran it through the Softmax, we also multiply it with the value. So now we get out our attendant values. The thing here is we also have another part of complexity and that is sequence length that are uneven. So if we have uneven sequences, we actually apply a masked softmax. That masks out padding values so they receive a zero attention, so that in the end. The each. Attention, Attention. Phone. 

*94.6 seconds*

### **9** Transformer – Positional Encoding 
Coming back to the positional encoding that I mentioned earlier. The initial transformer used, seen aside curves and those look like that so each dimension. Receives In addition. Those are in the value of one of those curves at its current location in the sequence, which means that, relatively speaking, the distance. If you measure the distance, for example with a cosine similarity, the distance of the relative distance is always the same. Between positions, but of course the absolute value is different.  

*60.16 seconds*

### **10** Transformer - Variations 
Oh well, so it would be kind of easy if we could say yeah, the one transformer paper solved all problems and we're all happy with that. Well, even though it's an incredible work and spurred alot of new ideas. There is a lot we can fix or improve upon this initial architecture, and as you can see, a lot of people did so, so I would say this is a very non exhaustive list of different transformer variants and improvements. There's a lot of focus. On efficiency, long. Take this quadratic runtime and especially memory requirement. And if you look at the. Year values in each citation. Here you can see that the speed of innovation is incredible and if you work on an improvement through transformer architecture, the chances are very high that someone else is doing almost the exact same thing and will publish before you. So there is an incredible rush going on at the moment.  

*89.56 seconds*

### **11** Masked Language Modelling 
OK, so how do we train those models? Well, we can of course train them supervised in their target domain, but. One thing that is becoming increasingly popular or the most popular usage of Transformers is actually to pre train them in an unsupervised way. Similar to how you would pre train word embeddings and then take the pre trained word embeddings and build upon it. If we come back to our example here, so we want contextualized representations for the word steps here and the unsupervised pretraining task, called mask language modeling. Text this text and it masks random words and then it forces to model to predict the original word. And then you can update your weights based on the Los of prediction versus. 

*76.0 seconds*

### **12** Masked Language Modelling 
The actual. Work. And this prediction runs over as a probability over the full vocabulary. That means that. You need a small vocabulary to make this prediction, because for the probability, again you have to use softmax activation to get the probability out of. Your model an the Softmax is very costly for large vocabularies, So what most people here do is we don't take a vocabulary large vocabulary that only contains single words, but we use different tokenizers to split infrequent terms in their parts so that they become multiple tokens. And with this the vocabularies become much smaller, so popular techniques are called World Peace or bite pair encoding. But the general idea is the same that they split infrequent words by some algorithm. And then you can. Map those part part word parts back together. 

*89.72 seconds*

### **13** BERT 
Bert standard big pre train transformer model that is out there right now. Very similar to how work back a couple of years ago. Over run everything. Bert coincidentally both come from Google, so bird stands for bidirectional encoder representations from Transformers. It showed large effectiveness gains out of the gate and. The more people work with it, the bigger the improvements get. So the general ingredients. That make all those gains possible are. WordPiece Tokenization. Small vocabulary. Covers infrequent. Up into pieces but. Very frequent terms get their own. Get their own representation so it's not like every word get split up, but only the infrequent ones get spin up then. Birth models are very large, so we're looking at the base model alone has 12 transformer layers with 700 dimensions each for each vector. So you require a lot of memory to train and also to infer. Then Bert. Uses a multi task approach where some parts of the model are reused between tasks and shared, and Bourke achieves that not by. Creating a special network architecture and connections inside the network or no bird achieves that by using specialized tokens and no specialized tokens are just part of the vocabulary and only how they are employed in the pre training and fine tuning stages. Their meaning comes to life, so to speak, so the most important one is the CLS token, the classification token. It is prepended to every sequence that's fed into Bird, and it is used as a pooling operator to get a single vector per sequence. And when you have the single vector, you can just add a single linear layer. That reduces this vector to score, or a class probability. For any prediction classification Multilabel Task you might have based on your. Encoded words. Then the second token is the mask token. So bird is pre trained with mask language modeling and for that it replaces some words with mask tokens and then the model is forced to predict those words. Next is separated token, so too. Allow for more. Tasks WordPiece birthday trained always with two sentences together. So we have one sentence. Then a separate the token and then the other sentence. This is also augmented with additional sequence encoding's that are learned. Hey, quick side note. Burt also learns positional encodings and does not use the. Use the fixed sinusoid curves as the original Transformers. And if you have the two sentences, you can do stuff like question answering for example, where the question is the first sentence in the fine tuning, and the answer is the 2nd sentence. But it is pre trained on. Random sentence is taken out of documents. And bird is pre trained very very long. So if you would do it on one GPU alone it would take weeks on end to do that. But of course the authors of birth didn't do that. They basically used a whole server rack full of purpose to train bird.  

*315.12 seconds*

### **14** BERT - Input 
Let's take a closer look at the input that gets fed into the bird model. We have either one or two. Sentence is so we can also omit the second sentence if we want bird. Adds. Embeddings. Embeddings sequence embeddings. And the segment based on if it's part of the first or second sequence. And also in this example you can see if you look closely at the end of the input you can see the effects of the word peace tokenization were the word playing is actually split up into play and in and ING is prepended with this special double hashtag, that's word peace, tokenization, way of telling us that this is a connection. To the previous word, no other connection than that exists, so the model has to learn that on its own.  

*74.77 seconds*

### **15** BERT - Model 
The actual model is quite simple, so I don't think we need a formal definition for it. It's basically end layers of stacked Transformers. With some special things like layer normalization, gelou activations which are like relu activations but with as, I would say like a grace swing under zero that allows for a meaningful gradient if you have values that are negative but you don't really want to activate based on those values. Basically you want to have a way. Of pushing them into the positive range if needed. With Rellos you can't do that. So if a rello is zero it. Basically, is dead because the flat line of the negative. Equals 0 output. Does not allow for a gradient. Then Bert. Cold task specific heads on top of the stack. Transformers to pool the CLS. Or individual token representations and this pooling most of the time means we have single linear layer that takes in the dimension of the CLS token and Outputs a single score or a multilabel score. Right every transformer layer receives input. The output of the previous one, just as you would in other transformer architectures. And as I said before, and I want to re emphasize this point to see less token is only special becausr we train it to be special in there is no mechanism inside the model that differentiates it from other tokens. And most in my opinion, most of the novel contributions in the bird model center around pre training and especially the workflow of how other people then build up and interact with the model. 

*152.97 seconds*

### **16** BERT - Workflow 
The Workflow. Is pretty simple. Someone who has a lot of hardware pre trains the model on a combination of mask language modeling and next sentence prediction. So this is something knew that I haven't told you before. The next sentence prediction actually uses the CLS token to try to predict the next. If the second sentence is actually. The next sentence in the document of the first sentence, or if it is randomly picked from another document. And once Bert is trained for a very long time, we can just download it and find unit on our task by switching the input and maybe switching the head on top of it. And to do that, there is a really awesome library called Transformers from hugging phase. Which incorporates numerous model variants. It incorporates a lot of pre trained models and a lot of simplicity to just get started with pre train transformer models. 

*83.94 seconds*

### **17** Beyond BERT 
And of course, just as with the transformer variance, there are now many burped variants, so we have pre trained bird models for many languages. We have pre trained word models for domains like biomedical publications etc. And we also have different architectures that have a similar workflow, but maybe at different pre training regime, little bit different. Architecture inside the Transformers to allow for bigger models, more efficient models and especially to allow for longer sequences as Bert is kept at 512 tokens in total for both sentences. Which of course, especially in information retrieval, tends to become a problem if you want to. Score full documents with thousands of tokens, but you don't actually want to run bird with longer sequences. As with Transformers, Burke also has a quadratic runtime in the sequence length.  

*83.2 seconds*

### **18** 
And now we're going to take a look at how to use Bird and transformer architectures for re ranking. And I would say like the general theme here is that. Well, they can be slow and very effective, or they can be fast and almost as effective.  

*25.75 seconds*

### **19** The Case for Contextualization in IR 
Our field of research with re ranking and transformer models. Is actually not disconnected to the real world becausr both the two biggest web search companies Google and Microsoft announced in the end of 2019 that they're both using bird based or bird style re ranking models for their web search engines in production.  

*36.9 seconds*

### **20** BERT Re-Ranking 
So how do you use bird for re ranking? Well, it's actually quite simple. You score one query and one passage, just as with all the other neural re ranking models we saw in previous lectures. This means you need a candidate set. I've documents of passages. Then you concatenate the two sequences to fit birds workflow so you have a CLS token. The query text, a separated token, and the passage text. To get a score, you pull the CLS token. With a single linear layer and you train with Pairwise Ranking Los or a fine tune, the pre trained bird model with the pairwise ranking Las. And it works awesome out of the box you fine tune it for a couple of hours. Well maybe a day. Well, I mean you have to find unit for like a week on a strong GPU, but then it works awesome. It provides major jumps in effectiveness across collections and domains. But of course it's slow, and one thing that's often overlooked. Yes, bird works awesome and incredible, but we don't actually know why. Can't see inside the model because it's very much the black box.  

*104.79 seconds*

### **22** BERT In-Efficiency 
And Bert is like really inefficient for inference. And that's what in the end really counts. We want fast query responses. And so last year we did a comparison study between those original first generation neural re ranking models that I showed you in the previous lecture K&RM convocatoria match pyramid Packer and wet. As well as Bird. And here in this plot you can see the run time in milliseconds for 250 passages per query that you re rank. And you can see that those IR specific networks are really fast. Right, so you are in the couple of milliseconds per query range for 200 documents. But they're not so good. And then you have bird which is on top in the upper right corner there. But Please note the jump. In the X axis, so bird does not take a couple of milliseconds. Bird actually takes a couple of seconds to do the same task. But it gives you much better results. So one thing to note here as well, is that, well, you might say OK, then the user has to wait 2 seconds. If it's a specific re ranking task, fine. But it's also infrastructure cost that comes with blocking a single GPU for two seconds at a time. And the trade off or in in classical learning to rank models between efficiency and effectiveness is well studied, but in Neural re ranking models were just getting started to understand what makes bird fast or how can we improve. How can we make something like bird but much faster?  

*142.57 seconds*

### **23** Efficiency – Effectiveness Tradeoff 
And we found. That there is, of course large influence on how you employ the model for this efficiency effectiveness trade off. So the main outside factor is of course how many documents to re rank. If you re rank. Fewer documents you will be faster. Even with GPU parallelization, there is an almost linear. Linear way of looking at this. To Faster models. In the same time as slower ones. And if you set out a time budget and say OK, I want to spend this many milliseconds to re rank something. It allows us to simultaneously evaluate the effectiveness and efficiency in a more realistic setting.  

*68.48 seconds*

### **24** TK: Transformer-Kernel Ranking 
And that's what we did by proposing the transformer kernel model. So last lecture we heard about Colonel pooling and we bring Colonel pulling in Transformers together to form a lightweight, interpretable and effective model. It has that it came and has very strong results compared to IR specific models. And if you apply a time budget constraint environment where you actually care about the speed of your model, it is the state of the art model and it beats bird. In the same time frame. It uses transformer blocks as contextualization layer and we have a trick there that we create hybrid contextualization by merging the context and non context vectors after the contextualization. And one design goal of ours was that we want to limit the number of transformer layers 'cause each additional layer takes considerable amount of time and you get diminishing returns by adding more layers.  

*79.53 seconds*

### **25** TK: Transformer-Kernel Ranking 
The model looks like that, so we have as input on the left side. Here a query and a document full text of both then. Very importantly, we run the contextualization independent, so that's. A strong difference to how bird works for Bertie always have to use query and documents together. In our model you can actually precompute document representations and save them or analyze them. In the contextualization phase we again adpositional encoding, run our terms through the transformer layers. And then combine them with the original non contextualized representations, with a weighted sum to improve or make the lexical matching more important. We do the same for the query, but again, independently and then only then when we have the contextual as representations, we actually. Combine the two via image matrix and cosine similarity, so this is a bottleneck inside the model where each. Query document term representation is represented by one value. We can use this value for interpretation analysis and visualizations. And then after that we basically run KNRM style kernel pooling. That counts how many matches we have. In certain similarity ranges, and then we score them. The individual kernel scores with some them up and we wait them to receive our final score. 

*131.1 seconds*

### **26** TK: Transformer-Kernel Ranking 
The results of the time budget analysis look like that, so the more. If you have more documents in the same time. You get better results, but of course after awhile. Burt takes over because it has overall better effectiveness results, but in a much higher cost of efficiency. And you can see that both war or 4 three times for MRR recall and Dennis Ritchie, our model in red here. Outperforms traditional retrieval models and is better. Big bird in a given time budget of up to 150 or in the case of recall where more documents are important. More important, over 600 milliseconds. 

*63.25 seconds*

### **27** Understanding TK  
So now how can we actually understand what our model is doing and? I. Set before we incorporated a bottle Neck of a single cosine match matrix and this cosine match matrix can be very nicely. Visualized and so we created a demo application called Neural IR Explorer. To actually visualize what the model CS in terms of the similarity of contextualized words so you can browse around and you can kind of get a feeling for how a test collection looks, how a model works, what season, what works, and what doesn't work, and we see it kind of as a complementing thing to metric. Based evaluation. 

*65.2 seconds*

### **28** 
Right, so here is a demo. You can see the Ms marker collection as you know from the exercise has a lot of queries. So to overcome this complex view we clustered the queries based on their contextualized representations. Another side output of our model, if you will. Then for each cluster you can sort the queries by how much better the Neural model improves over the initial beam 25 baseline you can filter out queries to find specific words. And then you can look at how a single query looks in terms of a result list compared to a couple of documents. At the same time. So you can for example, only focused on some words that you're interested in, and then you can look up why the model scored documents. Some documents higher than others. And of course, you can also compare two documents side by side to get a more detailed look at how this whole thing works and if you like to look at shiny colorful things, please check out the live demo that's available at Neural IR Explorer dot EC dot TOV dot AC .80. 

*109.6 seconds*

### **29** TKL: Transformer-Kernel for Long Documents 
Right, so with our DK model, we now have a very efficient and effective model for passage re ranking. But Information which we will we also encounter long documents that have thousands of tokens where we can't just or we shouldn't. Just cut off the 1st 200 and be done with it. Um? An most models currently out there, they don't contain any notion of region importance or. A way of handling the long document text well so the current best approach that has been done is we split document and we score individual sentences or paragraphs and then externally outside of the model we combine those scores, which is not an end to end approach. We proposed an. Long Documents. Which just recently has been accepted to a conference. Where we look at an end to end approach that you can actually. Train based on full documents at once. 

*93.83 seconds*

### **30** TKL: Transformer-Kernel for Long Documents 
And the Schematic overview looks like that. So you have again a query and a document, but this time our document is quite long and to overcome the quadratic runtime requirement of Transformers, we actually contextualize in overlapping chunks. Hum to only be. To to come to a linear runtime requirement. The document, again is contextualized independent from the query, so we could precompute document representations to speed up our production system. And then we apply the same term interaction and kernels. But for the scoring part, we actually don't want to look at the document as a whole, but we want to scan across the document and find regions of important because as it turns out, most of the time only a couple of Regions, multiple sentences, etc are relevant to the query, and so we want to detect those regions. Both for interpretability and. To get a better scoring representation out of it. So we do that. By scanning the saturation function. The key and rim style log based situation function or lock like saturation function across the document find the top relevance regions and scored them together in a weighted way.  

*121.95 seconds*

### **31** TKL: Why long documents? 
So why do we actually need long documents? We evaluated our model. On different document length and here again in red you can see that only our model consistently. Or she came on the consistantly improves when presented with longer document input. And that's also actually the main idea behind exercise. One we wanted to find out if the model was correct in this assumption, because right now we can only say Yeah, the model works better, but. Do we actually? No, via human annotations that. Documents. At the end, also contain relevant information and how. Thus the structure of relevant information look like, so that's what we did with exercise one we annotated. Documents in a very fine grain and thorough fashion, so now we can analyze all those things.  

*79.46 seconds*

### **32** TKL: Where is the relevance? 
And the detail model also gives us a way to look at where it found the relevance. So in our configuration we said OK, let's look at the top three relevant regions per document. And we found out that the occurrence pattern of those regions across the document length actually follows a so-called zipfian distribution. That represents a tailed distribution, so if you plot it via a log log scale like here, it looks like a linear. I'm in linear way. But in reality, if you wouldn't plot it in a log log plot, it would very much look like a tailed distribution. And. This relevance region detection could also be used as a sniper generation, right? So you can have better user interfaces. 

*76.32 seconds*

### **33** Summary: Contextualization & Re-Ranking 
And with that I'm at the end of my talk. I hope you like it. I liked it and what I really want you to take away from this talk is how Transformers work so that they apply self attention to contextualize words. Then that Burt provides enormous effectiveness jumps, but at the cost of speed, so you always have to keep that in mind. And I hope you liked my work on combining Transformers and Kernels which if deployed correctly can lead to a very nice compromise between efficiency and effectiveness.  

*49.32 seconds*

### **34** Thank You  
So until next time, see ya. 

*6.48 seconds*

### Stats
Average talking time: 90.6415625
# Lecture 4 - Word Representation Learning

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Hello everyone, today we're going to take a look at our first machine learning class and that is about Word Representation Learning.  

*12.38 seconds*

### **2** Today 
We are going to look at two specific things. First, are word embeddings. So how we actually transform words as we see it from characters to vectors, so that neural networks can work with them. And there, a very popular technique from the last years is Word2Vec and we're going to take a look at that as well. Then, we're going to take a first glimpse at what you could do with word embeddings in the information retrieval setting and we're going to look at unsupervised query expansion and how our group actually solved a problem of topic shifting that occurred from pre-trained word embeddings. So without further ado, let's get started. 

*51.59 seconds*

### **3** Differentiable Matrices & Transforms  
An important motivation for word embeddings is the fact that if we want to work with neural networks and we want to apply a gradient descent training algorithm, we need to operate in continuous spaces, so that means we need to have tensors with floating point values and continuous transformation functions that are differentiable. And without that we don't have a gradient, so we can't update our network parameters in the right quote, unquote direction, because we just don't know in which direction we should go to improve our result. So this means we can't just input the character values. For example, the ASCII codes of words in a linear algebra network and expect it to work because it wouldn't. So we need a method to map chars or word-pieces or words as a whole, defined by some tokenization to some sort of vector. To do that, we have a lot of options and we can't show every option today, but we'll look at a few and you can go and look up some more. Before we continue, I want to stress that you should take a look at the Pytorch Beginners Guide to understand various terminologies and basic ideas of how to train a neural network. I'm not going to repeat that in this lecture. 

*115.43 seconds*

### **4** First Pre-training then Fine-tuning  
A very common problem, and therefore practice how to deal with it in all kinds of machine learning projects, is: We often don't have a lot of data because labeling data is most of the time pretty expensive or you need a big user base to create noise free click data or so, so everything will work with language, we have the problem that we want good word-vectors or word representations for each individual word. But there are a lot of words in the English language so we can't cover all words in a fine-tuning step where we only have our own data, our task specific data. This means that those low frequency terms that don't or very little appear in our data, they could be a great indicator for your task. That's especially true in the retrieval domain. But they probably get bad representations from our little training data. So, what does the community do in that case? We pre-train word representations on a general text corpus, unsupervised, where we don't actually need labeled data and then we take that pre-trained parameters and put them in our model to fine-tune them so we train them with our own limited data, continue to train them with our own limited data. And this idea is a major factor in the advances of word embeddings and language models in natural language processing in the last couple of years. 

*130.19 seconds*

### **5** Take Pre-trained then Stack Together 
And of course we can take the pre-trained embedding and not only continue to use that exact pre-trained embedding, but we can stack it together like LEGO blocks with other stuff, right? So this is very common in the computer vision domain, but it's getting better and better in text domain as well. So for example, we can take our pre-trained module like a word embedding and put it in a bigger module, where the rest of the model that utilizes the word embeddings is randomly initialized and only trained in the fine-tuning task. Word Embeddings are often used as this first building block, and in contrast to that more recent large pre-trained language models such as BERT are purposefully trained on a multi-task setup so you don't need to come up with an extra model in many cases on top of BERT. 

*75.07 seconds*

### **6** Word Representation Learning
Background or big picture information. Let's have a look at the actual Word Representation Learning. So how do we transform our characters into vectors? 

*13.82 seconds*

### **7** Working with Natural Language 
Of course, when we work with natural language, we can draw inspiration from a lot of different peoples, but in this lecture we're going to look at something Ludwig Wittgenstein said and that is "for a large class of cases -though not for all- in which we employ the word "meaning" it can be defined thus: the meaning of a word is its use in the language". So, what can we take from that? Well, we can derive the meaning of a word when we look at how it's used in relation to other words. One thing we also have to know is that the human language is highly ambiguous and context dependent. So for example, "love" can mean a lot of different things in a lot of different contexts, and something that Wittgenstein also said is that language is symbolic, so this is a direct idea that is applied in Word Embeddings. 

*68.43 seconds*

### **8** Representation Learning 
And representation learning as a general field is concerned with learning representations of an object, of an entity, of a word, so that it makes it easier to build upon that when we want to, for example, use classifiers or other machine learning models that do something with the representations. The goal is to create abstract and usable representations that are in a dimensionality that can be used and that also contain the right content. 

*52.36 seconds*

### **9** Word Embeddings 
So let's get to the core of what we want to talk about today: Word Embeddings. Word Embeddings provide typically, not always, but typically a dense vector representation of words and common dimensions are 100 to 300, and those dimensions are abstract, so we can't say that dimension  number 25 contains a certain type of meaning. No. The dimensions are abstract and the vectors only make sense in relation to the other vectors in the vector space. And to measure the relations we can operate with mathematical operations in the vector space. So we can compute nearest neighbors to get the semantic relatedness of words that are close together in the space. Or we can create analogies, more on that in a second. And then, word embeddings, so the major factor why word embeddings are useful is that they can be pre-trained in an unsupervised way on huge text data sets such as Wikipedia or CommonCrawl, which is a web crawl data set that just contains general web text. Or, which is very good for word embeddings, instead you can just train them yourself from scratch on domain specific data. And then we can take this unsupervised pre-trained model and fine-tune it inside another model. What makes word embeddings also super nice to use is that they have a very simple data structure, so they basically are a dictionary or hash map between a string, meaning a word, and a floating point array. 

*131.24 seconds*

### **10** Word Embeddings 
For word embeddings, we have many unsupervised creation methods. I'd try to classify them here for you. So we have a 1-Word-1Vector type of class which includes Word2Vec, Glove and other specialized variants of the two. Then we have  1-Vector+Char-n-grams that can be summed up to one vector per one word, which is FastText which is based on Word2Vec. And then we have another completely different class of word representations, which are the Contextualized / context dependent models that have a much more complex structure that contain many layers where a vector is dependent on the context of a word in every situation, and this includes ElMo and Transformer models such as BERT. We'll talk more about those models in the second neural IR lecture, but now let's focus on the basics. I also want two to give a shout out to the very nice text processing and text modeling library Gensim which contains a lot of training and fine-tuning code for the simple word embedding models. 

*98.59 seconds*

### **11** Unsupervised Training: Language Modelling 
So how do we actually do unsupervised training? Unsupervised training basically means that we don't have explicit labels. That means unsupervised. But it's not like the model just learns out of thin air. No. We use real text, so how people use the language coming back to the Ludwig Wittgenstein analogy, and we model that. So in language modelling the basic task is to predict the next word given a sequence of words. So basically you want to write out a sentence. And this allows us to compute the loss based on the probability of this. So the prediction probability of the next word over a the whole vocabulary. And with that, because in our training data we actually know what should be the next word we can compute the loss, get an error and back propagate that error through the network to train the model.This language modelling is the main technique for text pre-training. Well, I just told you about one task here, but of course there are many variants to this task, such as "we want to predict context words that are around our word" like the previous ones and the next ones, or something that BERT, for example users is a "masked language model" where you get a full sequence and you masked out certain words and you want to predict those masked words that don't necessarily have to be the last one. 

*120.14 seconds*

### **12** Word2Vec 
So let's talk about the famous Word2Vec algorithm. Word2Vec basically, trains a 1 hidden layer neural network to predict context words, via language modelling. And then, well, you might think "Hm...but that's kind of not the task of learning word embeddings" we never actually tell the Word2Vec network that it need to learn word embeddings. But how we kind of do that as a secondary output is: the target words are encoded via 1-hot encoding. Meaning everything we have, an input vector, the size of the vocabulary and this input vector has exactly 1 value at the position of the word in the vocabulary that we want to look at. With that we can input and output words. We can input words and output the probability of words and what we do is like the crucial step in the Word2Vec architecture is that we harvest the word vectors from the network. So here you can see in this picture we take the first matrix of the two of the input matrix of the hidden layer basically, and we take it out. Right? And now, because we use the 1-hot encoding, each row in this matrix corresponds to the 1-hot position of a word. So we can just index the word to that position. The rest of the the network is ignored. Once we got the word embeddings out,the output matrix is ignored, the whole task around it gets discarded, we don't care about it anymore. 

*136.6 seconds*

### **13** Word2Vec 
There are actually two variants of Word2Vec and Word2Vec is sort of only the umbrella term. First, we have the Continuous Bag Of Words (CBOW) which predicts a word given its surrounding context words and then, the other way around, is Skip-gram, which predicts context words given a single word. Both methods have been shown to produce kind of the same or similar results, so it doesn't really matter which one we use, and it's just a difference in training, but the word embeddings we get out of it are very similar. We have to note that even though if you Google for it, it kind of seems like Word2Vec invented word vectors, which it did not. So of course there have been a lot of other techniques before it, but the difference is that Word2Vec rose to a lot of fame and got very influential because it's fast to train. So it contains a couple of contributions that make the training much more efficient. Which means that everyone can train on domain specific data from scratch, which is very nice. Funny enough, this whole idea changes in the current years. It's going back to leaning to download huge models. And of course, we have to say it, Word2Vec just sounds a lot better than other algorithms like LDA or SVD, which also do some sort of word vector and word embedding, but Word2Vec just sounds cool. Word2Vec is also not the state of the art anymore, so it has been in the state of the art when it was proposed, but not anymore. But it's still good enough for practical use in many scenarios, and combined with the efficiency it is used a lot in practice and that's why we're talking about it. So if you want to know more details, you can also look up those resources in the bottom there. 

*146.63 seconds*

### **14** Word2Vec - Training 
Just to give an overview of how training in Word2Vec work is that we take a full text in one big batch of text, so we don't care about sentence boundarie, we don't care about document boundaries were just concatenates all our full text together and then we move over that text with a sliding window of context words and words to predict. You can see such an example in the figure on the right here where we have an input source. It's just some bunch of words. And then, we select the word sentence as our main word, take the predicting or the context words to predict around it and we create so called window samples where we create pairs between our main word and each of the single window words. So you can see. Then we kind of also discard the distance between our main and the windowed words. Then, we look up our 1-hot encoding from our vocabulary position, and we feed each sample individually into the network so we always operate on pairs of words. We use the 1-hot encoding as an input an with a softmax output of the output layer, we get the probability of the neighboring words in our language model. And with that, we can compute negative log likelihood loss. The problem here is that this softmax needs to cover all terms in the vocabulary to get the probability over all terms, which is too costly because the vocabulary can reach millions of words. So Word2Vec introduced negative sampling of random terms to get rid of the problem that we actually have to look at all terms in document. I'm not going to go into more detail on that, but if you're interested in the mathematical aspect and how to derive all the formulas, please check out the Stanford Lecture on exactly that topic that only focuses on the mathematical of Word2Vec. 

*168.55 seconds*

### **15** Offshoots based on the …2Vec Idea 
And because as we said, the marketing of 2Vec is great, there have been a lot of offshoots based on this whole "take the context in a sequence and do unsupervised prediction based on the sequence". It has been applied to, of course, sentence, paragraph and document embeddings, but also, graph embeddings, entity embeddings and everything really with a sequence such as playlists or so. Of course there have been a lot of small and incremental adjustments and improvements to the original Word2Vec idea and now, we're going to take a look at one of those. 

*49.53 seconds*

### **16** FastText 
And this is FastText. So FastText improves on the Vord2Vec idea by using instead of whole words, workpieces, or character n-grams as a vocabulary. And character n-grams here means that we take n-characters at a time. So for example, we can take 3-grams, which means we take 3 characters at a time in a window of the word. With that we get a single vector for each character n-gram, and those vectors are then added together to form the single word vector, which has the following benefits. We have no more out-of-vocabulary words, because we can always just substitute it for a character n-gram. Then, we don't have a problem of low frequency words or the problem of low frequency words is much less, because low frequency words are made up of better known character n-grams, so we get better representations for those. It's very good for compound-heavy languages like German, which, for basically the same word, would need multiple word embeddings and, each time you add a compound you get a new entry in your vocabulary, but not so with FastText. It kind of offers similar performance and usability as Word2Vec. So we can still generate a single vector per term in our vocabulary and we can apply all the analysis tools and methods that have been developed for Word2Vec on FastText as well. And it comes with a very nice library and pre-trained models for virtually all languages. 

*141.78 seconds*

### **17** Similarity Measurements 
So I briefly talked about analysis of Word2Vec. What are those analysis based upon? Mostly, they are based upon some sort of similarity measure. So a word embedding is a vector space in which we can do math and mathematical operations, and a very common operation is to measure the distance between two points. In a word embedding this measuring the distance corresponds to measuring the similarity of words. A very standard measure for that is the cosine similarity, which measures only the direction and not the magnitude of the two points. It's implemented as the dot product of normalized vectors with length one, but the problem here is, you can always visualize cosine similarity in a 2 dimensional space, right? But it's very hard to visualize cosine similarity in a 300 dimensional spac, because here we can hav crazy things that you can't really visualize which is, for example, we can have so-called hub-vectors that are close to many others, but the others don't necessarily have to be close to each other. And the Gensim library implements a lot of those similarity computations for you. 

*102.66 seconds*

### **18** Analogies 
Another common tool that's especially suited for non technical audiences is to probe the relationships between more than two words with analogies. So we want to say A to B is as C to D, which can be reformulated as A - B + C = D. And it's kind of easy to understand, right? A very common example is King - Man + woman = Queen. But it also works for heads of state, capitals of countries, etc. It might not be as easy as it seems, so the analogies are an oversimplification, and they can be quite fragile between different runs, so when you have different initializations or small changes to the algorithm, those analogies can break apart and not show what you want him to show, and furthermore... 

*74.12 seconds*

### **19** Analogies – The Devil is in the Details 
... The devil is in the details, right? So I just showed you the "King - Man + Woman = Queen"-example, but it turns out it might not be true. So in May 2019, a surprise finding takes the Twittersphere by surprise. For some reason, a lot of people just reused the same library code that makes those analogy computations. But this popular library code, hardcoded that you cannot have the same word as result in your analogy. So when actually King - Man + Woman goes back to King, it wouldn't output King, but the second best thing. That kind of is a problem, so you can see that Ian Goodfellow, a very famous but general machine learning personality, tweeted about that. Followed by an answer from a also famous personality, Yoav Goldberg who wrote the book I recommended you. But he is a very NLP person and he is kind of shocked that no one knew this, which is fun. Because of course if you work with the library, you might know it, but if you don't look at the code, you could easily mistake it, which becomes a problem when you want to show bias in the word embedding and you kind of hardcode the correct answer out of the analogy. So always be careful when you use code that's not yours and which is highly crucial to your study: you should really know what you're executing there and not just blindly take something.  

*133.56 seconds*

### **20** Limitations: Word Ordering or N-Grams 
Another limitation of those 1-Vector-1-Word word embeddings, is that they don't actually pay attention to the word ordering or the word n-grams in this case. So, those two sentences that you see here : "it was not good, it was actually quite bad" versus "it was not bad, it was actually quite good" would receive the exact same representation based on the word embedding. The context in Word2Vec is only used during training, but not during the usage of the word embeddings. Right, and the problem here is Word2Vec kind of solves that problem by creating sometimes bi or tri-gram vocabulary entries for very popular co-occurring words, but to do that for every single word is not feasible. Because then you end up with two little training data and get no good connection between quite good and very good. If you see them as two distinct terms. But we'll hear more about that in the next lecture. 

*93.62 seconds*

### **21** Limitations: Multiple Senses per Word 
And then, another problem with Word2Vec, which kind of follows the same pattern, is they don't know different senses of the same word, right? So you have this one word, one vector, which is good for analysis but if you don't know the context of the word, you can't know the current sense it should represent. So if you have a word with multiple senses and there are a lot of them and you squash those senses together, you either create an average over all senses, or if they are very imbalanced, the most common sense that curse in the training data will win and dominate the others. And this is especially visible in domain -specific data versus general english-text data. Right, and this is definitely a missed opportunity for improved effectiveness, and that's why contextualized models have been proposed in the last year and beyond. And we will look at them also in the neural IR lectures. 

*92.65 seconds*

### **22** Word Embeddings in IR
Now let's take a look at how you could or a first glimpse at how you could use more embeddings in the information retrieval context. 

*10.91 seconds*

### **23** Query Expansion with Word Embeddings 
And this is with query expansion. So we use word embeddings for query expansion, and we use those word embeddings to expand the search space of a search query with similar words that we get from the word embedding. And in this work from our group from 2016, remind you that its a couple of years old, we updated collection statistics and adapted the relevance model to score a single document with multiple similar words together, so that the words kind of keep the connection between the original query and the similar words that have been expanded. How this works, is you can see this in this plot here you start with a sample query and then, you split the query and for each word you look up the similarity from a word embedding and then you define a threshold of cosine similarity such as 0.7 that those words have to be above, and then you search for all the terms in the inverted index, you gather the posting lists and then you weight each document with the connection based on the weight from the embedding. 

*99.54 seconds*

### **24** Problem: Topic Shifting 
And what we recognized here is a problem, and that's topic shifting. So if you take a word embedding that is trained on Wikipedia context, then words with similar context are close together. But they can be of a different topic, which is of course bad for the retrieval task. So the word Austria has close neighbors that correspond to physical neighbors, right? So Germany, Hungary, etc. And if you search for hotels in Austria, you probably only want results located in Austria and maybe expand the query with towns and cities in Austria, but not in other countries, because then you would search for the different countries. 

*55.75 seconds*

### **25** Retrofitting 
And to kind of solve that or mitigate that problem, we utilized a technique called retrofitting, which is very interesting when you actually incorporate external resources into an existing word embedding by moving the vectors of the existing word embedding in an iterative update function, where each word gets moved by a combination of its original position and similarity values to other words based on external information. So that means that, for example, in this example here you have two terms that are in this radius of neighborhood, but you don't want them there, and the external resource kind of tells you that they shouldn't be in your expansion. So if you apply retrofitting, then you can move the word vectors a little bit away from them so that they're not in the radius anymore. 

*69.4 seconds*

### **26** Topic Shifting – Examples from Wikipedia 
To kind of showcase that, we utilized a Word2Vec skip-gram embedding combined with an embedding from latent semantic indexing or short LSI, and you can see that in the original skip-gram embedding for "austria" you have words that don't belong there. But if you retrofit it with the word "austria" from LSI, you get a clean output. The other way around is also true if you have "austrian", in this example from Wikipedia you now have problems in your LSI inventing, but when you retrofit the two embeddings together, you get a better result overall. You get a better result than if you would use each of them idnividually. 

*64.88 seconds*

### **27** Application to Patent Retrieval 
That's what we did in the area of patent retrieval. So in patent retrieval, the recall is super important. So single relevant patent that is missed in a search session can be very bad, or at least people tell me that, I don't have any patents. Now, if you have a vocabulary mismatch, so you don't directly find your document or your patent even if you click through all results, as there are many domain specific terms in patent, you should expand the queries, right? We expanded the queries with retrofitted word embeddings and we kind of showed that if you use the word embedding alone to expand the query, you do not improve the results, but if you take a combination of the local context, skip-gram Word2Vec combining with this global LSI embedding via retrofitting, you can actually improved the recall significantly in a standard patterns test collection.  

*84.59 seconds*

### **28** Outlook	 
But we have to say, this work that we did, it was from 2018, and of course it had a lot of limitations. So it was an unsupervised method and in the end it kind of had a limited effectiveness and only increased the recall in the patent scenario. But since 2018, a lot of stuff happened in the field of neural information retrieval and we have kind of a recipe book now, with ingredients for better approaches. So we are using supervised learning with large retrieval datasets on top of unsupervised pre-training. Then, we have models with way more parameters to learn, then we did in 2018. We're using contextualization and we operate on the full text of query and documents. I think this is all very exciting and I hope you tune in to the next lectures where we're going to look at all those ingredients for neural re-ranking approaches and more.  

*75.88 seconds*

### **29** Summary: Word Embeddings 
OK, so, what I want you to take away from this talk is that we need to represent words as vectors instead of their character values with word embeddings. Unsupervised pre-training is a major strength of word embeddings and still also in more advanced models unsupervised pre-training is the key to success. Word embeddings have many potential applications in natural language processing but also in information retrieval such as query expansion which I just showed you. 

*43.65 seconds*

### **30** Thank You  
Well, thank you very much for your attention and I hope you tune in next time. See you then. 

*9.22 seconds*

### Stats
Average talking time: 84.09262083333333

# Lecture 4 - Word Representation Learning

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Hello everyone, today we're going to take a look at our first machine learning class and that is about word representation learning.  

*12.38 seconds*

### **2** Today 
We are going to look at two specific things. First, are word embeddings. So how we actually transform words as we see it from characters to vectors so that neural networks can work with them and they are a very popular technique from the last years is WordVec and we're going to take a look at that as well. And then we're going to take a first glimpse at what you could do with word embeddings in the information retrieval setting and we're going to look at unsupervised query expansion and how our group actually solved problem of topic shifting that occured from pre trained word embeddings. So without further ado, let's get started. 

*51.59 seconds*

### **3** Differentiable Matrices & Transforms  
An important motivation for word embeddings is the fact that if we want to work with neural networks and we want to apply a gradient descent training algorithm, we need to operate in continuous spaces, so that means we need to have tensors with floating point values and continuous transformation functions that are differentiable. And without that we don't have a gradient, so we can't update our network parameters in the right "direction", because we just don't know in which direction we should go to improve our result. So this means we can't just input the character values, for example, the ASCII codes of words in a linear algebra network and expect it to work because it wouldn't. So we need a method to map chars or word pieces or words as a whole defined by some tokenization to some sort of vector. And to do that, we have a lot of options and we can't show every option today, but will look at a few. And yeah, you can go and look up some more. Before we continue I want to stress that you should take a look at the PY torch beginner's guide to understand various terminologies and basic ideas of how to train a neural network. I'm not going to repeat that in this lecture. 

*115.43 seconds*

### **4** First Pre-training then Fine-tuning  
A very common problem, and therefore practice how to deal with it in all kinds of machine learning projects, is we often don't have a lot of data because labeling data is most of the time pretty expensive or you need a big user base to create noise free click data or so. So everything when we work with language, we have the problem that we want good word vectors or word representations for each individual word. But there are a lot of words in the English language so we can't cover all words in a fine tuning step where we only have our own data - our task specific data. And this means that those low frequency terms that don't or very little appear in our data could be a great indicator for your task. That's especially true in the retrieval domain. But they probably get bad representations from our little training data. So what does the community do in that case? We pre-train word representations on a general text corpus unsupervised, where we don't actually need label data. And then we take that pre-trained parameters and put them in our model to fine tune them so we train them with our own limited, we continued to train them with our own limited data. And this idea is a major factor in the advances of word embeddings and language models in natural language processing, in the last couple of years. 

*130.19 seconds*

### **5** Take Pre-trained then Stack together 
And of course we can take the pre-trained embedding and not only continue to use that exact pre-trained embedding, but we can stack it together like Lego blocks with other stuff. So this is very common in the computer vision domain, but it's getting better and better in text domain as well. So for example, we can take our pre trained module like a word embedding and put it in a bigger module in a bigger model where the rest of the model that utilizes the word embeddings is randomly initialized and only trained in the fine tuning task. And word embeddings are often used as this first building block, and in contrast to that more recent large pre-trained language model such as BERT are purposefully trained on a multi-task set up so you don't need to come up with an extra model in many cases on top of BERT. 

*75.07 seconds*

### **6** 
Round or big picture information. Let's have a look and the actual word representation learning. So how do we transform our characters into vectors? 

*13.82 seconds*

### **7** Working with Natural Language 
Of course, when we work with natural language, we can draw inspirations from a lot of different peoples. But in this lecture, we're going to look at something Ludwig Wittgenstein said and that is for a large class of cases, though not for all in which we employ the word meaning it can be defined thus: the meaning of a word is its use in the language. So what what can we take from that? Well, we can derive the meaning of a word when we look at how it's used in relation to other words. And one thing we also have to know is that the human language is highly ambiguous and context dependent. So for example, love can mean a lot of different things in a lot of different contexts. And something that LUdwig Wittgenstein also said is that language is symbolic, so this is a direct idea that is applied in word embeddings. 

*68.43 seconds*

### **8** Representation Learning 
And Representation Learning, as a general field is concerned with learning representations of an object, of an entity, of word. So that it makes it easier to build up on that when we want to, for example, use classifiers or other machine learning models that do something with the representations. And the goal is to create abstract and usable representations that are in a dimensionality that also contain the right content. 

*52.36 seconds*

### **9** Word Embeddings 
So let's get to the core of what we want to talk about today - word embeddings. Word embeddings provide typically, not always, but typically a dense vector representation of words and common dimensions are 100 to 300. And those dimensions are abstract, so we can't say that dimension #25 contains a certain type of meaning. No, the dimensions are abstract and the vectors only make sense in relation to the other vectors in the vector space. And to measure the relations we can operate with mathematical operations in the vector space. So we can compute nearest neighbors to get the semantic relatedness of words that are close together in the space. Or we can create analogies, more on that in a second. And then word embeddings, so the major factor why were embeddings are useful is that they can be pre-trained in an unsupervised way on huge text data sets such as Wikipedia or CommonCrawl, which is a web crawl data set that just contains general web text. Or which is very good for word embeddings is that you can just train them yourself from scratch on domain specific data and then we can take this unsupervised pre-trained model and fine-tuned inside another model. And what makes word embeddings also super nice to use is that they have a very simple data structure so they basically are a dictionary or hash map between a string, meaning a word, and floating point array. 

*131.24 seconds*

### **10** Word Embeddings 
For embeddings we have many unsupervised creation methods. And I try to classify them here for you. So we have a one word equals 1 vector type of class which includes Word2Vec, Glove and other specialized variants of the 2. Then we have one vector or character-n-gram that can be summed up to one vector per one word, which is FastText which is based on Word2Vec. And then we have another completely different class of word representations, which are the contextualized, context dependent models that have a much more complex structure that contain many layers where a vector is dependent on the context of a word in every situation. And this includes ELMo and transformer models such as BERT. And we'll talk more about those models in the second neural IR lecture, but now let's focus on the basics, and I also want to give a shout out to the very nice text processing and text modeling library gensim which contains a lot of training and fine tuning code for the simple word embedding models. 

*98.59 seconds*

### **11** Unsupervised Training: Language Modelling 
So how do we actually do unsupervised training? Unsupervised training basically means that we don't have explicit labels. That means unsupervised, but it's not like the model just learns out of thin air. No, we use real text, so how people use the language coming back to the Ludwig Wittgenstein analogy and we model that. So in language modeling, the basic task is to predict the next word given a sequence of words. So basically you want to write out a sentence. And this allows us to compute the loss based on the probability of this, sort of prediction probability, of the next word over the whole vocabulary. And with that, because in our training data we actually know what should be the next word, we can compute the loss, get an error and back propagate that error through the network to train the model. And this language modeling is the main technique for text pre-training and well I just told you about 1 task here, but of course there are many variants to this task, such as we want to predict context words that are around our word, like the previous ones and the next ones, or something that BERT, for example, uses is a mask language model where you get a full sequence and you mask out certain words and you want to predict those mask words that don't necessarily have to be the last one. 

*120.14 seconds*

### **12** Word2Vec 
So let's talk about the famous Word2Vec algorithm. Word2Vec basically, trains a one hidden layer neural network to predict context words, real language modeling. And then, well, you might think "but that's kind of not the task of learning word embeddings", we never actually tell the Word2Vec network that it needs to learn word embeddings, but how we kind of do that as a secondary output is the target words are encoded via one hot encoding, meaning everything we have a an input vector the size of the vocabulary. And his input vector has exactly one value at the position of the word in the vocabulary that we want to look at. And with that we can input and output words, we can input words and output the probability of words. And what we do as like the crucial step in the Word2Vec architecture is that we harvest the word vectors from the network. Here you can see in this picture we take the first matrix of the two of the input matrix of the hidden layer, basically and we take it out, righ? And now, because we used the 1-hot-encoding, each row in this matrix corresponds to the 1-hot position of a word, so we can just index the word to that position and the rest of the network is ignored. Once we got the word embeddings out, the output matrix is ignored, the all taks around it gets discarded, we don't care about it anymore. 

*136.6 seconds*

### **13** Word2Vec 
There are actually two variants of Word2Vec and Word2Vec is sort of only the umbrella term. First, we have the continuous bag of words which predicts a word given its surrounding context words, and then the other way around is Skip-gram, which predicts context words given a single word. And both methods have been shown to produce kind of the same or similar results, so it doesn't really matter which one we use, and it's just a difference in training, but the word embeddings we get out of it are very similar. And well, we have to note that even though if you Google for it, it kind of seems like Word2Vec invented word vectors which it did not. So of course there have been a lot of other techniques before it, but the difference is that Word2Vec rose to a lot of Fame and got very influential because its fast to train so it contains a couple of contributions that make the training much more efficient, which means that everyone can train on domain specific data from scratch, which is very nice. And funny enough this whole idea changes in the current years, it's going back to leaning to download image models. And of course we have to say it - Word2Vec just sounds a lot better than other algorithms like LDA or SVD, which also do some sort of word vector and word embedding. But Word2Vec just sounds cool. And Word2Vec is also not the state of the art anymore, so it has been in the state of the art when it was proposed, but not anymore. But it's still good enough for practical use in many scenarios. And combined with the efficiency it is used a lot in practice and that's why we're talking about it. So if you want to know more details, you can also look up those resources in the bottom there. 

*146.63 seconds*

### **14** Word2Vec - Training 
And just to like give an overview of how training in Word2Vec works is that we take a full text in one big batch of text. So we don't care about sentence boundaries, we don't care about document boundaries, we just concatenate all our full text together and then we move over that text with a sliding window of context words and words to predict. And yeah, you can see such an example in the finger on the right here, where we have an input source that's just some bunch of words. And then we select the word sentence as our main word, take the predicting on the context words to predict around it, and we create window so called Windows samples where we create pairs between our main word and each thing of the single window words. So you can see, then we kind of also discard that distance between our main and the window words. Then we look up our 1-hot encoding from our vocabulary position and we feed each sample individually into the network so we always operate on pairs of words. We used 1-hot encoding as an input and with a softmax output of the output layer, we get the probability of the neighboring words in our language model. And with that we can compute a negative log likelihood loss. And the problem here is that this softmax  needs to cover all terms in the vocabulary to get the probability overall terms and which is too costly because vocabulary can reach millions of words. So Word2Vec introduced negative sampling of random terms to get rid of the problem that we actually have to look at all terms in document. And I'm not going to go into more detail on that, but if you're interested in the mathematical aspect and how to derive all the formulas, please check out the Stanford lecture on exactly that topic that only focuses on the mathematical of Word2Vec. 

*168.55 seconds*

### **15** Offshoots based on the ...2Vec Idea 
And because, as we said, the marketing of 2Vec is great, there have been a lot of offshoots based on this whole take - the context in sequence and do unsupervised prediction based on the sequence. And it has been applied to, of course, sentence, paragraph and document embeddings. But also, graph embeddings, entity embeddings and everything really with the sequence such as playlists or so. And of course there have been a lot of small and incremental adjustments and improvements to the original Word2Vec idea. And now we're going to take a look at one of those. 

*49.53 seconds*

### **16** FastText 
And this is FastText, so FastText improves on the Word2Vec idea by using, instead of whole words, word-pieces, or character n-grams as a vocabulary. And character n-grams here means that we take N characters at a time. So for example, we can take 3 grams, which means we take 3 characters at a time in a window of the word. And with that we get a single vector for each character n-gram and those vectors are then added together to form the single word vector. Which has the following benefits, so we have no more out of vocabulary words because we can always just substitute it for a character n-gram. Then we don't have a problem of low frequency words or the problem of low frequency words is much less. Because low frequency words are made up of better known character n-grams, so we get better representations for those. And it's very good for compound-heavy languages like German, which for, basically the same word would need multiple word embeddings and each time you add a compound, you get a new entry in your vocabulary, but not so with FastText. And it kind of offers similar performance and usability as Word2Vex, so we can still generate a single vector per term in our vocabulary. And we can apply all the analysis tools and methods that have been developed for Word2Vec on FastText as well. And it comes with a very nice library and pre-trained models for virtually all languages. 

*141.78 seconds*

### **17** Similarity Measurements 
So I briefly talked about analysis of Word2Vec. So what are those analysis based upon. And mostly there are based upon some sort of similarity measure. So a word embedding is a vector space in which we can do math and mathematical operations, and a very common operation is to measure the distance between two points. And in a word embedding this measuring the distance corresponds to measuring the similarity of words. A very standard measure for that is the cosine similarity, which measures only the direction and not the magnitude of the two points. And it's implemented as the dot product of normalized vectors with length one. But the problem here is it's, so you can always visualize cosine similarity in a 2 dimensional space, right? But it's very hard to visualize cosine similarity in a 300 dimensional space. Because here we can have crazy things that you can't really visualize. For example, we can have so called hub-vectors that are close to many others. But the others don't necessarily have to be close to each other. And the gensim library implements a lot of those similarity computations for you. 

*102.66 seconds*

### **18** Analogies 
And another common tool that's especially suited for non technical audiences is to probe the relationships between more than two words with analogies. So we want to say A to B is as C to D, which can be reformulated as A - B + C = D. And it's kind of easy to understand, right? They're very common example is King - Man + Woman = Queen, but it also works for heads of state, capitals of countries etc etc. Well, it might not be as easy as it seems, so the analogies are an oversimplification and they can required fragile between different runs. So when you have different initializations or small changes to the algorithm, those analogies can break apart and not show what you want them to show, and furthermore...

*74.12 seconds*

### **19** Analogies - The Devil is in the Details 
...the devil is in the details, right? So I just showed you the King - Man + Woman = Queen example, but it turns out it might not be true. So, in May 2019 a surprise finding takes the Twittersphere by surprise. For some reason, a lot of people just reuse the same library code that makes those analogy computations. But this popular library code hardcoded that you cannot have the same word as result in your analogy. So when actually King - Man + Woman goes back to King, it wouldn't output King, but the second best thing. And, yeah, that kind of is a problem. So here you can see that Ian Goodfellow, a very famous but General Machine Learning Personality, tweeted about that, followed by an answer from a also famous personality, Yoav Goldberg who wrote the book I recommended you, but he is a very LP person and he is kind of shocked that noone new this which is fun. Because of course, if you work with the library, you might know it, but if you don't look at the code, you could easily mistake it, which becomes a problem when you want to show bias in the word embedding and you kind of hard code the correct answer out of the analogy. So always be careful when you use code that's not yours and which is highly crucial to your study. You should really know what you're executing there and not just blindly take something.  

*133.56 seconds*

### **20** Limitations: Social Biases 
And on the subject of social biases, yes, even though in this study it has been shown that the analogies and the computation of the analogies has been a problem in like highlighting social biases, we still know that taking biased text as input produces biased representations. Bias can take many forms such as gender or racial bias. And it easily can affect downstream task without anyone ever knowing. So if you want to use word Embeddings for hiring decisions, predictive policing or recommendation algorithms that don't show the work of minorities, for example, because the words that describe them are associated with a more negative way or missing in the training data etc etc. So there have been various methods proposed to proactively debias word embeddings after they have been trained. But the problem is and as has been shown in the paper linked below that those methods might not be truly effective and only cover up the bias, so this is an active field of research where we still have to put in a lot of energy in time to figure it out how to improve this problem. 

*98.68 seconds*

### **21** Limitations: Word Ordering or N-Grams 
Another limitation of those one vector for one word word embeddings is that they don't actually pay attention to the word ordering or the word n-grams in this case. So those two sentences that you see here, "it was not good, it was actually quite bad", versus "it was not bad, it was actually quite good" would receive the exact same representation based on the word embedding, so the context in Word2Vec is only used during training, but not during the usage of the word embeddings. Right and the problem here is so Word2Vec kind of solves that problem by creating sometimes bi and tri gram vocabulary entries for very popular co-occurring words. But to do that for every single word is not feasible because then you end up with two little training data and get no good connection between "quite good" and "very good". If you see them as two distinct terms. But we'll hear more about that in the next lecture. 

*93.62 seconds*

### **22** Limitations: Multiple senses per word 
And then another problem with Word2Vec is they don't, which kind of follows the same pattern, they don't know different senses of the same word, right? So you have this one word, one vector, which is good for analysis but it if you don't know the context of the word, you can't know the current sense it should represent. So if you have a word with multiple senses and there are a lot of them and you squash those senses together, you either create an average over all senses, or if they are very imbalanced, the most common sense that occurs in the training data will win and dominate the others. And this is especially visible in domain specific data versus general English text data. Right, and this is definitely a missed opportunity for improved effectiveness, and that's why contextualized models have been proposed in the last year and beyond. And we will look at them also in the neural IR lectures. 

*92.65 seconds*

### **23** 
Now let's take a look at how you could, or a first glimpse at how you could use word embeddings in the information retrieval context. 

*10.91 seconds*

### **24** Query Expansion with Word Embeddings 
And this is with Query Expansion. So we use word embeddings for query expansion and we use those word embeddings to expand the search space of a search query with similar words that we get from the word embedding. And in this work from our group from 2016, mind you that's a couple of years old, we updated collection statistics and adapted the relevance model to score a single document with multiple similar words together, so that the words kind of keep the connection between the original query and the similar words that have been expanded. And how this works is, you can see this in this plot here, you start with a sample query and then you split the query and for each word you look up the similarity from a word bedding and then you define a threshold of cosine similarity such as 0.7 that those words have to be above. And then you search for all the terms in the inverted index, you gather the posting lists and then you weight each document with the connection based on the weight from the embedding. 

*99.54 seconds*

### **25** Problem: Topic Shifting 
And what we recognized here is a problem, and that's topic shifting. So if you take a word embedding that is trained on Wikipedia context, than words with similar context are close together. But they can be of a different topic, which is of course bad for the Retrieval Task. So the word Austria has close neighbors that correspond to physical neighbors, right? So Germany, Hungary, etc. And if you search for hotels in Austria, you probably only want results located in Austria and maybe expand the query with towns and cities in Austria. But not in other countries, because then you would search for the different countries. 

*55.75 seconds*

### **26** Retrofitting 
And to kind of solve that or mitigate that problem, we utilized a technique called retrofitting, which is very interesting where you actually incorporate external resources into an existing word embedding by moving the vectors of the existing word embedding in an iterative update function. What each word gets moved by a combination of its original position and similarity values to other words based on external information. That means that, for example, in this example here you have two terms that are in this radius of neighborhood, but you don't want them there, and the external resource kind of tells you that they shouldn't be in your expansion. So, if you apply a retrofitting and then you can move the word vectors a little bit away from them so that they are not in the radius anymore. 

*69.4 seconds*

### **27** Topic Shifting - Examples from Wikipedia 
And. To kind of showcase that, we utilized a Word2Vec Skip-gram embedding, combined with an embedding from latent semantic indexing or short LSI, and you can see that in the original Skip-gram embedding for Austria you have words that don't belong there. But if your retrofit it with the word Austria from LSI, you get a clean output and the other way around is also true. If you have Austrian, in this example from Wikipedia, you now have problems in your LSI embedding. But when you retrofit the two embeddings together, you get a better result overall. And, yeah. And you get a better result than if you would use each of them individually.

*64.88 seconds*

### **28** Application to Patent Retrieval 
And that's what we did in the area of pattern retrieval. So in pattern retrieval the recall is super important. A single relevant patent that it's missed in a search session can be very bad, well, or at least people tell me that. I don't have any patents. No, if you have a vocabulary mismatch so you don't directly find your document or yout pattern, even if you click through all results as there are many domain specific terms in patterns. You should expand the queries right? and we expanded the queries with retrofitted word embeddings and we kind of show that if you use the word embedding alone to expand the query, you do not improve the results, but if you take a combination of the local context, skip-gram Word2Vec embedding, combining with this global LSI embedding via retrofitting, you can actually improve the recall significantly. In a standard patent test collection.  

*84.59 seconds*

### **29** Outlook	 
Yeah, but we have to say, Well, this work that we did was from 2018 and of course it had a lot of limitations so it was an unsupervised method and in the end it kind of had a limited effectiveness and only increase the recall in the patent scenario. But since 2018 a lot of stuff happened in the field of neural information retrieval and we have kind of a recipe book now with ingredients for better approaches. So we are using supervised learning with large retrieval datasets on top of unsupervised pre-training. Then, we have models with way more parameters to learn then we did in 2018. We are using contextualization, and we operate on the full text of query and documents. And well, I think this is all very exciting and I hope you tune into the next lectures where we're going to look at all those ingredients for Neural reranking approaches and more.  

*75.88 seconds*

### **30** Summary: Word Embeddings 
OK. So, what I want you to take away from this talk is that we need to represent words as vectors instead of their character values with word embeddings. And unsupervised pre-training is a major strength of word embeddings and still also in more advanced models, unsupervised pre training is the key to success. And word embeddings have many potential applications in natural language processing but also in information retrieval such as query expansion which I just showed you. 

*43.65 seconds*

### **31** Thank You  
Well, thank you very much for your attention and I hope you tune in next time. See you then. 

*9.22 seconds*

### Stats
Average talking time: 84.56333669354841

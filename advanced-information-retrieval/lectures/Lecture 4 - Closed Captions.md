# Lecture 4 - Word Representation Learning

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Hello everyone, today we're going to take a look at our first machine learning class and that is about word representation learning.  

*12.38 seconds*

### **2** Today 
We are going to look at two specific things. First, are word embeddings. So how we actually transform words as we see it from characters to vectors so that neural networks can work with them and they are a very popular technique from the last years is work back and we're going to take a look at that as well and then. We're going to take a first glimpse at what you could do with word embeddings in the information which we all setting an we're going to look at unsupervised query expansion and how our group actually solved problem of topic shifting that occured from pre trained word openings. So without further ado, let's get started. 

*51.59 seconds*

### **3** Differentiable Matrices & Transforms  
An important motivation for word embeddings is the fact that if we want to work with neural networks and we want to apply a gradient decent training algorithm. We need to operate in continuous spaces, so that means we need to have tensors with floating point values and continuous transformation functions that are differentiable. And without that we don't have a gradient, so we can't update our network parameters in the right quote, Unquote direction, because we just don't know in which direction we should go to improve. Our result. So this means we can't just input the the character values. For example, the ASCII codes of words in a linear algebra network and expect it to work because it wouldn't. So we need. A method to map chars or word pieces or words as a whole. Defined by some tokenization to some sort of vector, and to do that, we have a lot of options and. We can't show every option today, but will look at a few. And yeah, you can go and look up some more before we continue. I want to stress that you should take a look at the PY torch. Beginner's guide to understand various. Terminologies and basic ideas of how to train a neural network. I'm not going to repeat that in this lecture. 

*115.43 seconds*

### **4** First Pre-training then Fine-tuning  
A very common problem, and therefore practice how to deal with it in all kinds of machine learning projects, is. We often don't have a lot of data because labeling data is most of the time pretty expensive or you need a big user base to create noise. Free click data or so. So everything when we work with language. We have the problem that we want. Good word vectors or word representations for each individual word. But there are a lot of words in the English language so. We can't cover all words in a fine tuning step where we only have our own data. Our task specific data and. This means that those low frequency terms that don't or very little appear in our data. Could be a great indicator for your task. That's especially true in the retrieval domain. But they probably get bad representations from our little training data. So. What does the community do in that case? We pre train word representations on a general text corpus. Unsupervised, where we don't actually need label data and then we take that. Pre trained Parameters and put them in our model to fine tune them so we train them with our own limited. There are continued to train them with our own limited data. And this idea is a major factor in the advances of morning Bannings and language models in natural language processing. In the last couple of years. 

*130.19 seconds*

### **5** Take Pre-trained then Stack together 
And of course we can take the pre trained embedding and not only. Continue to use that exact pre trained embedding, but we can stack it together. Like Lego blocks. So we can. So this is very common in the computer vision domain, but it's getting better and better in text domain as well. So for example, we can take our pre trained module like a word embedding and put it in a bigger module in a bigger model where the rest of the model that utilizes the word embeddings is randomly initialized and only trained in the fine tuning task. And word embeddings are often used as this first building block, and in contrast to that more recent large pre trained language model such as bird are purposefully trained on a multi task set up so you don't need to. Come up with an extra model in many cases on top of bird. 

*75.07 seconds*

### **6** 
Round or big picture information. Let's have a look and the actual word representation learning. So how do we? Transform our characters into vectors. 

*13.82 seconds*

### **7** Working with Natural Language 
Of course, when we work with natural language, we can draw inspirations from a lot of different peoples. But in this lecture we're going to look at something Ludwig Wittgenstein said and that is for a large class of cases, though not for all in which we employ the word meaning it can be defined. Thus, the meaning of a word is its use in the language. So. What what can we take from that? Well, we can derive the meaning of a word when we look at how it's used in relation to other words. And one thing we also have to know is that the human language is highly ambiguous and context dependent. So for example, love can mean a lot of different things in a lot of different contexts, and something that will be good canstein also said is that language is symbolic, so this is a direct. Idea that is applied important paintings. 

*68.43 seconds*

### **8** Representation Learning 
And Representation A. Is concerned with. Learning representations of. City of a word, so that it makes it easier to. Build up on that when we want to, for example, use classifiers or other machine learning models that do something with the representations and the goal is to create abstract. Abstract in usable representations that. Are in a dimensionality that can be used in that are in. That also contain the right content. 

*52.36 seconds*

### **9** Word Embeddings 
So let's get to the core of what we want to talk about today. Word embeddings word embeddings provide typically. Not always, but typically a dense vector representation of words and common dimensions are 100 to 300. And those dimensions are abstract, so we can't say that dimension #25 contains a certain type of meaning. No, the dimensions are abstract and the vectors only makes sense in relation to the other vectors in the vector space. And to measure the relations we can operate with mathematical operations in the vector space. So we can. Compute nearest neighbors to get the semantic relatedness of words that are close together in the space. Or we can create analogies more on that in a second. And then more embeddings. So the major factor why were embeddings are useful is that they can be pre trained in an unsupervised way on huge text data sets such as Wikipedia or common crawl, which is a. Web crawl. Data set just contains general web text. Or which is very good for word embeddings is that you can just train them yourself from scratch on domain specific data and then we can take this unsupervised pre trained model and find unit inside another model. And what makes word embeddings also super nice to use is that they have a very simple data structure so they basically are a dictionary or hash map between a string meaning a word and floating point array. 

*131.24 seconds*

### **10** Word Embeddings 
Forward embeddings, we have many unsupervised creation methods. Um? And I try to classify them here for you. So we have a one word equals 1 vector type of class which includes work back laugh and other specialized variants after 2 then we have one vector. Or character Engram that can be summed up to one vector per one word, which is fast text which is based on more track. And then we have another completely different class of word representations, which are the contextualized context dependent models that have a much more complex structure that contain many layers where a vector is dependent on the context of a word in every situation, and this includes Elmo and transformer models such as Bird and. We'll talk more about those models in the second neural IR lecture, but now let's focus on the basics, and I also want to give a shout out to the very nice. Text processing and text modeling library gensym which contains a lot of training and fine tuning code for the simple word embedding models. 

*98.59 seconds*

### **11** Unsupervised Training: Language Modelling 
So how do we actually do unsupervised training? Unsupervised training mean basically means that we don't have explicit labels. That means unsupervised, but it's not like the model just learns out of thin air. No, we use real text. So how people use the language coming back to the Ludwig Wittgenstein analogy? And we model that. So in language modeling, the basic task is to predict the next word given a sequence of words. So basically you want to write out a sentence. And this allows us to compute the Los based on the probability of this sort of prediction probability of the next word over the whole vocabulary and with that. Because in our training data we actually know what should be the next word we can. Compute loss. And back propagate that error through the network. To train the model. And this language modeling is the main technique for text pre training and well. I just told you about 1 task here, but of course there are many variants to this. Today's tasks, such as we want to predict context words that are around our word like the previous ones in the next ones or something that hurt. For example, users is a mask language model where you get a full sequence and umask out certain words and you want to predict those mask words that don't necessarily have to be the last one. 

*120.14 seconds*

### **12** Word2Vec 
So let's talk about the famous work back algorithm. More traffic. Basically, trains a one hidden layer neural network to predict context words, real language modeling. And then, Well, you might think, but that's kind of not the task of learning morning paintings. We never actually tell the word back network. That it leads to learn more embeddings, but how we kind of do that as a secondary output is. The target words are encoded via one hot encoding, meaning everything we have a an input vector. The size of the vocabulary. And his input vector has exactly 1 one value. And the position of the word in the vocabulary that we want to look at. And with that we can input and output words. And we can input words and output the probability of words. And. What we do is like the crucial step in the word vec architecture is that we harvest the word vectors from the network. So we here you can see in this picture we take them eight. The first matrix of the two of the. On the input matrix of the hidden layer, basically. End. We take it out. Right and now, because we used the one Hot Encoding, each row in this matrix corresponds to the one hot position of a word, so we can just index the word to that position and the rest of the network is ignored. Once we got the morning paintings out. Output matrix is. Round it gets discarded. We don't care about it anymore. 

*136.6 seconds*

### **13** Word2Vec 
There are actually two variants of work to make and work back is sort of only the umbrella term. First, we have the continuous bag of words which predicts a word given its surrounding context words, and then the other way around is skip gram, which predicts context words given. A single word and both methods have been shown to produce kind of the same or similar results, so it doesn't really matter which one we use, and it's just a difference in training, but the word embeddings we get out of it are very similar. And well, we have to note that even though if you Google for it, it kind of seems like work back invented word vectors. Which it did not. So of course there have been a lot of other techniques before it, but the difference is that work back. Um rose to a lot of Fame and got very influential be'cause its fast to train so every so it contains a couple of contributions that make the training much more efficient, which means that everyone can train on domain specific data from scratch, which is very nice and funny enough. This whole idea changes in the current years. It's going back to leaning to download image models. And of course we have to say it worked back. Just sounds a lot better than other algorithms like LDA or SVD, which also do some sort of word vector and one embedding. But work to make just sounds cool and. Porta Beck is also not the state of the art anymore, so it has been in the state of the art when it was proposed, but not anymore. But it's still good enough for practical use in many scenarios. And combined with the efficiency it is used a lot in practice and that's why we're talking about it. So if you want to know more details, you can also look up those resources in the bottom there. 

*146.63 seconds*

### **14** Word2Vec - Training 
And. Just to like give an overview of how it raining import back works is that we take a full text in one big batch of text, so we. Don't care about sentence boundaries. We don't care about. Document pounders would just conquered and ate all our full text together and then we move over that text with a sliding window of context words and words to predict. And. Yeah, you can see such an example in the in the finger on the right here where we have an input source that's just some bunch of words. And then we select the word sentence as our main word, take the predicting on the context words to predict around it, and we create window so called Windows samples where we create pairs between our main word and each thing of the single window words. So you can see. Then we kind of also discard that distance between our main and the window words. Then we look up our one hot encoding from our vocabulary position. And we feed each sample individually into the network so we always operate pair on pairs of words. We used one hot encoding as an input and with a softmax output of the output layer. We get the probability. Um of the neighboring words. In our language model and with that. We can compute a negative knock likelihood Los and the problem here is that this softmax. Needs to cover all terms in the vocabulary to get the probability overall terms and which is too costly because vocabulary can reach millions of words. So worked back, introduced negative sampling of random terms to get rid of the problem that we actually have to look at all terms in document and. I'm not going to go into more detail on that, but if you're interested in the mathematical aspect and how to derive all the formulas, please check out the Stanford lecture on exactly that topic that only focuses on the mathematical. Off work back. 

*168.55 seconds*

### **15** Offshoots based on the ...2Vec Idea 
And because as we said, the marketing of two back is great, there have been a lot of offshoots based on this whole. Take the context in sequence and do unsupervised prediction based on the sequence. And it has been applied to, of course, sentence, paragraph and document embeddings. But also, graph embeddings, entity embeddings and everything really with the sequence such as playlists or so. And of course there have been a lot of small and incremental adjustments and improvements to the original quarterback idea. And now. We're going to take a look at one of those. 

*49.53 seconds*

### **16** FastText 
And this is fast text, so fast text. Improves on by. Whole words, workpieces, or character engrams as a vocabulary? And character Ingrams here means that we take N characters at a time. So for example. We can take. 3 grams, which means we take 3 characters at a time in a window of the word. And with that. Hum. We get a single vector for each character N Gram and those vectors are then added together to form the single word vector. Which has the following benefits, so we have no more. Out of vocabulary words. Be cause we can always just substitute it for a character Engram. Then We have we don't have a problem of low frequency words or the problem of low frequency words is much less. Because we can. Uh, means low frequency words are made up of better known character in grams, so we get better representations for those. And it's very good for compound heavy languages like German, which. For basically the same word would need multiple. Someone embeddings an each time you add a compound, you get a new entry in your vocabulary, but not so with fast text and it kind of offers similar performance and Usability is worth of X, so we can still generate a single vector per term in our vocabulary. And we can apply all the analysis tools and methods that have been developed for work back. On fast text as well. And it comes with a very nice library and pre trained models for virtually all languages. 

*141.78 seconds*

### **17** Similarity Measurements 
So I briefly talked about analysis of work back. So what are those analysis based apon and mostly there are based upon some sort of similarity measure. So a word embedding is a vector space in which we can do math and mathematical operations, and a very common operation is to measure the distance between two points. And in a word, embedding this corresponds this. Measuring the distance corresponds to measuring the similarity of words. A very standard measure for that is the cosine similarity, which measures only the direction and not the magnitude of the two points an. It's implemented as the dot product of normalized vectors with length one. But the problem here is it's so you can always visualize cosine similarity in a 2 dimensional space, right? But it's very hard to visualize cosine similarity in a 300 dimensional space. Because here we can have. Crazy things that you can't really visualize which is. For example, we can have hub so called hub vectors that are close to many others. But the others? Close to each other. And the Jensen Library implements a lot of those similarity computations for you. 

*102.66 seconds*

### **18** Analogies 
And another. Common tool. That's especially suited for non technical audiences is to probe the relationships between more than two words with Analogies. So we want to say A to B is SC to D, which can be reformulated as a minus B Plus C equals D. And a very and it's kind of easy to understand, right? They're very common example. Is. King Minos man plus woman equals Queen, but it also works for heads of state, capitals of countries etc etc. Well. It might not be as easy as it seems, so the analogies are an oversimplification an they can required fragile between different runs. So when you have different initializations or small changes to the algorithm, those analogies can break apart and not show what you want him to show, and Furthermore 

*74.12 seconds*

### **19** Analogies - The Devil is in the Details 
The Devil is. Right, so I just showed you the King Minos man plus woman equals Queen example, but it turns out it might not be true. So in May 2019 is surprise. Finding takes the twittersphere by surprise. For some reason, a lot of people just reuse the same library code that makes those analogy computations. But this popular library code. Hardcodes That you cannot have. The same word as result in your analogy. So when actually King Minos, man plus woman goes back to King. It wouldn't output King, but the second best thing and. Yeah, that kind of is a problem, so here you can see that Ian Goodfellow, very famous but General Machine Learning Personality tweeted about that. Followed by an answer from a also famous personality. You have Goldberg who wrote the book. I recommended you, but he is a very LP person and he is kind of shocked. No one newness which is fun because of course, if you work with the library, you might know it. But if you don't look at the code, you could easily mistake it, which becomes a problem when you want to show bias. In the morning bedding and you kind of hard code the correct answer out of the analogy. So always be careful when you use code that's not yours and which is highly crucial to your study. You should really know what you're. Executing there and not. Just blindly take something.  

*133.56 seconds*

### **20** Limitations: Social Biases 
And on the subject of social biases, yes, even though in this study it has been shown that the analogies and the computation of the analogies has been a problem. Hum in like highlighting social biases. We still know that. Taking biased text. Bias representations Bias. Gender or racial. Easily can affect. Without anyone ever. Word Embeddings. Predictive policing Recommendation. The work of minorities, for example, because the words that describe them are in a more associated with a more negative way or missing in the training data etc etc. So there have been various methods proposed to proactively Debias Warren embeddings after they have been trained. But the problem is and as has been shown in the paper linked below. That those methods might not be truly effective and only cover up Tobias, so this is an active field of research where we still have to put in a lot of energy in time to figure that out. How to improve this problem. 

*98.68 seconds*

### **21** Limitations: Word Ordering or N-Grams 
Another limitation of those one vector for one word, word embeddings is that they don't actually pay attention to reward ordering or the word engrams in this case. So. Those two sentences that you see here, it was not good. It was actually quite bad, versus it was not bad. It was actually quite good. Would receive the exact same representation based on the word embedding, so the context in work back is only used during training, but not during the usage of the word embeddings. Right and the problem here is so worked back kind of solves that problem by creating sometimes by entry gram vocabulary entries for very popular Co occurring words. But to do that for every single word. Is not feasible. Because then you end up with two little training data and get no good connection between quite good and very good. If you see them as two distinct terms. But we'll hear more about that in the next lecture. 

*93.62 seconds*

### **22** Limitations: Multiple senses per word 
And then another problem with word back is. They don't. Which kind of follows the same pattern they don't know. Different senses of the same word, right? So you have this one word, one vector, which is good for analysis but it. If you're. If you don't know the context of the work, you can't know the current sense. It should kind of. It should represent. So if you have a word with multiple centers sensors and there are a lot of them. Squash those senses together. You either create an average overall senses, or if they are very imbalanced, the most common sense that occurs in the training data will win and dominate the others. And this is especially visible in domain specific data versus general English text data. Right, and this is definitely a missed opportunity for improved effectiveness, and that's why contextualized models have been proposed in the last year. And beyond. And we will look at them also in the neural IR lectures. 

*92.65 seconds*

### **23** 
Now let's take a look at how you could, or a first glimpse at how you could use more embeddings in the information retrieval context. 

*10.91 seconds*

### **24** Query Expansion with Word Embeddings 
And this is with. Query Expansion with. For query expansion and we use those word embeddings to expand the search space of a search query with similar words that we get from the warning bedding. And in this. Work from our group from 2016 minus units a couple of years old. We updated collection statistics. And adapted the relevance model to score a single document with multiple similar words together, so that the words kind of keep the connection between the original query and the similar words that have been expanded. And how this works is you can see this in this plot. Here you start with a sample query and. Then you split the query and for each word you look up the similarity from a warning bedding and then you define a threshold of cosine similarity such as zero point Seven that those words have to be above. And then you search for all the terms in the inverted index. You gather the posting lists and then you weight each document with the connection based on the weight from the embedding. 

*99.54 seconds*

### **25** Problem: Topic Shifting 
And what we recognized here is a problem, and that's topic shifting. So if you take a word embedding. That is trained on Wikipedia context. Than words with similar context are close together. But they can be of a different topic, which is of course bad for the Retrieval Task. So the word Austria has close neighbors that correspond to physical neighbors, right? So Germany, Hungary, etc. And if you search for hotels in Austria, you probably only want results located in Austria and maybe expand the query with towns and cities in Austria. But not in other countries, because then you would search for the different countries. 

*55.75 seconds*

### **26** Retrofitting 
And to kind of solve that or mitigate that problem, we utilized a technique called retrofitting, which is very interesting where you actually incorporate external resources into an existing word embedding. By moving the vectors of the existing born in fighting in an iterative update function. What each word gets moved by a combination of its original position and similarity values to other words based on external information. That means that, for example, in this example here you have two terms that are in this radius of neighborhood, but you don't want them there, and the external resource kind of tells you that they shouldn't be in your expansion, so. If you apply a retrofitting and then you can move the word vectors a little bit away from them so that they are not in the radius anymore. 

*69.4 seconds*

### **27** Topic Shifting - Examples from Wikipedia 
And. To kind of showcase that, we utilized a word back, skip gram embedding, combine with latent with an embedding from latent semantic indexing or short LSI, and you can see that in the original Skip Gram embedding for Austria you have. Words that don't belong there. But if your retrofit it with the word Austria from LSI. You get a clean output and the other way around is also choose if you have Austrian. In this example from Wikipedia you now have. Problems in your LSI inventing. But when you retrofit the two embeddings together, you get a better result overall. End. Yeah. And you get a better result than if you would use each of them. 

*64.88 seconds*

### **28** Application to Patent Retrieval 
And that's what we did in the. Area of pattern retrieval sewing Pattern Retrieval. The recall is. Single relevant patent. Session can be very bad, well, or at least people tell me that. Don't have any patents. No, if you have a vocabulary mismatch so you don't directly find your documentary of pattern. Even if you click through all results as there are many domain specific terms in patterns. You should expand the queries right and we expanded the queries with retro fitted word embeddings and we kind of show that if you use the word embedding alone to expand the query, you do not improve the results, but if you take a combination of the local context, skip gram work back embedding, combining with this global LSI embedding via retrofitting, you can actually improve the recall significantly. In a standard. Hum.  

*84.59 seconds*

### **29** Outlook	 
Yeah, but we have to say, Well, this work that we did was from 2018 and of course it had a lot of limitations so it was an unsupervised method and in the end it kind of had a limited effectiveness and only increase the recall in the patent scenario. But since 2018 alot of stuff happened in the field of neural information which we will and we have kind of a recipe book now with ingredients for better approaches. So we are using supervised learning with large retrieval datasets on top of unsupervised pretraining. Then we have models with way more parameters to learn. Then we did in 2018 were using contextualization, and we operate on the full text of query and documents. And well, I think this is all very exciting any. I hope you tune into the next lectures where we're going to look at all those ingredients for Neural re ranking approaches and more.  

*75.88 seconds*

### **30** Summary: Word Embeddings 
OK, So what I want you to take away from this talk is that. We need to represent words as vectors instead of their character values with morning meetings. And unsupervised pre training is a major strength of word embeddings and still also in more advanced models. Unsupervised pre training is the key to success and weren't embeddings have many potential applications in natural language processing but also in information and we will such as query expansion which I just showed you. 

*43.65 seconds*

### **31** Thank You  
Well, thank you very much for your attention and I hope you tune in next time. See you then. 

*9.22 seconds*

### Stats
Average talking time: 84.56333669354841
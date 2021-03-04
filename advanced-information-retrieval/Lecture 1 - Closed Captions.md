# Lecture 1 - Crash Course - Fundamentals

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Hello and welcome to our crash course on information retrieval. Today we're going to talk about the fundamentals. My name is Sebastian. If you have any questions, please write me an email. 

*18.78 seconds*

### **2** Today 
Our fundamentals talk today will look at 2 main aspects of traditional information retrieval, and that is first the inverted index and its data structure, an creation process, and then second how we use this inverted index for search and relevance scoring models that build upon the statistics saved in the inverted index. 

*38.73 seconds*

### **3** Information Retrieval 
Information retrieval, in its most basic core concept, asks the question of a given query (here in this example  "Elephant weight"). How relevant is this query to a given document? Can we quantify this relevance? 

*21.27 seconds*

### **4** Information Retrieval (Finding the needle in the haystack) 
Now the problem is we actually don't have a single document, but potentially millions and billions of documents. So how can we efficiently find this needle in the haystack, the one or the couple of documents that are relevant to the given query in the sea of non relevant ones? 

*28.49 seconds*

### **5** Notes on terminology 
In this course and in this lecture specifically, we use the term 'documents' interchangeably for all kinds of different file formats, so this can be a web page, a word file, any text file and email, etc, and we assume it to be a simple text file without any markup, to make things easier. And there are of course a lot of details to look out for in the real world, such as encoding, languages, hierarchies in different document fields, etc. A collection for us is a set of documents, and here in this lecture we assume it to be static to make again things easier. But of course, in the real world, you would need to be able to update those set of documents in a lot of scenarios. And then relevance is defined as: does a document satisfy the information need of a user, and does it help complete the users task? 

*76.83 seconds*

### **6** Relevance (based on text content) 
And to continue this relevance idea based on text context, we expand our example. So here elephant and weight are two terms. If a word appears more often in a document that we search for, we can assume that this document is more relevant. So how do we go about this solution? We just count every word in every document. If a document is longer, words tend to appear more often, so we also have to take into account the document length. And the problem now is if we count every word in every document only when we have a query, this is very inefficient, so we need to come up with a data structure that allows us to precompute most of our needed computation.  

*66.13 seconds*

### **7** 
This is done in an inverted index. Here we transform text based information. 

*7.71 seconds*

### **8** Inverted Index 
The inverted index allows us to retrieve documents very efficiently from a very large collection. And the inverted index is a very basic data structure that stores statistics per term, and those statistics are then used later on by a scoring model to produce relevance score. And the main statistics we work with today in this lecture are document frequency, so how many documents contain a term, and term frequency per document, so how often does a term appear per document. And Furthermore, we can also save the document length and the average document length. And we need to save those statistics in a format that is accessible by a given term because at query time we only know the query terms. And Additionally, of course we can also save metadata about document such as the name, the title, and some paths to be able to display the document to the user later on in the user interface. 

*85.1 seconds*

### **9** Inverted Index 
A very simple overview over this data structure is shown here. So on the left side we have both document data and term data. And it's important for efficiency reasons, but also to keep it simple, that every document gets an internal document ID in smaller indexes that are not webscale. This can be for example a simple integer, that just counts the index of an array. And the term dictionary must be saved as a search friendly data structure. So we need to be able to locate the so-called posting list that contains a list of document IDs and frequencies per term, and we need to be able to locate this very efficiently. Because of course we don't want to go ahead and compare our query term with every single available term in our vocabulary every time a user types a query. That would be again very inefficient. 

*76.46 seconds*

### **10** Creating the Inverted Index 
And to create this inverted index, we process documents in a simple pipeline. So here we process a document 1 by 1. And it starts off by registering metadata and assigning a document ID. Then we tokenize the document and, for example, lowercase every single word. Then we conduct something called stemming where we reduce every word to its word stem. In this example, only the S from infos is removed, so we find both info and infos for that term. Then we filter the so-called stop words that are very frequent words that often but not always don't carry relevant information. And now we have our set of words, or bag of words if you will that we can add to the dictionary, and for every term we update our posting list with our statistics per document. And the query text and the document text both do those same steps, so words always mapped to the same common stem. 

*101.01 seconds*

### **11** Tokenization 
Let's look at each of these steps in a bit more detail, starting with tokenization. A very naïve baseline just splits on each whitespace and punctuation character. So for example, if you have something like U.S.A, you would split that into three different terms, U, S and A. Or if you have a date, you will also split that date into something that would be hard to recognize if you search for that exact date. However, even though we find those examples that are not the best showcase, is still a very strong baseline for the English language, and an improvement on top of that would be for example to keep abbreviations names and numbers together as one token. For example, the open source tool from Stanford that creates an tokenization which can also handle emoji characters, which are surprisingly hard to be aware of. 

*76.21 seconds*

### **12** Stemming 
The next point is stemming. Here we reduce the terms to their roots and we just chop off the end of a term. This of course is language dependent and does not work equally well across languages, but again, for English it's a pretty good baseline. In more advanced form is so called lemmatization, where we keep a dictionary to reduce variant forms to their base form. So for example am,are,is  all reduced to be. Computationally, this of course is more expensive than have a simple rule based chopping off the end. 

*58.21 seconds*

### **13** 
OK, so let's assume we build our inverted index. We implemented it, we filled it with hundreds of thousands of documents, and now we have all those terms statistics per document in our efficient data structure. So how do we actually use this data structure to conduct a search?  

*26.19 seconds*

### **14** Querying the Inverted Index 
The query workflow is as follows. We start off with a query and we want to answer if it's relevant so we look up every term in our inverted index and gather our statistics that are saved in the inverted index. Those statistics are fed into a scoring model that only operates on the statistics without needing to read er open the full document. And we only have to operate the scoring model on the subset of documents that are referenced by our given query terms, which of course reduces the amount of computation required in most cases. And now the scoring model gives us a score for each document and query pair so that we can sort the documents by score and retrieve the most relevant documents. And then we can open only the top documents and display them to the user. Of course, this is a simplified view, because a document could be relevant without actually containing the exact query terms, and we will spend a lot of time in the neural IR lectures concerned with that problem. But now let's keep it simple and say document is only relevant if it contains at least one occurrence of an exact term. 

*111.53 seconds*

### **15** Types of queries (including, but not limited to) 
And those types of queries that we can do on top of an inverted index are already pretty diverse, so we can search for exact matching where we match full words or concatenate multiple query words with an or operator. We can also conduct Boolean queries where we say both words or all the words have to appear or certain words are not allowed to appear in a document to filter our results further. And then of course we can expand queries and incorporate synonyms or other similar or relevant words into the query. Furthermore, we can conduct wildcard queries, phrase queries, phonetic queries, a lot of different query processing techniques can be applied before we look up statistics from the inverted index. 

*66.25 seconds*

### **16** Inverted Index: Dictionary 
A key component of the inverted index is the actual implementation of the dictionary that holds our vocabulary terms, and this dictionary maps text to an object and this object, in our case, here is a posting list, but of course we can also augment it with other data if that's needed for specific scoring models. And the properties that we want from this dictionary is first and foremost a very fast random look up. We don't want to sequentially iterate through the dictionary to find something we want to be able to randomly look up something very fast, and it should also be very memory efficient so we can keep the complete dictionary in our memory. And naturally, there are a lot of choices created by computer scientists over the years. 

*72.4 seconds*

### **17** Dictionary data structures 
The most easiest and probably common solution for simple inverted indices is a hash table, that simply Maps the hash value of a word to a position in a table. A more advanced method would be a tree, Oor a prefix tree that stores the alphabet per node and a path through the tree forms words. This is very helpful if you want to have something like autocomplete or part of word matching. Then if you have a B-tree, you can have a self balancing tree that improves the efficiency of a normal tree. And often also used is a finite state transducer which is a very memory friendly automat that allows you to create your dictionary once as this automated walks through different characters. Related here is the so called Bloom filter that's not used directly in the inverted index, but it's a very cool data structure that tests if an element is in a set, and this can be done very, very efficiently. So for example if you need to spread out your inverted index across multiple nodes you can use a Bloom filter to check if your term is local to a certain node, before conducting a very expensive network operation. 

*127.64 seconds*

### **18** Hash table 
The hash table uses a hash function to quickly map a key to a value. It's a very common and broadly applicable data structure, and it powers most of the basic dictionary implementations of most programming languages. It allows for a fast look up. Which is in terms of all O(1), but I want to caution that this doesn't mean it's free, so you still have to compute something and it's not instant. You don't need any sorting or sorted sequential access, which is very good for our inverted index. But it only does direct mapping, so there are no wild card queries and you can't do autocomplete easily with that. 

*60.23 seconds*

### **19** Spell-checking 
To also improve the term matching capabilities you can use spell checking, especially if you work with text that's not of a very high quality. And what you want to do here is first correct documents that are indexed but also correct user queries to retrieve the correct answers in a "did you mean XYZ?" style. There are 2 main flavors of spellchecking of simple spellchecking where one new conduct spell checking on isolated word so each word is checked on its own for misspelling's, and while this is very simple and efficient to do, it will not catch typos that  come from a context of words. And a context sensitive spellchecking has to look at surrounding words to know if a certain word is the correct applicable here in this scenario. 

*78.41 seconds*

### **20** Spell-checking by Peter Norvig 
Of course, with spell-checking, as with every other task, there are numerous neural network methods that try to solve this problem, but a very simple solution by Peter Norvig is self contained in a few lines of code that does simple and isolated spell checking. And if you're interested in spell-checking, I can highly recommend that you check out the details, implementation, and explanations of why this simple solution also produces good spell-checking results. Already, it uses a text file of a million words collected from books for correct spelling information, and then in works based on the probability of each word occurring based on the frequency in the books. So you get a set of candidate words from your text that you want to spell check and then the most probable correct spelling is induced from the available candidates. Please go check it out if you're interested for more. 

*86.47 seconds*

### **21** 
So now we switch gears and we talk about how we can compute a relevance score and basically select the top ten documents out of millions of available candidates.  

*17.93 seconds*

### **22** Scoring model 
A scoring model is defined as having an input of statistics from our inverted index, and the output is a single floating point value, the score. And we do evaluate this model in a pairwise fashion, so one query and one document are scored at a time. So if we want to score 1000 documents, we have to call this scoring model a thousand times, and it captures or tries to capture the notion of relevance in a mathematical model. And today we focus on free text queries and so called ad hoc document retrieval where we only look at the document content and not possibly other values such as page rank, recency or click counts, etc. 

*62.44 seconds*

### **23** Search algorithm 
The most basic search algorithm looks at query terms one by one, so here it goes. We have a result variable that keeps track of our partial scores. Then for each query term, we fetch the posting list for this query term via the inverted index. Then for each pair in our posting list so the document ID and the term frequency and potentially other statistics, are calculated partially by the scoring model and then we aggregate the score per document ID. So after this nested for-loop, we get the final scores per document ID.  

*66.71 seconds*

### **24** Relevance 
Let's recall our relevance intuition from before. So if a word appears more often, it is more relevant and the solution we are coming up with is to count the words, right? So with our search algorithm we are good on that part. Basically we are able to efficiently count the words in the documents once during the inverted index creation phase, and now during query we only look at the statistics. OK good. So if a document is now longer, words will tend to appear more often, so we also need to take that into account. 

*47.25 seconds*

### **25** Relevance 
And before we continue, I also want to emphasize again that here in this scenario words are meaningless. We see them as discrete symbols, and documents are kind of a stream of meaningless symbols, and we only try to find patterns or trends. This is different from most of the topics later in this course where we try to understand the relevance by having a meaningful representation of each word. 

*43.17 seconds*

### **26** Relevance limitations 
And the limitations of this simple pattern based approach is that relevance means the relevance of the need rather than to the query, and it's hard to quantify the need in such large scale scenarios, and relevance also again here is assumed to be a binary attribute, so either a document is relevant to a query or it's not. Of course in reality we have more of a graded and finer grained view on this issue, which we will look at in future lectures. But right now we need to make these oversimplifications to create and evaluate those simple mathematical models.  

*63.16 seconds*

### **27** 
Let's go into details and start with the most basic count based scoring model there is, and that's TF-IDF or in the long form term frequency inverse document frequency. 

*16.21 seconds*

### **28** Term Frequency – conceptional data view 
The conceptional data view of term frequency in a bag of words approach that we here use where the word order is not important and you just count the occurrences of words. You basically get a sparse matrix, and this sparse matrix is efficiently represented in inverted index, but from purely conceptional view, if you have terms and documents, you create this matrix indexed by term and document to get a term frequency number, per term and document.  

*51.04 seconds*

### **29** Term Frequency – actual data storage 
And again, the actual data storage is much more efficient because we have so many documents in an index that saving such a matrix with basically almost everywhere 0 is not a good idea, and the inverted index saves only non zero entries. Therefore it's not good at those random index based look ups that you could do in such a in an array based matrix data structure. However, we don't actually need to do a random look up on the document dimension because we have to score every single document per posting list.  

*52.45 seconds*

### **30** TF - Term Frequency 
So the term frequency is a measure of how often the term T appears in the document deep. It's a very powerful starting point for various numbers of retrieval models. And it's also is the main point of our intuition from the beginning. However, empirical evaluation over time showed that using the raw frequency is not actually a very good solution, so you can either use relative frequencies, or you can dampen the values with the logarithm.  

*42.86 seconds*

### **31** Term Frequency & Logarithm 
And this logarithm based dampening looks like that. So on the X axis we have our raw term frequency and on the Y axis we have the output of either the raw count or the log dampened value, and you can see that in long documents where terms appear more often, potentially hundreds of times, the log based term frequency dampens this effect quite a bit, and retrieval experiments show that using this logarithm is more effective than just the raw counts. And a commonly used approach that we use here is to add 1 to the term frequency and then apply the logarithm. 

*57.93 seconds*

### **32** Document Frequency 
The document frequency on the other hand, is a measure of in how many documents does the term T appear totally in? And the intuition for that is that rare terms are more informative than frequent terms. So consider if we have a query that contains the word TUWIEN, and we want to only find documents containing this term. It's very likely that TUWIEN, a document with TUWIEN in it is more likely to be relevant to a query that also contains TUWIEN, then maybe other words in addition to that. So we want a high weight for rare terms like TUWIEN. 

*60.09 seconds*

### **33** IDF – Inverse Document Frequency 
And the way we get this weighting for rare terms is the inverse document frequency. And a common way of defining it is as follows. We again use a log-dampen effect, and we divide the total number of documents by the amount of documents that contain the term at least once. And it's an inverse measure of the informativeness of a term, and the document frequency is of course smaller or at maximum equal to the number of documents. 

*50.74 seconds*

### **34** TF-IDF 
So TF-IDF just put together those two measures we just talked about. So we have the term frequency and we have the inverse document frequency, both dampened with a logarithm and then multiplied with each other. Those values are then summed up over all our terms in the query and in the document that we want to score. This part term frequency increases with the number of occurrences within the document. And this part increases with the rarity of the term in the collection. And so you can see if a rare word in the collection appears a lot in one document, we create a very high score and common words are downgraded on the other hand. Of course, this is only one variation of this TF-IDF formula, and there are many other ways to slightly change the combination of term frequency and inverse document frequency, and you can look up the very detailed Wikipedia page for more info on that. 

*86.49 seconds*

### **35** TF-IDF – Usage 
TF-IDF is not only useful as a standalone ranking model. But its weights can also be used as a base for many other retrieval models, such as the vector space model that works better with TF-IDF weights, and it's also useful in a generic word weighting context for many NLP tasks, so we can have a task agnostic importance of a word in a document in a collection. And we can of course assign every word in a collection Its TF-IDF score. An example for that is again LSA latent semantic analysis that has been shown to work better if you utilized TF-IDF weights with it. 

*56.12 seconds*

### **36** 
Now, our main improvement today over the TF-IDF baseline is the so-called BM25 or best match 25 scoring model, that has been the backbone of many if not most open source search engines for the last 30 years. 

*29.98 seconds*

### **37** BM25 
It has been created in 1994, and it's grounded in probabilistic retrieval. And overall you can say that BM25 improves a lot over TF-IDF across most collections it's been tried on. But one caveat is that it has only been set as a default scoring in the popular Lucene search engine in 2015. So before then used TF-IDF. 

*42.91 seconds*

### **38** BM25 (as defined by Robertson et al. 2009)  
So let's dive in straight into the BM25 formula as defined by Robertson and al. in 2009. So again, we see that we have two main components, one on the left is concerned with the term frequency and the one on the right is concerned with the document frequency. And the changes in comparison to TF-IDF are mainly here in the first part concerned with the term frequency that now also takes into account a normalization of the document length by the average document length and it introduces 2 hyperparameters, k and b, that can be tuned to define the influence of this document length normalization. And on the inverse document frequency side we see that now the formulation has changed a bit where we subtract the document frequency are of term team from the total before dividing it. Of course, we have to emphasize here that it's again only one of many possible variations, and we assume that we have no additional relevance information. If we do, a lot of different modules have been proposed to be able to put into this basic formula right here, and it's also simpler than the original formula from 1994 because over time it was shown that more complex parts concerned with query length etc are not needed in empirical studies. If you want to know more about BM25, and also of the origins and probabilistic theories behind it, there is a great overview paper with a lot of details linked here. 

*147.56 seconds*

### **39** BM25 vs. TF-IDF 
The simple case of BM25 looks a lot like TF-IDF, as we just saw. The one main difference is that BM25, the term frequency component contains a saturation function an this has been shown to be the source of more effectiveness in practice. And of course, BM25 can be adapted for many different scenarios such as long queries or multiple fields.  

*36.08 seconds*

### **40** BM25 vs. TF-IDF - Saturation 
If we take a closer look at the differences between BM25 and the TF-IDF term frequency saturation, we can see that even though TF-IDF uses a logarithm to dampen the raw values, the saturation from BM25 is even more dampened and diminishes even more quickly based on the K hyper parameters that are used. 

*34.16 seconds*

### **41** BM25 vs. TF-IDF - Example 
No, let's look at a simple example that showcases the differences between BM25 and TF-IDF in a more practical way. So suppose your query is "machine learning" an you have two documents with the following term counts, right? So in document one it talks about learning a lot, but machine only appears once, and in document two you have learning an machine in roughly a similar range, but together they appear way less than learning in document one. And TF-IDF actually produces a higher score for document one, then documents two whereas being 25 with the default key hyperparameter, produces a higher score for document two than document one. And here, of course, this is only a very simplified example, and we don't know the document contents, but neither do the scoring models themselves, but we can assume here that document two is more relevant to the query machine learning because it contains more machine and learning together.  

*91.95 seconds*

### **42** Hyperparameters 
The hyperparameters of BM25 are set by us, right? The developers, the practitioners, and K controls the term frequency scaling. If we set K to zero, it suddenly becomes a binary model and if we set K to a very large number, we basically look at any raw term frequency. And then the hyperparameters B controls the document length normalization, so zero again is no length normalization and set to one is only relative frequencies fully scaled by the document length. And of course there are common and default ranges and if you tune those hyperparameters you can improve your results a bit for your specific test collection. But we also have to note that in general you can just take the default hyperparameters and BM25 will work fine.  

*73.65 seconds*

### **43** BM25F   
To extend BM25 to cover more of a document structure, the BM25F scoring model was proposed, where we look at in more real world use case, where documents have a title and abstract or headers etc. And  BM25F allows for multiple fields or streams in a document. For example, you can think of a news article or so or publication as having three streams, a title, and abstract and a body, and now BM25F allows you to assign different weights to the individual streams so you can increase the weight of the title and the abstract, for example. 

*60.44 seconds*

### **44** BM25F (as defined by Robertson et al. 2009)  
The formula now looks like that, so we introduce this new stream length and stream weight parameter, but other than introducing these additional fields, the formula components themselves don't change, we only add another weighted sum to the composition.  

*33.39 seconds*

### **45** BM25F   
In BM25F we first combines the streams and then the terms, so this is different than just chaining together BM25 results on different streams. And the saturation function is applied at the stream level, so this follows the property that title and body are not independent from each other and they should not be treated as independent. And of course their stream lengths and average stream lengths can vary quite a bit, and in most cases you would assign a higher stream weight to a title or abstract for example. But again, those are weights that you can tune and do a hyperparameter search for your specific collection that you're using it in. 

*62.37 seconds*

### **46** 1998: Google 
Finally, I want to give you a quick hint to a very interesting paper, and that is a paper from 1998 that was the firts paper to introduce Google as a search engine. And Google actually started as a research project at Stanford and obviously it contain a lot of good ideas such as page rank. And it saw information retrieval as a problem of context and scale, and I want to emphasize that here at the end of this lecture, to make you aware that this paper contains a lot of interesting implementation details of this first initial version of Google, that is very easy to read and hopefully good to follow once you heard this lecture. 

*62.62 seconds*

### **47** Summary: Crash Course – Fundamentals 
With that I want to finally emphasize the key takeaway messages from our crash course on fundamentals, and that is first we save statistics about terms in an inverted index so that we can then, at query time, access those statistics in a very efficient way, by a given term in the query. We looked at two different scoring models, TF-IDF and BM25 that both use term and document frequencies to score a query and document pair. 

*48.79 seconds*

### **48** Thank You  
Well, this is it for today. Thank you very much for your attention and I hope to see you next time. 

*10.26 seconds*

### Stats
Average talking time: 58.18389453124998

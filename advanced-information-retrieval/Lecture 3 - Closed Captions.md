# Lecture 3 - Crash Course - Test Collections

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Welcome to this crash course on information retrieval. Today we're going to talk about test collections. My name is Sebastian, and as always, if you have any questions, remarks or suggestions, please feel free to write me an email. 

*17.83 seconds*

### **2** Today 
The topics of today are as follows. We're going to first look at some existing test collections and look at what is a test collection? Then we will see how to create your own test collection by running your own annotation campaign and some techniques that are information retrieval specific to be able to actually create such judgment annotations that we need for our information retrieval evaluation. And then, last but not least, we're going to look at analyzing biases. This is, on one hand social biases that are concerned with learning from text and then, on the other hand in more technical position bias that is induced in passages and documents. 

*61.02 seconds*

### **3** The Need for Datasets 
The need for datasets, in my opinion is very clear: Machine learning is driven by data and every task needs some sort of task specific dataset that models what we want to achieve. And for empirical evaluation, if we want to evaluate a model's effectiveness, for example, is our system better than the baseline according to some metric? We need data to be able to make this judgment. And most of the time, but of course, not always, we also need a data for training or fine-tuning a pre-trained model. And in many cases, the more data the merrier. But again, with the caveat that we need also to focus on the quality of the data, so the more quality the merrier as well. And there are many different tasks in machine learning and the different subfields of computer science, which means there are many datasets for various tasks. However, the more specific you drill down to a certain task, the scarcer the datasets usually become. A great hub for text-based datasets is created by Hugging Face and gives you a nice overview of all the different tasks that people are currently trying to solve and work on. 

*102.75 seconds*

### **4** Offline IR Evaluation Setup 
In information retrieval the setup - if you recall from our evaluation lecture - is the following: We have a ranked retrieval result from some retrieval system that we want to evaluate, and on the other hand we have known judgements for a given query. We have a pool or a set of documents or passages that contain labels that tell us if this document is relevant to this given query. And this pool of judgements does not necessarily cover the whole list, and if we have missing judgements, we have to consider them as non-relevant. So now, in the evaluation lecture, we looked at how to use judgements to produce metrics scores and, in this lecture, we're going to look at what judgements are available and how to create those judgements. 

*67.31 seconds*

### **5** Existing IR Test Collections
Existing IR test collections are the foundation for all information retrieval research because without test collections we would be totally in the dark of how good our systems are actually working. 

*18.9 seconds*

### **6** A Test Collection 
A test collection is usually an umbrella term for the following set of data. It's used for offline evaluation with a fixed set of documents or passengers, a fixed set of queries and a fixed set of judgment pairs that cover some, but not necessarily all query document combinations, and those judgements can usually either be binary (one or zero) or on a graded scale: for example with four different judgements classes and the higher the number, the more relevant the document is. The query source can vary wildly, so we can have handcrafted queries from experts for a set of documents that try to probe the collection as diverse as best as possible. Or we can use sampled queries, for example, from actual users as is done in the MS MARCO collection. And the document source is very task specific in most cases. Although, in general, I would say as a sort of theme for our research, we focus a lot on well parsed web text, news articles, Wikipedia articles, etc.  

*95.76 seconds*

### **7** Existing IR Datasets 
Here is a short overview of existing IR datasets and I want to start this overview by mentioning two annual annotation campaigns that actually produce new datasets and new judgments for existing collections each year. And those are TREC, the "Text REtrieval Conference", organized by NIST, which we will hear about a bit more in a couple of minutes, and NTCIR, which is organized in Japan. Some example datasets include CORD-19, which is a very recent one which has been developed during the COVID pandemic to help in scientific publication search with topics concerned with COVID. And then there is ClueWeb, which is a large web-crawl data collection with some web searches associated with them. A very common collection in the last few decades, I have to say, is TREC Robust 04, which is based on topical news article searches and it has been influential for a very long time and used by army searches over and over again. Also, Newsgroups and Reuters news collections are mentionable. There is a brand-new hub to get IR data which is "ir-dataset.com" and you can check it out if you want to know about more existing IR collections. 

*114.2 seconds*

### **8** MS MARCO Microsoft MAchine Reading COmprehension Dataset 
Now we'll turn to some of the most influential test collections from the last few years and here I want to start with maybe THE most influential one for everything concerned with neural ranking and that is MS MARCO or the Microsoft MAchine Reading COmprehension Dataset. It's the first re-ranking dataset with a lot of training data and I want to emphasize this "a lot of training data", because now the scale of training and evaluation data becomes an issue, which is very much a luxury problem that IR researchers in most cases, especially in academia, didn't have before. The MS MARCO dataset is created from real-world web search queries and passage-level answers from the Bing search engine. Its initial focus was on Question/Answer, the answer generation. And it's been released by Microsoft Research. It contains sparse judgements with labels for over 500,000 queries for training. The queries are selected with a very high occurrence count in Bing so that it is impossible to leak personal data, and those sparse judgments are humanly annotated. However, we only have, on average, bit over one relevant judged document per query.  

*105.79 seconds*

### **9** Importance of MS MARCO 
In my opinion, the MS MARCO collection was a major contributor in the paradigm shift towards neural networks because it allowed academic researchers to participate in large-scale information retrieval that was only possible for web-search company internal researchers before. And the leaderboard that was accompanying MS MARCO - and still is - is very popular with over 100 entries already. Most neural IR papers nowadays make use of the MS MARCO data in some form. Also, I want to mention including work from Google which is very nice to see. And if you look closely, here on this leaderboard screenshot, and you squeeze, you can even see one of our entries, here at the bottom on number 11. 

*64.41 seconds*

### **10** MS MARCO – Data Example 
Let's look at a data example of MS MARCO. So, training triples are as follows. We have a query and then, our annotated relevant document and, usually, we sample some non-relevant document that has something to do with the query, but it's not actually relevant. So here "what fruit is native to Australia" and the relevant document talks about a rare passion fruit that's native to Australia. But the non-relevant document talks about a fruit that's native to Africa, which would in fact not be a relevant answer. And in the re-ranking scenario, the evaluation tuples that we're looking at is as follows. So here we look a query document pairs each come with an ID, so we can map it to the judgments and model now has to give a score of those two entities. 

*61.49 seconds*

### **11** MS MARCO Passages & Documents 
Up until now, I talked about the MS MARCO dataset as one thing, but in reality, it's two. So, we have an MSMARCO-Passage dataset, which is the one I was talking about until now and we also have an additional MSMARCO-Document collection where researchers went back to the queries and judgements and URLs from the MSMARCO-Passage and actually re-scraped the web pages to include the full web page titles and URLs from the corresponding passages. There is a slight time-mismatch and there is a non-perfect matching, but it's not a huge issue and the documents here are very long, on average more than 2000 words. Our plaintext size actually increases from 3 gigabytes of plaintext from the MSMARCO-Passage collection to over 20 gigabytes for the Document which gives us another challenge to work on. 

*67.34 seconds*

### **12** Limitations of MS MARCO 
And I also want to mention that MS MARCO does have limitations and we should be aware of them. So, the way they MSMARCO passages were sampled from the Bing index is as follows: we start with queries and then take the top ten passages for each of those queries from Bing to create the passage corpus which might make the task easier in comparison to a web-scale index that has not those cookie cutter selections around the queries. The focus is on natural language question style queries because the initial project was focused on this QA task, so it's not perfectly representative of a web-search task. And it's only for the English language, so we don't have other language data in MS MARCO. Sparse judgements can be a problem if we say: OK, we have only one judgement per query, which means that we probably miss a lot of additional relevant results if they are ranked higher. This can somewhat be mitigated with the large query count for evaluation to make the evaluation less noisy. 

*86.91 seconds*

### **13** TREC – Text Retrieval Conference 
Let's switch gears and look at a completely different way of generating IR datasets, and this is with the TREC Text REtrieval Conference, an annual event organized by NIST, which includes various tasks (or tracks) with a challenge for participating systems. Those tasks are very much IR-oriented, but with many interesting twists on domains and settings in news podcast, medical or web search. Many teams then try to solve this task and submit solutions or so-called runs in the summer months and TREC has a budget and the expertise for annotations. Organizers create judgements for all participating systems, as many as they can, and then at the conference in November, everyone presents their approaches and gets the metrics from their runs, so the teams can connect. The TREC conference established many IR standards and test collections over the last decades. And it's just an awesome culture of information sharing and sort of a fun competition. 

*89.79 seconds*

### **14** TREC Deep Learning Track 
One of those track tasks is the TREC Deep Learning Track which started in 2019 and continued since. It's based on a MSMARCO-Passage and Document collections, and it is setup to evaluate a diverse set of systems trained on this large data. The way it's been done is as follows: we selected a small subset of MSMARCO queries, 43 and 54, and then participants create runs for those queries, TREC pulls the runs and creates as many graded judgements as possible per query. It is a way to complement the sparse judgements from MSMARCO. The main results so far are that neural methods are as good as we thought before this track started, and sparse judgements are actually a good approximator for certain evaluations compared to the high-quality dense judgements. 

*71.38 seconds*

### **15** TREC-DL 2020: Neural vs. Traditional Methods 
If we look at the comparison between neural and traditional retrieval models, in the TREC-2020 edition, the Passage task shows a very large spread between the best pre-trained language models, and then the best neural model from scratch as well as another large spread towards the traditional IR models. It's a strong validation of the neural hype and we also have to note a few things about this plot on the left here: so first the y-axis is a huge difference, so normally you would see very small gaps, but in this case, we have very large gaps and the top pre-trained language model runs are very slow ensembles so they represent the highest achievable, but not realistically deployable models. And in the document task the results are sort of in the same direction, but with less of a spread and the neural models from scratch do better than in the Passage task. The passage and document numbers are not directly comparable, so we can only compare inside a task. In my opinion, the way in the document task (the spread), is not as big because the community invests more time and focuses more on the Passage task at the moment. 

*115.87 seconds*

### **16** TREC-DL 2020: Sparse vs. Dense Judgements 
If we compare sparse and dense judgements, we see that here on the left we have a plot comparing on the y-axis the dense TREC labels and on the x-axis the sparse MSMARCO labels and all the participating models are plotted for the 50 or so queries that we're using here. And we can see a clear correlation between sparse and dense judgements, although the small differences, especially at the top, shouldn't be used for definitive system ordering, for so few queries. But it also shows that we can continue to use the sparse labels on their own.

*56.7 seconds*

### **17** Creating IR Test Collections
Now that we've seen what's out there, let's look at how to create our own test collection for specific scenarios and how to know what we don't know. 

*12.1 seconds*

### **18** Who creates the dataset?
One of the first choices you have to make for creating your own data set is: who creates the dataset? And every decision has tradeoffs. Here is an initial assessment of pros and cons between doing it yourself or with your team, versus hiring external annotators to do the task. For external annotators you need to do recruitment, you need to have compensation and you need to train or prepare workers for the task. All of those things you don't have to do if your team does it on its own. The problem by doing it on its own is that it doesn't scale very well if you want a large or high scale of annotations and here, the external annotators can be able to scale very well, and the expertise might also differ. So, for example, in the medical domain you might be in need of hiring doctors, and other expert annotators to actually look at the data that you, with your computer science background, might not understand.  

*82.18 seconds*

### **19** Task Design 
Then, before you start annotating, you have to create your task design, so you have to have a clear picture of what you want to accomplish and think about the caveats of the task you want to solve and how to handle them. Take a step back and look at: does it make sense what you planned? Definitely try it out yourself before continuing. And one crucial aspect of good annotation workflow is to make the task as simple as possible or decompose complex tasks into smaller bits. This gives you less ambiguity, better agreement between annotators, and of course it's easier for annotators to understand a short description versus having to read a 20 page or so manual before starting, which is definitely not motivational. A simple task is also easier to analyze afterwards and the annotations, depending on the task, might even be used for more than one goal if we recombine them in different ways. 

*78.95 seconds*

### **20** Task Design -Example FAQ Annotation
Here is an example of such a clever task design by my colleague Marcus Zlabinger, in which he created FAQ annotations where the answer catalog can have hundreds of entries, but is a very fixed set, and looking up this answer can be very time consuming, so the idea here, in this efficient answer annotation, was to annotate questions with the same-intent as a group and by grouping questions with an unsupervised semantic text similarity method, the group-wise average annotation time could be reduced to 25 seconds per entity versus 42 seconds if we do it 1 by 1. So, the result is a more efficient annotation.  

*58.33 seconds*

### **21** Task Guidelines and Examples
After you settled on a task design, the next step is to create task guidelines and examples for your annotators and guidelines are a textual description of what the annotator should do. It's important that this description is clear to the overall goal and the individual tasks of the annotation. It should be short and concise and very important, is to incorporate examples for different scenarios so that it's practically demonstrated of what should be done. 

*37.52 seconds*

### **22** Conduct a Test Run
The next step is to conduct a test run to see if everything works. So, does my setup work? Does the output data of the system is actually in the right format and contains all elements that I want? Do annotators understand the task? And is the resulting label quality sufficient or are test labels all over the place? If so, you might reconsider your task design and start from scratch. This can be performed on a small percentage of all samples and is also possible with some annotators or all annotators for a short while. After the test run, you can start your full-scale annotation campaign. 

*54.59 seconds*

### **23** Crowdsourcing
And one way of conducting such an annotation campaign is by using crowdsourcing and marketplace platforms such as Amazon Mechanical Turk or Figure Eight, which allow you to post tasks and workers then can decide if they want to complete them. You can set requirements on country, language, education level, etc. You can set how much you want to pay them, and it allows for a very large scale of annotations. So here, important to note is that scaling costs a lot of money and also the time to set up and supervise the annotations. With crowdsourcing you need a lot of focus on quality control, as cheating is common, and the task design needs to be done so that random cheating is not possible, and a large number of majority voting is necessary which drives up costs. 

*67.36 seconds*

### **24** Majority Voting 
Speaking of majority voting, it's a technique to let different annotators judge the same examples, which gives you a higher quality of the resulting data. On the other hand, you have to pay n-times more cost in terms of effort, and with randomness so that you don't actually create groups of annotators that could miss, certain groups that only do bad judgments. And this allows us to monitor the quality of individual annotators. So here, some disagreement is always there, but if one always disagrees, this might warrant a more thorough investigation and it also allows us to better understand the task and we want to solve, so how to interpret agreement ratios? Is the overall agreement high, or is it low? 

*66.29 seconds*

### **25** Evaluate Annotation Quality 
The question now is how to evaluate the annotation quality. We can measure the label quality of annotators based on their inter-annotation agreement, if we utilize majority voting, but we can also base it on a few samples that aren't labeled by an expert and then subsequently labeled by all annotators. The Inter-annotator agreement tells us if humans agree on the results and the general agreement there is an important prerequisite for an algorithmic or machine learning solution, because if humans can't agree on a solution, how is the machine supposed to learn from that? And if you want to compute the Inter-annotator agreement between two annotators, you usually use the Cohen's Kappa, and if you have two or more annotators to compare, then you use Fleiss' Kappa and the Kappa result interpretation is somewhat subjective and always depends on task, what strong, moderate, or weak agreement actually mean for him. 

*85.98 seconds*

### **26** Pooling in Information Retrieval 
Now, from those more general informations on how to create a data set, we turn to a very IR specific technique and that's called pooling. In information retrieval, we need systems to retrieve from a "full", meaning a large collection of documents, otherwise we would be not modeling the real-world conditions right. And of course, we can't annotate all documents per query, which would mean potentially millions of annotations for every single query, where most documents are not relevant. So, we get retrieval models to help us and retrieval systems proven retrieval systems create candidate documents for us to annotate, and here, if we use a diverse pool of different systems, we can then even reuse those pool candidates and this gives us confidence that we have at least some of the relevant results in those pooling results. It allows us to drastically reduce the annotation time compared to conducting millions of annotations by hand.  

*91.39 seconds*

### **27** Pooling Process (1/5)
The pooling process is as follows. We gather a diverse set of retrieval models (here in the four colors) and we index the same collection with each model. We then select a list of queries and repeat this pooling process. So, for each query we let each system retrieve their top ranked documents, passages, or elements. I'm using documents here as a term for all these sub-terms. 

*36.13 seconds*

### **28** Pooling Process (2/5)
Then, after retrieving this ranked list of documents, we choose a cutoff point K where we guarantee to include results per system. This is a very simple method that allows us to compare all those systems up to a depth of K. But of course, there are more complicated and improved methods created for pooling at various steps. But right now, we're going to keep it simple.  

*36.15 seconds*

### **29** Pooling Process (3/5)
Then to create our pool, we remove duplicated entries, so this then actually makes pooling more annotation-efficient than simply relying on a single system at a time because we save annotation time by removing duplicates. And now, those query document pairs can all be annotated. 

*26.77 seconds*

### **30** Pooling Process (4/5)
With this annotation task and the pairwise labeling, we now know for each of our pairs, or for every list, we know the label for the documents. 

*18.02 seconds*

### **31** Pooling Process (5/5)
And as the last step after every pair is annotated, we of course map the labels back to all the duplicated documents to receive our full list per system. So now we can use the ranked list evaluation metrics we heard about in the evaluation crash course to compare our systems. Hurray!  

*26.31 seconds*

### **32** Pool Bias
But what if, let's say, we did this annotation campaign a while ago, or someone else did it for us and we want to create a new system that does things differently and you want to reuse those annotations? You have to be aware of something called "pool bias" that if a new system retrieves a new set of documents, very likely this set will include unjudged documents. So, in that case, the missing documents have to be assumed non relevant, which of course puts the new model at some disadvantage in comparison to the models who participated in the annotation campaign. 

*50.13 seconds*

### **33** Annotating for Recall 
If we want to annotate focusing on recall, we kind of need to know judgments for "all" the relevant items, but we don't know which of the items is relevant beforehand. So, the test collection depends on this pre-selection of documents. Relevant items might be missing because of vocabulary mismatch or other small details. What we can do in this case is actively work on this problem with iterative annotation cycles and active learning and the HiCAL tool is such an annotation system that integrates active learning in an iterative query/document pooling process. Or at least if our test collection is not prepared for such a high recall scenario, for example, if we use sparse judgments, we should be at least aware of those limitations when interpreting all our metric results. 

*74.63 seconds*

### **34** FiRA: Fine-Grained Relevance Annotations 
I now want to highlight some annotation work our group did in 2020 and this is FiRA or Fine-grained Relevance Annotations. And here we went back to the TREC deep learning document ranking annotations and annotated every highly relevant document on a passage and word level, which gave us a total of 24,000 query-passage pairs using four classes for graded relevance.

*42.35 seconds*

### **35** FiRA-DL‘19: Relevance Classes & Majority Voting 
I'd like to highlight some of our results here concerning with different relevance classes and majority voting. Keep in mind that every document overall in this evaluation, as judged by TREC, is relevant at some point and what we find is that many passages in those relevant documents are not relevant, which gives us new intuitions for passage score aggregation techniques. And here on the left you can see that majority voting is important, especially if we look at the four level graded relevance as most relevant passengers are not unanimously decided and there is some ambiguity. 

*53.85 seconds*

### **36** FiRA-DL‘19: Relevance Uncertainty 
And this uncertainty of relevance is also highlighted in this case where we selected some passages here two, for example, that are annotated by all our annotators. So, we can qualitatively study the subjectivity of relevance annotations. So here, in this case, most agree on the general relevance between the relevant or not relevant classes. However, if we look at the four level results, it's not that clear anymore, and the heatmap of selected regions, so if a word is more saturated in green in the background, it means more annotators selected it as a relevant word. And in the upper example we can see that the uncertainty of what is actually relevant in this passage is way higher than uncertainty in the second passage, where it's pretty clear that only those two sentences are relevant to the query and the rest of the passage is not. 

*82.66 seconds*

### **37** Analyzing Biases
Speaking of biases, one important aspect of all things data is to analyze biases and also analyze the downstream impact of existing bias in the data. 

*19.02 seconds*

### **38** The Many Types of Biases 
Of course, there are many types of biases, and here in this lecture we only scratch the surface of research directions in this field. So, taking biased text as input will very likely give you bias output representations. You might not be able to easily measure the bias, but it is there, and bias can take many forms. Such as gender or racial bias, and it can have a decisive effect on people’s lives, often and easily unnoticed by the responsible people. For example, if we employ machine learning models in hiring decisions or predictive policing, or in recommendation algorithms that marginalize minorities, all those applications have a downstream effect of those machine learning models that are in the backend and learned by bias data can produce dangerous outcomes. But we also are concerned with less dangerous forms of bias, such as learning to ignore certain positions of text in a document, which is a very technical form of a bias. 

*91.58 seconds*

### **39** Bias in Word Embeddings 
Let's talk a bit more about bias in word embeddings. Word embeddings are usually trained on large scale unlabeled text and for example a common resource for that is the usually associated high quality Wikipedia. Important here is to know that if the training data is biased, the vectors will be biased as well. And in this example on the right, we can see that when Word2Vec is trained on Wikipedia, the resulting vectors contains significant gender bias with female to male analogies associating certain stereotypical words more with the female label than the male "he" word. And the good thing about word embeddings is that because we only have one vector for one word in our vocabulary, we can make those comparisons and those analysis to find the bias in that way. Various methods have been proposed to debias word embedding, however, we also have studies that show those debiasing methods might not be truly effective and only cover up the bias.  

*93.46 seconds*

### **40** Bias in (Large) Language Models 
In large language models, the bias becomes much more complicated to measure than in word embeddings. Recently, the so-called stochastic parrot paper and the subsequent refusal by Google to let it publish and all actions that followed, created quite a stir in the community in early 2021. I want to highlight this opinionated paper because it has a great survey of the state of the field, and it makes a lot of important points for researchers to think about going forward. So, the authors argue that exploding dataset sizes in training large language models, including using outdated text, will result in unchecked biases. And if you apply static stop-gap measures such as removing documents that contain certain keywords, might hide minority communities and have an adverse effect, counter to what you're actually employed for. Language models are only seemingly coherent, especially if they create text. So, it's not true. Language understanding and language models also amplify training bias in real world use. The authors finally argued that we need a mindset that weaves responsibility and thinking about where deploy certain systems in our whole research process. It is a great read; I can only recommend that you take a look at the paper. 

*122.07 seconds*

### **41** Social Impact of Ranking 
Furthermore, if we look at ranking models and recommendation models, and the social impact they can have, is always concerned with: what those models are optimized for? If for example you optimize a model for time spent on a platform, it's easy to let users fall down a rabbit hole because scandalous and "Click here to find the truth" videos etc. are more likely to keep you on the platform and thus generating more watch time. And if you look at the world as a whole, you have to be aware that of course there are many, many languages, and if you manually invest time in blocking English content or for example weird autocomplete content in English, it does not automatically translate to other languages. This problem is of course not easily solvable, but we should be aware that it exists. 

*82.96 seconds*

### **42** Technical Term-Position Bias 
Now, we turn to something a bit more technical and of course less dangerous, but still interesting. And this is the term position bias. So, in a bag-of-words model such as the traditional IR models we saw in the crash course on fundamentals, they don't know about word positions. This problem doesn't exist there. But in most neural models, a term has positional information associated with it, which then can models learn to overemphasize this positional information. If bias on the position is in data, it will be learned. For example, if you learn a bias on the start of a passage or a document. And you want to find or retrieve the same information later positioned in another passage or document, it puts this other passage and a disadvantage, even though it would contain the same information. This is a problem for generalization and also it shows somewhat a limitation of neural models and I think we should be aware of that. This is different from the result position bias that is most often associated with the term position bias where users tend to click on 1st results in a list of web search results even if relevant ones are further down the list. 

*115.26 seconds*

### **43** Where is the passage-level relevance? 
Our group tried to find the answer for where is the passage level relevance and we started off by observing a bias in the MS MARCO QA data where answers tend to involve words at the beginning of a passage. And remember the MSMARCO-Passage collection only contains one relevant judgment for query, so this bias directly follows the retrieval labels as well. And with FiRA, we conducted a rotation-based controlled experiment to see if we can replicate this phenomenon. Although we have to have here a disclaimer that it's not on the same passages because we looked at the documents, but it's the same domain. And our results show that almost uniformly selected relevant words, an only in the rotation, there is a slight drop around the corners where we did rotate the documents on the passages. 

*73.33 seconds*

### **44** Tracking Position Influence in Transformers 
We now want to do go ahead and see if we can detect this position bias in transformer-based models. And yes, we did actually find that in the TK model, which you will hear about more and as part of this course where transformers use positional information of terms with an embedding, the influence of the position is trained. Here on the left you can see our plot with a cosine similarity between same term occurrences across passages and on the x-axis, we plot the position delta between those two occurrences that are compared across passages. In the original training in red, you can see that various passage position has a very strong influence on the similarity of the same term, mind you. But if we de-bias the training data, here in blue, we observe much less positional information as part of a contextualized vector. 

*87.79 seconds*

### **45** Where is the document-level relevance? 
So finally, I want to extend this question and ask: where is the document level relevance? So where is the likelihood of relevant passages across a long document? What we find here is that relevant passages are more likely to start at the beginning of a document, although it's definitely not exclusively for the first passage, and it also extends far behind in the tail of a document. This keeps us giving you insights into "how to design neural models that work with such a long text?". 

*53.52 seconds*

### **46** Summary: Test Collections 
To summarize this talk on test collections, I want you to take away that IR test collections are incomplete, and we must deal with the uncertainty that comes from that. Judgement pairs should use pooling of many diverse system results so that we might be able to reuse them. And finally, bias exists everywhere in different forms, and it's at least important to be aware of it, and even better to act according to this bias so that it does not have negative effects on downstream applications.  

*49.5 seconds*

### **47** Thank You  
Thank you very much for your attention and I hope you enjoy this talk and see you next time. 

*8.08 seconds*

### Stats
Average talking time: 65.44125398936167

# Lecture 2 - Crash Course - Evaluation

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Hi and welcome back to our crash course on information retrieval. Today we're going to talk about evaluation and if you have any questions, comments, or anything else, please feel free to write me an email. 

*18.3 seconds*

### **2** Today
Today we will talk about three things. First, about the general setup of how we evaluate retrieval results that are represented by ranked lists, which is a little bit different to other machine learning applications where we don't have lists to evaluate. Then, we're going to talk about binary relevance metrics, so how we can actually, if we have a label that tells us if a document is relevant or if it is not relevant, how we can evaluate ranked list based on that. And then, to take it one step further, because of course there are different kinds of relevance, we are going to look at graded relevance metrics and specifically nDCG. 

*55.89 seconds*

### **3** Evaluation 
So evaluation, we evaluate systems to observe concrete evidence for a hypothesis that we pose. So, is our system better than the other one? I think that's the most basic hypothesis that we can evaluate. We want to measure if our approach is better than the baseline. Well, for information retrieval systems, this is kind of hard. First, we have to be aware that relevance is very ambiguous. So, what is relevant? In which context? For which users? And not only is this a general question, but practically humans differ a lot in their relevance judgments. So how can we say something is relevant? And then the second problem is the collection size. So if we want to annotate every query document pair that is available in the collection, we will never finish. Alright, so in general, there are two different types of result quality evaluation. First is an offline evaluation, when we have a fixed set of documents, queries and labels. And then the second one is an online evaluation. So, if you have a production system, you can observe the behavior of users for different kinds of systems. 

*93.11 seconds*

### **4** Online Evaluation 
In this online evaluation scenario, it only works if you have thousands and thousands of users to get significant results from your test. And to do that, you can run an A/B test, which means that a portion of your users are presented with search results from the novel test system and the other control group still receives results from your old system. Well, there are a couple of problems right out of the gate. So, if you have multiple concurrent tests that you want to run, you require enormously complex infrastructure to handle different changes in your retrieval pipeline. And then, which I personally find more interesting, is a philosophical question. Is it okay to degrade or potentially degrade the performance for some users that you are testing on, to be able to improve the service over the long term or is this not okay? So for example, if you have a business to business application where every user actually pays you for your service, this might not be option that you can actually experiment with your users, because every user pays you for your service. A very interesting read is linked here if you want to know more about how, for example, Microsoft is running a massive number of controlled experiments in Bing. 

*109.83 seconds*

### **5** The World of Evaluation 
So today we are focusing on the evaluation of result quality of our own information retrieval system. Right, so we want to answer the basic question: Does a document contain the answer for our query? But of course, there are many other possibilities of what we can actually evaluate. Here is just a non-complete list. Very important is the efficiency, so how fast can we index return results for a query and how large becomes our index on the disk if you want to save it. Then there are a range of other dimensions that we can evaluate on. So, how fair are results? How diverse? How good is the content quality? The source credibility? How much effort does it take for a user to get the information from a document or a snippet? And then of course retrieval can also be looked at in a larger context. So, if we are a website that actually wants to sell stuff and so we sell products and people find products via our search, then we can measure how many products can we sell through our search interface with different A/B tests for example. Or, another option is, which is a huge field on its own, is how well does our website, even if we ourselves don't do any search at all, so how well does our website, our structure, our HTML integrate with Google, Bing and other search engines when we actually want to optimize a blackbox? Yeah, if you're interested in the efficiency parts in the introduction to information retrieval course, we take a closer look at how to actually implement efficient text processing, and if that interests you, you can check out the lecture on GitHub. 

*134.42 seconds*

### **6** 
Right, now, to our main topic today, offline evaluation. How we can systematically compare retrieval models?

*10.4 seconds*

### **7** Offline Evaluation Setup I
Our systems produce a ranked list of documents, here. And we compare that by a pool of judgements, and this pool of judgements does not necessarily cover the whole list. So if we have missing judgements, we have to consider those documents as not relevant. And a judgement always is a combination of one query and one document. But the judgements themselves often do not contain a ranking, especially not in the binary setting.  

*45.65 seconds*

### **8** Offline Evaluation Setup II
So what we do is we take those judgements that we have, and we map them to the documents in our ranked retrieval result. 

*12.19 seconds*

### **9** Test Collection 
And those judgements come from test collections. So, test collection is the name of this fixed set that we're using. We have a fixed set of documents, a fixed set of queries and judgements. And those judgements usually do not cover all query-document pairs. So, the first thing that we must think about is where do our queries come from. There are different options, of course, so we can handcraft the queries, where experts think that this is a good idea to search for, to measure how good a retrieval system is. And then there are collections that contain sampled queries from actual users, that show actual information needs. And, yeah, if you want to know more about how we create test collections, how this process works, in the data lecture you will hear more about that after this lecture. 

*72.1 seconds*

### **10** Sources & Types of Judgements 
So, we must think about different types of judgements to better understand how we evaluate them. So first, we could create judgments from automatic click data if we have a big search engine with thousands of users that actually can give us insight about specific queries. But we don't have that. So, we here at the university we focus on manual annotations, where annotators manually judge query-document pairs. And here we have different types of granularities. So, first are the sparse judgements. And they are called sparse because most of the time they only have one judged document per query, judged is relevant. And well, this makes sense if you have thousands and thousands of queries, which of course you get very noisy judgements, but it covers many terms. And here the MSMARCO Training & DEV sets that we will hear more about also in the neural IR lectures are very successful with those because they can provide us with tens of thousands of training and test queries. On the other hand, dense judgements usually created in the track evaluation campaigns have hundreds of judged documents per query and here you have already a good test collection if you have 50 or more queries. So, you don't need thousands of queries, but of course the cost of creating a judgment becomes much higher than for a sparse judgement. 

*114.75 seconds*

### **11** Comparing Systems  
So now that we know about ranked list and judgments, let's say we have multiple IR systems that run on the same documents in the same query. Each of those systems gives us a result and we know which documents are relevant and which aren't. So how do we compare them? Well, evaluation metrics to the rescue. We are going to systematically compare them and not just look at a couple of qualitative examples. 

*31.46 seconds*

### **12** Precision & Recall 
So the first thing you can of course do is to calculate the precision and recall, which are two very, fairly standard methods of evaluating different kinds of results. So, the precision basically tells you how many of the returned documents are relevant. And the recall tells you, well how many relevant documents did we actually return from the pool of all relevant documents that we know about. 

*45.09 seconds*

### **13** Evaluating Recall of Search Engines 
And this brings us to an interesting question. So, if we have recall, which depends on the judgements of all relevant items, but we don't know if there is another relevant item out there. So, if a test collection depends on the pre-selection of candidate documents, relevant items might be missing from that pre-selection because for example of vocabulary mismatch. So if the document is relevant but does not contain the actual terms we search for. So, what we can do here is we can actively work on this problem with an iterative annotation cycle. And there is a pretty cool tool out there that does that, the HiCAL tool. But of course, this means even more work for our annotators. So, if a test collection is not prepared for this high recall scenario, we should at least be aware of the limitations of a test collection when we interpret recall results. 

*73.72 seconds*

### **14** Ranking List Evaluation Metrics 
So, going from precision and recall to more list-focused metrics. We're going to look at the Mean Reciprocal Rank (MRR), the Mean Average Precision (MAP) and the normalized Discounted Cumulative Gain (nDCG). And typically, we measure all those metrics at a certain cutoff at "k" of the top retrieved documents. And for MAP and recall this is typically done at 100 and at 1000, whereas for position MRR and nDCG we have a much lower cutoff, so at 5, at 10 or at 20 to kind of get the same experience as users would do. 

*60.02 seconds*

### **15** 
Let's start with a more detailed look on the binary relevance metrics MRR and MAP. 

*7.27 seconds*

### **16** MRR: Mean Reciprocal Rank 
So, MRR, the Mean Reciprocal Rank, is a metric where you could think about this metric as users look at the results from the top and they get annoyed very fast and once they found the first relevant document, they don't care about the rest. So, if we take a closer look at the MRR formula, we can see that the first part is just a mean over all queries in our query set "Q". And the metric per query is actually 1 divided by the first rank which is called the reciprocal rank. And this first rank starts at 1, so for 1 it's 1 and for second rank it's a 1/2. So MRR really puts the only focus on the first relevant document, which is very much applicable for sparse judgements, where you assume that you only have one relevant document. Or, in a scenario where you also assume that your users create queries with the assumption that they only want to know one piece of information or one document or they're satisfied with the first relevant document, right? 

*93.69 seconds*

### **17** MRR: The Reciprocal Rank 
And if we take a closer look at how this reciprocal rank actually looks like - 1 divided by x - which is plotted here, you can clearly see this strong emphasis on the first position and that after like the 10th result there are virtually no differences anymore in the rest of the results. So, it truly makes sense to only look at MRR with a cut off of 10. 

*36.55 seconds*

### **18** MRR: An Example 
And here, as an example, in our three search systems we have three different first ranks of relevant documents. In the first, we have the reciprocal rank at 1, the reciprocal at 1/3 for a second and 1/2 for the third. And I say reciprocal rank here, because the M in MRR stands for mean over multiple queries. And here we only look at the same half-sour query. 

*37.78 seconds*
 
### **19** MAP: Mean Average Precision I
The mean average precision is a much more complex evaluation measure. So here we assume that users look at the results very closely. So, every time they find a new relevant document, they look at the full picture of what has been before. In the formula, we again start with a mean over all queries in our query set and then, the upper part here is the precision for each relevant document. So, we traverse our list from 1 to "k" and for each position "i" where we find a relevant document, we take the precision up until up until this point at "i". And we divide it by the number of relevant documents that we have to get the average precision. So, MAP is hard to interpret, right? Because it does a lot of things. And it also corresponds to the area under the Precision-Recall curve, which is a visual tool to highlight the relationship between the precision and the recall measure, so you can look that up if it interests you. 

*98.89 seconds*

### **20** MAP: Mean Average Precision II
And again we look at an example for average precision. When we assume that we have two relevant documents and again the M stands for mean which is calculated over multiple queries for each system. So, for the first system we have the first relevant document at position 1. Here, the precision, of course, is 1, we have only one document and this is relevant. For the second relevant document, which is at rank 3, the precision is 2/3, right? And then in our average precision we just sum them up and divide them by the number of relevant documents, which gives us an average precision of 0.8. In the second example, we only have one relevant document, and this is at the third position, so we can kind of guess that the score will be pretty low here, which it is. So, we just take this 1/3 divided by 2 because we have relevant documents overall, but the second one has not been retrieved by system B, which gives us a pretty bad score of 0.16. And in the third system, we again have both relevant documents but at different positions than in the first one and it sums up to a much lower score alone than the first installment. So kind of it shows, that MAP cares a lot about the position that you actually find the documents in. 

*119.47 seconds*

### **21** Graded Relevance Metrics
For the binary relevance metrics, we kind of saw that they follow the same recipe, right? So, they look at an ranked list, they value earlier positions more than later positions, but one thing that's missing in the ingredients for binary relevance metrics is the importance of relevance. And this is now solved via graded relevance metrics. And here we look at nDCG, although I must mention that also MRR and MAP are initially binary, but they have versions that make use of graded relevance. 

*47.79 seconds*

### **22** Graded Relevance 
Binary relevance labels are good in the sense that they are simple. But often they are too simple. A major problem, is that of course we can have different importance of our relevance and with the binary labels we can't distinguish that. So, we use graded relevance which allows us to assign different values of relevance. This can be a floating-point value or a fixed set of classes which is commonly used in manual annotation because if you use a floating-point value and you let annotators set that value,  you won't get any agreement whatsoever. But the floating-points can still be useful if you infer the relevance from click data coming from logs. 

*58.75 seconds*

### **23** Common Graded TREC Relevance Labels  
So commonly, if you look at manual annotation, here the TREC, retrieval conference organized by NIST it sets kind of a standard. So, you have four relevance classes which are perfectly relevant, highly relevant, kind of relevant and completely irrelevant. And those classes are denoted with 3, 2, 1, 0, which are the values then used in the nDCG metric. 

*37.04 seconds*

### **24** nDCG: normalized Discounted Cumulative Gain 
Alright, nDCG: the normalized Discounted Cumulative Gain, we can think about that as users take for each document the relevance grade and the position into account and then we have to normalize that by the best possible ranking for a query. We compared the actual results with the maximum achievable results per query. The relevance is graded and nDCG at 10 is most commonly used in modern offline web search evaluation. So, very often, if you talk to people from a big search company, they will ask you about like "And, what about nDCG at 10", right? so, they're very interested in this because it kind of gives you the first result page so to say of a modern web search engine.  

*71.52 seconds*

### **25** nDCG: A Closer Look 
Let's take a closer look at the nDCG formula. We again, just as with the binary relevance metrics, start with a mean over all queries and then, we have the actual results of our query divided by the best possible sorting, kind of our ground truth. And DCG here stands for Discounted Cumulative Gain, which is very easy. So, if you think about it, the three words are just what is in the formula, so you have sum of a gain and then you discount it by the position. The gain here is the relevance value. As I showed you before, it's commonly 0 to 3, but of course you can take your own. For example, if you want to value highly relevant documents stronger, then you can increase their relevance gain. And then the position discounting in this case is a log with the base of 2. So, this is the standard implementation also used in the TREC conference, but of course there are different varieties, as you can imagine. 

*87.49 seconds*

### **26** nDCG: Position Discounting 
So if we take a look at this position discounting and compare it with the reciprocal rank of MRR that we just saw a few minutes ago, we can see that the nDCG log-based discounting is not as strong as the reciprocal rank, which I find quite interesting. But you can definitely see the same trend that you value the first position above all else. 

*34.99 seconds*

### **27** nDCG: Example 
And here in our example for nDCG we hate to make it a little bit more complicated. So ,we now assume two relevant documents where one has relevance of 3 and the other, 1. So, if we compute our ideal DCG that we're going to use for every single of our systems, we can see that we sorted the documents, so 3 gets position 1 and then the second document with relevance label 1 gets the second position. This is a completely arbitrary value, so 3.63 doesn't tell you anything. But, if we then use it in the nDCG, it actually starts to make sense. So, for our first system we have the first document with our lower relevance score and the third document with our higher relevance score. We compute then: gain divided by position discounting and then sum it up. As well as we divide the sum by our ideal DCG, that could be possibly reached if we had the right ranking, and this gives us a value of almost 0.7. For the second system, only have one document so that the second system is kind of bad, which we could see with our naked eye, right? So, we have only one relevant document, and that's not very relevant. Um, of course the nDCG score here becomes lower. But much more interesting is the third system, where again we have both relevant documents in our result list, but they are switched, and the first position is occupied by a non-relevant document. So, what does that mean? In this example, we actually get a little bit lower nDCG than system A. Although, I wouldn't, like, make a generalization out of that. It's just interesting to look at how position versus gain value intertwines with each other and the relationship between the two. 

*165.72 seconds*

### **28** Bonus: Confidence in the Evaluation
Alright, so now that we got our metrics, I'm not going to let you off just about now, we're going to talk as a surprise bonus about the confidence in our evaluation, which is measured by the statistical significance. 

*24.29 seconds*

### **29** Statistical Significance I
So, what does statistical significance actually mean? Very often, people use the phrase "Well, it's significantly different", right? And normally if non-researchers say that it's just a phrase, a common phrase. But if researchers, especially in publications, write the term significantly different, they must prove it. You can't just say it, so significantly different has a very specific meaning and this meaning is that we test whether two systems produce different rankings that are not different just by chance, so not just some random perturbation that gives us differences. Our hypothesis is that those systems are the same and now we test via a statistical significance test on a per-query basis. So, we compare average precision, reciprocal rank and nDCG per query as lists in either a paired T-test or a, for example, a Wilcoxson test. And we test those with a specific significance level, the p-value. So, we set the p-value quite low, which means that if the test returns a lower value than our threshold, we can have confidence, we can have this confidence that we can reject our hypothesis that the systems are the same, meaning that they are actually different. 

*114.89 seconds*

### **30** Statistical Significance II 
And, as in all statistical significance tests, the more queries we have, the better does this statistical significance test work and the more meaning it actually carries. So, commonly we started with a minimum of 50 queries, but of course if you have a couple of 100 queries, it is much better. Additionally, in information retrieval, we have to be aware of the multiple testing problem. So, if we publish a new model, we often compare multiple baselines and multiple model instances with each other. But if we now test every model combination for a statistical significance with multiple metrics, we increase the probability that we run into cases that are just significant by chance, because of course the p-value of a statistical significance test only gives us a probability that those systems actually produce different results and not a guarantee. So, the solution here to this multiple testing problem is the so-called Bonferroni correction, where, simply speaking, we divide the p-value by the number of comparisons that we do, which creates often a very small p-value, which means that, well, we can be more confident that a single test in those multiple tests is not by chance. And the best explanation for that is the following xkcd comic, which makes this problem very much aware if you talk about scientific results in a broader non-scientific context. 

*135.49 seconds*

### **31** Evaluating Non-Deterministic Models 
And again, so statistical significance is good if we operate on traditional retrieval systems that are deterministic. So, if you have an inverted index with BM25, you know that the same input always produces the same output, also if you tune the BM25 parameters, the two that are in there. But if you work, as we are going to do in this course, with neural networks, you initialize the neural network components with randomness and this randomness gives you different initializations for different random seeds. And here, very well, you could produce significant differences just by chance of your different initializations. And a solution to that problem is, if you have a single architecture, you run it multiple times with different initializations and then you report the mean result value and not pick one of the best out of the multiple runs that you did. Well, the problem with that is that it is very resource-intensive, and the next best thing that you definitely need to do is to at least fix the randomness for all the experiments with a fixed seed at the beginning of all experiments. And yeah, so if you find it interesting: initializations, random initializations and neural networks, you should definitely check out the paper linked here that talks about the lottery ticket that you can get with the right random initialization. 

*118.86 seconds*

### **32** Summary: Evaluation 
And with that I'm going to conclude this talk about evaluation. So, the three takeaway messages that you need to take away from having heard or read those slides is that we compare systems with a set of query and document relevance labels. Then we have binary metrics, namely MRR and MAP that are a solid foundation for evaluation, especially because binary labels are easier to create. And then, on top of that, we have graded relevance labels that allow us to do more fine-grained metrics via the nDCG metric, for example.  

*59.51 seconds*

### **33** Thank You  
Thanks for listening. I hope you enjoyed this talk and I hope to see you next time. 

*8.08 seconds*

### Stats
Average talking time: 67.72768939393939


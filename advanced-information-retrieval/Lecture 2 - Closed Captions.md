# Lecture 2 - Crash Course - Evaluation

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Hi and welcome back to our crash course on information retrieval. Today we're going to talk about evaluation and if you have any questions. Comments or anything else? Please feel free to write me an email. 

*18.3 seconds*

### **2** Today 
Today we will talk about three things. First, about the general setup of how we evaluate retrieval results that are represented by ranked lists, which is little bit different to other machine learning applications where we don't have lists to evaluate. Then we're going to talk about. Binary relevance metrics. So how we can actually, if we have a label that tells us if a document is relevant or if it is not relevant, how we can evaluate ranked list based on that and then to take it one step further? Because of course there are different kinds of relevance we are going to look at graded relevance metrics and specifically and DCG. 

*55.89 seconds*

### **3** Evaluation 
So evaluation, we evaluate systems to observe concrete evidence for a hypothesis that we posed. So is our system better than the other one? I think that's the most basic hypothesis that we can evaluate. We want to measure. If our approach is better than the baseline. Well, for information retrieval systems, this is kind of hard. First we have to be aware that relevance is very ambiguous, So what is relevant in which context for which users and? Not only is this a general question, but practically humans differ a lot in their relevance judgments. So how can we say something is relevant? And then the second problem is to collection size. So if we want to annotate every query document pair that is available in the collection, we will never finish. Alright, so in general there are two different types of result quality evaluation. First is an offline evaluation. When we have a fixed set of documents, queries and labels and then the second one is an online evaluation. So if you have a production system, you can observe behavior of users for different kinds of systems. 

*93.11 seconds*

### **4** Online Evaluation 
In this online evaluation scenario, it only works if you have. Thousands and thousands of users to get significant results from your test, and to do that you can run and a B test, which means that A portion of your users are presented with search results from the novel test system and the other control group. Still receives. Results from your old system. Well. There are a couple of problems right out of the gate, so if you have multiple concurrent tests that you want to run. You require enormously complex infrastructure to handle different changes in your retrieval pipeline. And then. Which I personally find more interesting is a philosophical question. Is it OK to degrade or potentially degrade the performance for some users that you are testing on? To be able to improve the service over the long term. Or is this not OK, so for example, if you have a business to business application where every user actually pays you for your service, this might not be. Option that you can actually experiment with your users, because every user pays you for your service. A very interesting read is linked here if you want to know more about how, for example, Microsoft is running. A massive amount of controlled experiments in Bing. 

*109.83 seconds*

### **5** The World of Evaluation 
So today we are focusing on the evaluation of result quality. Of our own information retrieval system, right? So want to answer the basic question. Does a document contain the answer for our query? But of course there are many other possibilities of what we can actually evaluate. Here is just a non complete list. Very important is the efficiency. So how fast can we index return results for a query and how large becomes our index on the disk? If you want to save it then there are a range of other. Dimensions that we can evaluate on. So how fair are results? How diverse? How good is the content quality? The source credibility? How much effort does it take for a user to get the information from a document or a snippet? And then of course retrieval can also be looked at in a larger context. So if we are a. A website that actually wants to sell stuff and so we sell products and people find products via our search. Then we can measure how many products can we sell through our search interface with different. With again with different different AP tests, for example or. Another option is, which is a huge field on its own, is how well does our website, even if we ourselves don't do any search at all. So how well does our website our structure? Our HTML integrate with Google, Bing and other search engines. When we actually want to optimize a blackbox. Yeah, if you're interested in the efficiency parts in the introduction to information retrieval course, we take a closer look at how to actually implement efficient text processing, and if that interests you, you can check out the lecture on GitHub. 

*134.42 seconds*

### **6** 
Right now to our main topic today offline evaluation how we can systematically compare retrieval models. 

*10.4 seconds*

### **7** Offline Evaluation Setup 
Our systems produce a ranked list of documents here and we compare. That by a pool of judgements, and this pool of judgements does not necessarily cover the whole list. So if we have missing judgements, we have to consider those documents as not relevant and judgement always is in combination of 1 query and one document. But the judgements themselves often do not contain a ranking. Especially not in the binary setting.  

*45.65 seconds*

### **8** Offline Evaluation Setup 
So what we do is we take those judgements that we have and we map them to the documents in our ranked retrieval result. 

*12.19 seconds*

### **9** Test Collection 
And those judgements come from test collections, so. Test collection is the name of this fixed set that we're using of fixed set of documents. A fixed set of queries and judgements, and those judgements usually do not cover all query document pairs so. The first thing that we have to think about is where do our queries come from? There are different options, of course, so we can handcrafted queries. Square experts think that this is a good idea to search for to measure how good a retrieval system is, and then there are collections that contain sampled queries from actual users. That show actual information needs. And. Yeah, if you want to know more about how we create test collections, how this process works. In the data lecture you will hear more about that after this lecture. 

*72.1 seconds*

### **10** Sources & Types of Judgements 
So we have to think about different types of judgements to better understand how we evaluate them. So first we could create judgments from automatic click data if we have a big search engine with thousands of users. That actually give can give us insight about specific queries. But we don't have that. So we here at the University we focus on manual annotations where. Annotators. Manually judge query document pairs. And here we have different types of granularity's. So first are the sparse judgements and they are called sparse because most of the time they only have one charge document per query. Judged is relevant and well, this makes sense if you have thousands and thousands of queries, which of course you get very noisy judgements, but it covers many terms. And here the M MSMARCO training and DEV sets that we will hear more about. Also in the neural IR lectures are very successful with those because they can provide us with 10s of thousands of training and test queries. On the other hand, dense judgements usually created in the track evaluation campaigns. Have hundreds of judged documents per query and here you have. Already a good test collection. If you have 50 or more queries so you don't need thousands of queries. But of course the cost of creating a judgment. Becomes much higher than for sparse judgements. 

*114.75 seconds*

### **11** Comparing Systems  
So now that we know about ranked list and judgments, let's say we have multiple IR systems that running on the same documents in the same query. Each of those systems gives us a result and we know which documents are relevant in which armed. So how do we compare them? Well, evaluation metrics to the rescue. We are going to systematically compare them and not just look at a couple of qualitative examples. 

*31.46 seconds*

### **12** Precision & Recall 
So the first thing you can of course do is to calculate the precision and recall which are two very fairly standard methods of. Evaluating different kinds of results. So the precision. Basically tells you how many of the return documents are relevant and the recall tells you well how many relevant documents did we actually return from the pool of all relevant documents that we know about. 

*45.09 seconds*

### **13** Evaluating Recall of Search Engines 
And this brings us to an interesting question. So if we have recall, which depends on the judgement of all relevant items. But we don't know if there is another relevant item out there, so. If a test collection depends on the preselection of candidate documents. Relevant items. Might be missing from that preselection because for example of vocabulary mismatch, so if the document is relevant but does not contain the actual terms we search for. So what we can do here is we can actively work on this problem with an iterative annotation cycle. And there is a pretty cool tool out there dusted. The high call tool. But of course this means even more work for our annotators. So if a test collection is not prepared for this high recall scenario. We should at least be aware of the limitations of test collection when we interpret recall results. 

*73.72 seconds*

### **14** Ranking List Evaluation Metrics 
So. Going from precision, an recall to more list focused metrics, we're going to look at the mean reciprocal rank MRR. The mean average precision MAP and the normalized discounted cumulative gain end is itchy. And typically we measure all those metrics at a certain cutoff at K of the top retrieved documents and four MAP and recall this is typically done at 100 and 1000, whereas for position MRR and NDC G. We have a much lower cutoff, so at five at 10 or at 20. To kind of get the same experience as users would do. 

*60.02 seconds*

### **15** 
Let's start with a more detailed look on the binary relevance metrics MRR and MAP. 

*7.27 seconds*

### **16** MRR: Mean Reciprocal Rank 
So MRR, the mean reciprocal rank, is a metric where you could. You could think about this metric as users. Look at the results from the top and they get annoyed very fast and once they found the first relevant document, they don't care about the rest. So if we take a closer look at the MRR formal. Formula we can see that the first part is just a mean over all queries in our query set Q. And the. Metric per query is actually 1 divided by the 1st rank which is called the rezza procul rank. And this first rank starts at one. So for one it's one and for second rank it's a half. So MRR really puts the only focus on the 1st relevant document, which is very much applicable for sparse judgements, where you assume that you only have one relevant document or. In a scenario where you also assume that you're query yet to users. Create queries. With the assumption that they only want to know one piece of information or one. Document or there satisfied with the first relevant document, right? 

*93.69 seconds*

### **17** MRR: The Reciprocal Rank 
And if we take a closer look at how this reciprocal rank actually looks like 1 / X which is plotted here. You can clearly see this strong emphasis on the first position and then after like the 10th result there are. Virtually no differences anymore in the rest of the results, so it makes it truly makes sense to only look at MRR with a cut off of 10. 

*36.55 seconds*

### **18** MRR: An Example 
And here is an example. In our three search systems, we have three different first ranks of relevant documents. In the first, we have the reciprocal rank at one, the rezza brokering at 1/3 for a second and a half for the third, and I say reciprocal rank here, because the M in M RR stands for mean over multiple queries. And here we only look at the same half sour query. 

*37.78 seconds*

### **19** MAP: Mean Average Precision 
The mean average precision. Is a much more complex even evaluation measure, so here we assume that users look at the results very closely, so every time they find a new relevant document. They look at the full picture of what has been before. In the formula, we again start with a mean over all queries in our query set. And then. The upper part here is the precision. For each relevant documents we traverse our list from one to K and for each position I where we find a relevant document. We take the precision. Up until up until this point at I. And we divide it by the number of relevant documents that we have to get the average precision. So MA P is hard to interpret, right? Um, because it does a lot of things, and. It also corresponds to the area under the precision recall curve, which is a visual tool to highlight the. Relationship between the precision and the recall measure so you can look it up if it interests you. 

*98.89 seconds*

### **20** MAP: Mean Average Precision 
And again, if we look at an example for average precision. When we assume that we have two relevant documents and again the M stands for mean which is calculated over multiple queries for each system. So for the first system we have the first relevant document at position one. Here, the precision, of course, is one we have only one document, and this is. Relevant. For the second relevant document, which is at rank 3. The precision is 2/3 right? And then in our average precision we just sum them up and divide them by the number of relevant documents, which gives us a precision average precision of 0.8. In the second example, we only have one. Relevant document and this is at the third position, so we can kind of guess that the score will be pretty low here. Which it is. So we just take this 1/3 / 2 because we have. Relevant documents overall, but the second one has not been retrieved by System B. Which gives us a pretty bad score of zero point 16. And in the third system. We again have both relevant documents, but at different positions than in the first one. An IT sums up to a much lower score alone than the first installment. So kind of it shows. Also, MA P cares a lot. About the position that you actually find the documents in. 

*119.47 seconds*

### **21** 
For the binary relevance metrics, we kind of saw that they follow the same recipe, right? So they look at an ranked list. They value earlier positions more than later positions, but one thing that's missing in the ingredients for binary relevance metrics is the importance of relevance, and this is now solved via graded relevance metrics. And here we look at. N DCG, although I have to mention that also MRR and imapi have binary are initially binary, but they have versions that make use of graded relevance. 

*47.79 seconds*

### **22** Graded Relevance 
Binary relevance labels are good in the sense that they are simple, but often they are too simple. Some major problem, instead of course we can have different importance of our relevance and with the binary labels we can't distinguish that. So we use graded relevance which allows us to assign different values of relevance. This can be a floating point value or a fixed set of classes which is commonly used in manual annotation because if you use a floating point value and you let annotators set that value, you won't get any agreement whatsoever. But the Floating Points can still be useful if you infer the relevance from click data coming from logs. 

*58.75 seconds*

### **23** Common Graded TREC Relevance Labels  
So commonly, if you look at manual annotation here, the TREC retrieval conference organized by NIST is sets kind of a standard. So you have four relevance classes which are perfectly relevant. Highly relevant kind of relevant and completely irrelevant, and those classes are denoted with three 210, which are the values then used in the NDC G metric. 

*37.04 seconds*

### **24** nDCG: normalized Discounted Cumulative Gain 
Alright, and is each in the normalized discounted cumulative gain can we can think about that as users take for each document the relevance grade and the position into account and then. We have to normalize that by the best possible ranking per query. We compared actual results with the maximum achievable results per query. The relevance is graded and ended. City at 10 is most commonly used in modern offline web search evaluation so very often. If you talk to people from. A big search company they will ask you about like an hour. What about in the city at 10, right? So they're very interested in this because it kind of gives you the 1st result page. So to say offer modern web search engine.  

*71.52 seconds*

### **25** nDCG: A Closer Look 
Let's take a closer look at the ending city formula. We again just as with binary relevance metrics, start with a mean over all queries. And then. We have the actual results of our query. Divided by the best possible sorting. Kind of our ground proof. And. Easy cheap here stands for discounted cumulative gain which is very easy. So if you think about it, the free words are just what is in the formula, so you have some of again then you discounted by the position. The gain here is the relevance value as I showed you before. It's commonly 023, but of course you can take your own. So for example, if you want to value. Highly relevant documents, stronger than you can increase their relevance gain. And then the position discounting in this case is a log. With the base of two, so this is the standard implementation also used in the track conference, but of course there are different varieties as you can imagine. 

*87.49 seconds*

### **26** nDCG: Position Discounting 
So if we take a look at this position discounting and compare it with the reciprocal rank of MRR that we just saw a few minutes ago. We can see that the ending CG log based discounting. Is not as strong as the reciprocal rank, which I find quite interesting, but you can definitely see the same trend that you value the first position above all else. 

*34.99 seconds*

### **27** nDCG: Example 
And here in our example for NDC G we have to make a little bit more complicated. So we now assume two relevant documents where one has relevance of three and the other one. So if we compute our ideal DCG that we're going to use for every single of our systems, we can see that we sorted the documents, so 3 gets position one and then document the second document with relevance label. One gets the second position. So this is a well, it's a completely arbitrary value, so 3.63 doesn't tell you anything. But if we then use it in the. And this itchy, it actually starts to make sense. So for a first system we have the first document with our lower relevant score and the third document with our higher relevance score. We compute them. Gain divided by position discounting and then sum it up. As well as we divide the sum by our ideal. DCG, that could be possibly. Reached if we had the right ranking and this gives us a value of almost 0.7. For the second system, only have one document so that the second system is kind of bad, which we could see with our naked eye, right? So we have only one relevant document, and that's not very relevant. Um, of course. The end City score here becomes lower. But much more interesting is the third system, where again we have both relevant documents in our result list, but they are switched and the first position is occupied by a non relevant document. So what does that mean? In this example, we actually get a little bit lower and ECG then system A. Although I wouldn't like. Make a generalization out of that. It's just interesting to look at how position versus gain value. Intertwines with each other and the relationship between the two. 

*165.72 seconds*

### **28** 
Alright, so now that we got our metrics, I'm not going to let you off just. About now, we're going to talk as a surprise bonus about the confidence in our evaluation, which is the which is measured by the statistical significance. 

*24.29 seconds*

### **29** Statistical Significance  
So what does statistical insignificant actually mean very often? People use the phrase well, it's significantly different, right? And normally if non researchers say that it's just a phrase in common phrase. But if researchers, especially in publications, write the terms significantly different, they have to prove it. You can't just say it so significantly different has a very specific meaning, and this meaning is that we test whether two systems produce different rankings that are not different just by chance. So not just some random perturbation that gives us differences. Our hypothesis is that the systems are the same, and now we test via a statistical significance test on a per query basis. So we compare average precision. Rezza Proco rank and NDCG per query as lists in either A paired T test or a, for example, a wilcoxson test. And. We test those with a specific significant. Sing significance level the P value. So we set the P value quite low, which means that if the test is and returns a lower value than our threshold, we can have confidence we can have the this confidence that we can reject our hypothesis that the systems are the same, meaning that they are actually different. 

*114.89 seconds*

### **30** Statistical Significance  
And. As in all statistical significance tests, the more queries we have. The better does this statistical significance test work and the more. Meaning it actually carries so commonly we started with a minimum of 50 queries, but of course if you have a couple of 100 queries, it is much better. Additionally, in information which we will, we have to be aware of the multiple testing problem. So if we publish a new model, we often compare multiple baselines and multiple model instances with each other. But if we now test every model combination for a statistical significance. With multiple metrics. We run into we. Increase the probability that we run into cases that are just significant by chance, 'cause of course the P value of a statistical significance test only gives us. A probability that those systems actually produce different results and not a guarantee. So the solution here to this multiple testing problem is the so-called Bonferroni correction, where. Simply speaking, we divide the P value by the number of comparisons that we do. Which creates often a very small P value. Which means that, well, we can be more confident that. A single test of in those multiple tests is not by chance, and the best explanation for that is the following X case D comic, which makes this problem very much. Aware if you talk about scientific results in a broader nonscientific context. 

*135.49 seconds*

### **31** Evaluating Non-Deterministic Models 
And again, so statistical significance is good if we operate on traditional retrieval systems that are deterministic. So if you have an inverted index with RPM 25. You know that the same input always produces the same output. Also, if you tune the beam 25 parameters, the two that are in there. But if you work as we are going to do in this course with neural networks, you initialize the inner network components with randomness, and this randomness gives you different initializations. Four different random seeds, and here very well you could produce significant differences just by chance of your different initializations and in solution to that problem. Is if you have a single architecture. You run it multiple times with different initializations and then you report the moon result value and not pick. One of the best out of the multiple runs that you did. Well, the problem with that is that it is very resource intensive. And the next best thing that you definitely need to do is to at least fix the randomness for all the experiments with the fixed seed at the beginning of all experiments. And yeah, so if you find an interesting initializations, random initializations in there networks, you should definitely check out the paper linked here that talks about the lottery ticket that you can get with the right random. Initialization 

*118.86 seconds*

### **32** Summary: Evaluation 
And with that I'm going to conclude this. Rock. About evaluation, so the three takeaway message messages that you need to take away from having heard or read those slides is that we compare systems with a set of query and documents relevant labels. Then we have binary metrics, namely MR&M AP that are a solid foundation for evaluation, especially 'cause binary labels are easier to create and then on top of that we have graded relevance labels that allow us to do more fine grained metrics. Via the NDC G metric for example.  

*59.51 seconds*

### **33** Thank You  
Thanks for listening. I hope you enjoyed this talk and I hope to see you next time. 

*8.08 seconds*

### Stats
Average talking time: 67.72768939393939
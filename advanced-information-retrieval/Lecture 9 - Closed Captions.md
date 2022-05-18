# Lecture 9 - Domain Specific Applications

*Automatic closed captions generated with the Azure Speech API*

### **1** Domain Specific Applications
Hello and welcome to the 9th lecture of Advanced Information Retrieval about Domain Specific Applications. Maybe you're wondering why you have hearing a different voice. My name is Sophia Althammer and I am a PhD colleague of Sebastian Hofstaetter, who normally does this lecture and my research focuses on Information Retrieval in Domain Specific Applications. This is why I'm really happy to introduce you to this topic and tell you more about the other applications. And if you have any questions you can reach out to me via email or via my Twitter account and I'm happy to answer all the questions you have. 

*38.36 seconds*

### **2** Today 
Lecture is about Domain Specific Applications. First I will introduce you to what is the specific domain and which task do we have in those specific domains and how we can we characterize them and then we will go into two domains where we recently have some research in neural information retrieval. So first one is the medical domain here we will discuss the task of systematic reviews or clinical decision support for patients. We will also introduce some evaluation campaigns like TREC Covid, TREC Biomedical Track or CLEF eHealth, where it is the main goal to get valuable datasets for the specific domains and the specific domain tasks. Furthermore, we will then discuss the applications in the legal domain. Here the tasks of prior case retrieval or eDiscovery are very important, and then we will also introduce emulation campaigns like COLIEE or the FIRE AILA evaluation campaigns.  

*65.27 seconds*

### **3** What is domain specific information retrieval? 
So first of all, we ask ourselves what is domain specific information retrieval. And so far in this lecture there was mainly web search and extractive question answering discussed, so like as Google which is a normal web search application. But there's not only Google out there, there are lots of other domains where there are the needs for search engines and information retrieval methods. One example you can see here on the right hand side which is a screenshot of the TU Vienna Library in search interface, where you can look for the books which are stored in the library or which you can borrow from the library. So this is an example for academic search because there you have books or papers from the academic domain and you also have search queries coming from academic context with the users also being within academic background. So this is an example for an application of information retrieval in the academic domain, but there are lots of other domains too. So for example, the medical, the legal or the patent domain, and here this lecture we will discuss two of them, the medical and the legal domain. 

*88.26 seconds*

### **4** What is domain specific information retrieval? 
So how is that domain specific information retrieval characterized? One characteristic is that the search is in databases of specific domains. So we look now at the medical domain, the legal domain and the scientific domain as examples and first of all there we have a specific database containing documents coming from this domain. So let's say for example in the medical domain you would have a database of clinical trials, in the legal domain you would have a database of prior cases or legal regulatories or in the scientific domain, you have a database containing all the papers published in a certain area. So that's one key characteristic. 

*48.9 seconds*

### **5** What is domain specific information retrieval? 
Another characteristic of domain specific applications are then the users and the users in the domains are users with specific backgrounds with specific knowledge. For example in the medical domain you could have doctors or medical researchers as users of those search interfaces, in the legal domain you could have legals or paralegals who are looking for regulatories or prior cases, or in the scientific domain, the users will be academics or researchers would who are trying to find prior research in their fields for example. 

*39.38 seconds*

### **6** What is domain specific information retrieval? 
And then additionally to the specific users, you also have specific tasks. So in web search it is mainly ad-hoc retrieval. So you have one short query and then you go to the answer and then most of the time you already have the answer you want to find. But in the specific domains, maybe it could be that you need to refine your queries multiple times and have a query session with multiple queries before you get to the first result or before you get to have an overview of what is out there. So for example in the medical domain you have the specific tasks off systematic reviews or clinical decision support. We will go more into detail in this in the first chapter. In the legal domain you have specific tasks like prior case retrieval or eDiscovery and then inside the scientific domain you also have specific search tasks like making a prior literature review or surveys. 

*69.68 seconds*

### **7** What is domain specific information retrieval? 
So in this domain, specific tasks have very different characteristics to web search where in web search often the query is just a short text and then you find the relevant website or documents in a short amount of time. For example in medical domain you have long exhaustive search processes, for example if you want to give a drug to a patient and you need to consider all the side effects of a drug. So you need to find all clinical trials which are about this drug and which use this drug. And it's very critical if you don't find one clinical trial, which may be relevant in this case or may be relevant for this patient with a relevant side effect, which the drug could have for using in treating the patient. So in the medical domain therefore it's very important to make an exhaustive search and find all the relevance clinical trials, for example. And then it's also very important that the search process itself is transparent and reliable so that he can explain how you find the documents which you found and how you decide when to stop the search process and how you decide when you found all relevant documents. It's also similar in the legal domain. Here you also have long search sessions which could have multiple queries, so you query not only one time, but you do multiple queries after another, which you refine, and then you hopefully find all the documents you need for example for finding evidence for a new case, how it should be decide. And it's also similar in the scientific domain. For example, if you do a literature review, you have very long search sessions with multiple queries, which you refine then you find hopefully all relevant research to this specific area. Also, what a main characteristic here is also the specific domain language of the research area. So these are all characteristics which we need to address when applying neural information retrieval bottles to tasks in those specific domains. And those characteristics make the tasks here very different. So and now we will go into another domain which is the medical domain. 

*168.34 seconds*

### **8** 
We'll discuss the tasks which are there in medical domain. We will introduce some evalution campaigns and also some solutions in the medical domain how those tasks and problems are addressed. 

*13.33 seconds*

### **9** Different textual health information 
So in the medical domain, there's a lot of different textual health information. So there's first of all patient specific information, or there's knowledge based information. And the patient specific information is common to the health practitioners working daily with patients, so these are the health records there store about the patients. So this could be structured like lab results and vital signals. But it also could be in a narrative way so where they document how the progress is going and how patient is feeling every day. So this is patient specific information which is in the medical domain and then there's also knowledge based information from let's say research. This could be from the primary source, like original research, which is published in journals, books or report or this also could be secondary presearch results like summaries of research like systematic reviews or practice guidelines. And we will also discuss systematic reviews and the process of making a systematic review in this chapter. 

*77.04 seconds*

### **10** Different textual health information 
So because of the difference of the information sources in the medical domain, there are also different tasks. So for example, for the patient specific information: a task for medical practitioners in their daily work could be clinical decision support where they have the patient specific information, but on the other side they also need some knowledge-based information to decide what is the best treatment for every patient where they also need to consider the current status of the patient, they need to consider all the diseases or the characteristics of the patient, but also what the current status of the research is and what treatments are there for the disease the patient has. And another task we will discuss is systematic reviews. They are concluding about research about experiments about clinical trials for certain drugs or for a given production request, and they're concluding the studies to find evidence and make a summary of the research to give some guidelines for then the practitioners: Which drugs to use? In which cases are which side effect could occur?

*88.84 seconds*

### **11** Systematic reviews 
So doing a systematic review is one such task in the medical domain for information retrieval and systematic review itself is a secondary study that summarises all the available data fitting for a pre-specified criteria to answer specific research question. And it does this summary by using rigorous scientific methods to minimise the bias and generate solid conclusions from which the health practitioners can make decisions for the patients. And also to verify all empirical evidence, researchers in the search step must find all publications which are relevant to the research question, so therefore it is a long and exhaustive process to do a systematic review. It's very resource and labor intensive and mostly relies on human labour. So for example, doing exhaustive, systematic review takes like six months to one year for a whole team of professional searchers who do systematic reviews and evaluate the studies, how they contributes to the overall specific research question. And then also with the exponential growing number of publications in the medical domain. It even gets harder and more resource intensive to create those systematic reviews. 

*93.2 seconds*

### **12** Systematic reviews 
And how those systematic reviews are created is followed a hierarchical approach. This is now a picture you can see from Cochrane, Austria, they made a handbook public on how to do a systematic review for interventions and made some criteria how to maintain the quality, how to maintain that all the relevant studies are in the systematic review and are found. And how to comprise the knowledge, So what they do is first that they define a search strategy and refine the research question. And then they do a first stage retrieval where most of the time they use Boolean search queries on different search interfaces, because they're a lot of online providers of different studies and not all studies are published everywhere. So what they do is that they create a Boolean query and let it run on all those different databases for clinical trials. What they also do is that they use some specific terms or categorizations like mesh terms, which they include to search for studies in this category. This is how they, in the first step, select the studies and then in the second step, the appraisal, they first screen all the abstracts and titles of those studies and determine what was relevant there. And then afterwards they do a more refine screening on the whole document level where they read the whole clinical trial and find out if it's relevant to the specific research question of the systematic review. And only after those steps, there's the synthesis of the results where they synthesize all the studies they found which are relevant to the research question and answer the research question. An example could be the effect of the COVID-19 regulations on the psychological health of 20 to 30 year olds and this could be a research question. And then for this research question they would first need to find studies, find out which of the studies they found in the first stage retrieval are relevant to the specific research question and then synthesize and find evidence for some hypothesis. 

*177.56 seconds*

### **13** Systematic reviews 
So the systematic reviews also underlie some legal requirements for the review and the review is required to be explainable and transparent, so that the people who do the review can explain how the results get found and maybe also if some relevant clinical trials are missing, they can explain why they are missing and also the search is required to be reliable and reproducible. So at a later stage they would need to be able to reproduce which studies they found and on which basis they selected, which study is included in the systematic review and which is not. And because of those requirements so far traditional Boolean search queries and lexical matching methods are dominant in this field of IR, because with this they can easily explain, for example, why some studies didn't get found because they only do lexical word matching, and if one word of the search query is not in the document, the document will not get found. But this is also a huge opportunity in this field, because neural information retrieval methods can bring more contextualized knowledge and include most contextualized knowledge in the search and can improve the search results. But at the same time, there's the requirement to make the search process explainable and reliable, and this is a very exciting opportunity in this area. 

*101.26 seconds*

### **14** Clinical decision support 
Another Information Retrieval task in the medical domain is clinical decision support and clinical decision support systems should provides clinicians, staff and also patients with knowledge, and they should also consider the patient specific information from the electronic health record of the patient and to generate with this patient specific information, specific case advice for the treatment of one patient and therefore the system should link health observations after patient together with health knowledge. So this is also an interesting information retrieval task, where you also need to consider patient specific information and also the current state of the research and combine this. So and now, after introducing the specific tasks which are there in the medical domain, we will go into some evalution campaigns. 

*62.74 seconds*

### **15** Evaluation campaigns and research datasets 
So in order to advance research in those specific domains, there is data and relevance annotations needed in order to evaluate models, compare models so and therefore there are numerous evaluation campaigns with the goal of producing research datasets. For example, there's the TREC Clinical Trials track. And here the task is to matches patients depending on the health record of the patient to clinical trials. So and yeah, therefore you can participate in this TREC and based on your participations you get back some relevance annotations of the documents or of the clinical trials which your systems returns. And this is how the researchers said advanced for decision support systems. Then there's also BioASQ that is a challenge about biomedical question answering, and there's also CLEF eHealth. They have tasks about an patient centered information retrieval, or consumer health search, which is also more concerning the task of clinical decision support, but they also provide tasks about technology assisted reviews in empiricial medicine, so here it is the task to do a systematic review or find clinical trials for systematic review using technology assisted methods like neural information retrieval methods, for example. 

*106.37 seconds*

### **16** Evaluation campaigns: TREC Covid1 
Another very awesome evaluation campaign started last year is called TREC Covid and it started because the research papers about COVID-19 were just immensely growing because there were a lot of publications about COVID-19, everyone is affected by it and a lot of clinical trials, and in order to enhance the research on IR systems to support the researchers who researched COVID-19 and to support them to keep track of all the new published clinical trials, this task was created, or this challenge was created and within this task they created a research corpus of clinical trials and studies which only relate to COVID-19. And then they also created some queries about COVID-19 like about the vaccinations and these were the test queries for this task of retrieving evidence for COVID-19 related queries.  

*63.11 seconds*

### **17** Challenges 
So, and as I already mentioned, they are specific challenges in the medical domain, which are challenges also for neural information retrieval methods. For example, you have domain specific language in the medical domain and therefore if you want to use neural information retrieval methods for the medical domain, it is really important that you also use domain specific language models. So in previous lectures there was the BERT language model introduced and exactly the same BERT model exists but for specific domains. So this language models BERT is then trained on either scientific domain or biomedical domain or clinical trials and to represent the language of the medical or biomedical studies more precisely and more informative. So there are different models like SciBERT which is trained on scientific abstracts of semantic scholar. Then there's BioBERT trained only on biomedical study abstracts and PubMedBERT which is only trained on the abstracts of clinical trials. 

*70.47 seconds*

### **18** Challenges 
Another challenge for training neural information retrieval methods for tasks in the medical domain are the few labelled data you have in those domains, and as I already said before, there are some evaluation campaigns with the goal to create more data to have more training data to use neural information retrieval methods, and then, for example, you could also take relevance labels from other sources, for example from the click logs. So if you have a search engine, you can track what the people click on and what the people click on, you can consider as relevant. And this is what for example was done for creating the TripClick dataset. This is a large corpus of medical studies containing the title and abstract of the studies and relevance annotations based on the click data of the users of the TripClick search database. So yeah, having few labelled data is really a challenge in the specific domains and another one in the medical domain is the high recall setting. So it is really catastrophic if you miss a relevant document, for example for side effect of a drug. If yeah, because it could happen that your patient has this side effect or should not get this side effect, so it is really important to find all the relevant documents or clinical trials to your given search query. 

*91.16 seconds*

### **19** Neural IR approach 
So and then the neural information retrieval approaches for tasks in those domains have a similar structure as in web search. At first there's the first stage retrieval, which is typically nowadays done in Boolean queries, but can also be approach with dense retrieval approaches which will be covered in the next lecture, and then in the second step there was the neural re-ranking and here the domain specific challenges need to be addressed like the domain specific language models or the domain specific relevance annotations or having fewer labels and these are all challenges to consider when you create a neural information retrieval method. 

*45.77 seconds*

### **20** Legal domain
A lot about applications in the medical domain. I also want to introduce you to applications of information retrieval in the legal domain and also discuss here tasks, evaluation campaigns and solutions for tasks in the legal domain. 

*14.02 seconds*

### **21** Different legal systems 
So first about what you need to consider in the legal domain is that you have very different legal systems which vary a lot from nation to nation and broad overall you can say there are two systems:the statute law system and the case law system. In the statutes law, the statutes and legal regulatories are the primary information source, and therefore it is really important if you want to solve an new legal case that you find relevant statutes and legal regulatories to this case, and this system is mostly in European countries and different to this system there's the case law system where precedent cases so like prior cases which are already decided by court, are the primary source for legal evidence and not legal regulatories. So in those systems it is really important to find all the prior cases to your case which you want to solve. And this system mainly is in the Canada, in US or in Australia. 

*69.36 seconds*

### **22** Prior case retrieval in case law systems 
So I wanted to focus now more on case law systems and namely here the task of prior case retrieval in case law systems so and the task should lead to prior cases which should be noticed for solving the current case. So let's say a lawyer has a given current case and you want to find legal evidence for solving this case and this is when you do a prior case search and here the information source is the primary literature containing the previous court decisions how prior cases got decided and desired output of the search is a list of prior cases which should be sorted by relevance or also temporal aspects because like if the decision is very old, maybe it's not relevant anymore. So then the newer the decision is, the more relevant it is. There are also some aspects which need to be considered like hierarchical aspects like from which Court does the decisions come: Is it from a high Supreme Court or just local court and some courts overrule the other ones decisions? 

*74.14 seconds*

### **23** Prior case retrieval in case law systems 
So in the aspect of this task, you have more precision oriented task and it is not as exhaustive as for example, systematic reviews where it's really important to find all the relevant documents. Here it is more important to have highly relevant cases at the top of the search and another aspect is the domain specific language of the court decisions as they are written in legal narrative forms and also legally binding text. What I already said, there's the hierarchy of decisions which needs to be considered depending from which court the decision comes, and also the temporal aspects of the decisions need to be considered as the older the decision is maybe it's not relevant anymore to your case. 

*59.72 seconds*

### **24** eDiscovery 
Another task in the legal domain is called eDiscovery, which is short for electronic discovery. In here it's the task to discover and produced critical evidence for a case in legal litigation. And this process is subject to rules of civil procedures, and those rules are agreed upon the process and you have a requesting party which makes a production request for critical evidence for legal litigation of a case, and then this process often involves review for privilege and relevance before that data is returned over to the requesting party. And in this task of eDiscovery, it is also really important to find all the critical evidence and all the critical cases to litigate a case and therefore we also are here in the high recall setting. 

*61.53 seconds*

### **25** Evaluation Campaigns and research datasets 
And similarly, as in the medical domain, there are so numerous evaluation campaigns in the legal domain to enhance the research for legal information retrieval, and also to create research datasets. For example, there's the TREC Legal track. Here is the task to do eDiscovery for a given production request. There's the COLIEE competition, which is a competition on legal information extraction and entailment for case law and statute law systems. And here they are concerned with Canadian and Japanese law systems. There's also the FIRE AILA Track, which is also about precedent, an statute retrieval, but here is only concerned for Indian case law system. 

*46.56 seconds*

### **26** Challenges 
And then because of the specific tasks in the legal domain, we also have specific challenges. So one challenge is the domain specific language in legal domain, as the statues and the cases are written in a very characteristic way and using specific terms which are specific to the legal domain. And therefore if you want to use neural information retrieval models, there's also the need to use domain specific language modeling. For example, there's a BERT variant called Legal BERT, which is only trained on legal documents from court listener and therefore is way better suited to embed legal domain language into a vector space. Then there's also the challenge of really long documents where, which is different to web search where web search and queries are very short often. But then in the legal domain, if you have tasks like prior case retrieval where you actually query what you give to the search is the whole document or the new case and you want to find similar prior cases to this one. So the actual query is one legal case and is very long and you can tackle this problem of long documents if you use neural information retrieval models by breaking the documents and the queries down to a paragraph level and using methods on the paragraph level and then aggregate those two whole documents relevance or another approach which is used is that given long documents you create a summary of the case of the long document and then you use this summary which is a shorter as input for your neural information retrieval model. And then another challenge is that the task is more precision-oriented than in systematic reviews. It's not this exhaustive, and therefore there's also the need for relevance based re-ranking in the second stage of the retrieval. 

*135.58 seconds*

### **27** Neural IR approach 
So if you want to apply neural information retrieval methods from web search to task in the legal domain, there's a similar setup as in web search. Again, there's a first stage retrieval, which is done with Boolean queries or dense retrieval approaches, which are discussed next lecture, and there's then the neural re-ranking stage where you have to consider the domain specific language of the retrieval task. You also have to consider the domain specific relevance annotations, and the challenge that you have very long queries and very long documents and you can address this challenge by again writing summaries of the cases or you do a paragraph level approach by splitting up the case on the paragraph level, and this is also what I want to introduce you. Now, with a method called BERT-PLI. 

*53.8 seconds*

### **28** Neural re-ranking: BERT-PLI1 for case law retrieval 
So BERT-PLI is a neural re-ranking method for case law retrieval and as the name already suggests, it is a BERT-based re-ranking method which re-ranks the first stage retrieved candidates for the task of prior case retrieval. And here the re-ranking is reduced to a binary classification problem. So given the query case and a candidate case, the model predicts if the two cases are relevant to each other or not. So and what is special in this model is that it has to handle the long documents. So first the query is a long document and also the candidate in the corpus is a long document and it does this or handles this long documents by splitting up the document into their paragraphs and then modelling the interaction of the candidates paragraphs and the queries paragraphs using a BERT_CAT approach. So it does this by splitting up the query case and the candidate case into its paragraphs. And then concatenating each paragraph of the query and each candidate paragraph together and then predicting the relevance between those paragraphs. So, but I will also introduce it to you in the next slide. 

*82.64 seconds*

### **29** Neural re-ranking: BERT-PLI
So here you can see the re-ranking approach of BERT-PLI. So on the left corner down on the left lower corner you can see the query document q which is split up in its paragraphs and then it's also used to retrieve the first stage candidates, the top K candidates, and then each candidate is also split up into its paragraphs. And then for each paragraph candidate pair all the paragraphs of the query document and all the paragraphs of the candidate document are concatenated with each other, and the relevance between the query paragraph and the candidate paragraph is predicted using a BERT model. And you need to use this splitting up because the BERT model has a limited input size of only 512 tokens. This is why you need to split up the whole document into its paragraphs and only model the relevance on the paragraph level. And then what you get by modeling the relevance on the paragraph level is an interaction matrix of all the query paragraphs with all the paragraphs of the candidate document. 

*75.58 seconds*

### **30** Neural re-ranking: BERT-PLI
This interaction matrix is then used to maxpool the representations for each query paragraph. And then the document level interaction is predicted by intentional recurrent neural network, and this then predicts if the query document is relevant to this candidate document or not. So and by this approach of first splitting up the cases and the queries into its paragraphs, modeling the interactions on the paragraph level, and then aggregating it on the document level and predicting the relevance on the document level, this neural information retrieval methods addresses the problem of long documents and also how to handle them with a BERT-based approach. 

*51.59 seconds*

### **31** Summary: Domain specific applications 
So now we also want to summarize what we've seen in the lecture. So there's not only web search out there, but there's also search and information retrieval task in a variety of specific domains. We have targeted two tasks in the medical domain: the systematic reviews, which are very exhaustive and take long and are a very labor intense processes. And also clinical decision support systems which focus more on the patient centered view. And then we've also introduced tasks in the legal domain, namely, the prior case retrieval and the eDiscovery. And we've shown also method how to address the challenges of long documents and of domain specific language in prior case retrieval. 

*49.71 seconds*

### **32** Thank You  
Thank you all for your attention. I hope you enjoyed the lecture and I hope you could learn a little bit more about applications of information retrieval methods for also domain specific tasks. And if you have any further questions I'm really happy to answer them in the question session. Thank you. 

*20.34 seconds*

### Stats
Average talking time: 71.98912500000002

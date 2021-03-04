# Lecture 0 - Course Introduction

*Automatic closed captions generated with the Azure Speech API*

### **1** 
All right, welcome everyone to advanced information retrieval in the summer semester of 2021. My name is Sebastian and I'm going to be your lecturer. And today we take a look at the course, the contents, and some organizational and grading aspects. 

*26.31 seconds*

### **2** AIR – 2021 
Unfortunately, this semester is going to be an unusual one again, and so to start off, I want to give you a quick overview of our Online Format and how we plan to conduct this lecture over the semester. So we're going to provide you weekly YouTube uploads of recorded lectures and we try to fit them in 45 minutes at maximum. And additionally we will provide you with PDF slides and automatically generated closed caption text if you prefer to read than to listen. We offer a more flexible grading structure than in normal times, because we all know it's challenging right now, and. To still be able to connect with all of you, we will conduct online office hours for both the exercises and lectures also every week and you can join the zoom meeting and come and chat with us. And finally, our exam type is going to be a 24 hour take home exam where we will offer you two dates and more on the exam details in a bit. If you have any problems, questions and of course feedback, you can use the provided TUWEL Forum or just write an email to me. 

*100.45 seconds*

### **4** 
Information retrieval is a very broad term that can be used in many different contexts and can mean a lot of different things and systems behind it. One can simply google the terms information and retrieval to find out what an information retrieval system is, and by conducting such a Google search, we already use an information retrieval system. Here the Google web search engine. That gives us many different aspects in in terms of result presentation for example. 

*38.59 seconds*

### **5** Information Retrieval 
In recent years, both commercial web search engines but also the academic research concerned with Information Retrieval, started to move away from simply displaying you 10 document results where you can click on them and then move to the document to a more user friendly user interface, where if you have a question that can be answered in a single statement, then this statement can be extracted from a certain web document and directly shown to the user, so you don't have to click on the web page to get the results. And here in this example that I started two years ago because I wanted to know the size of an A3 paper sheet. And in 2019. Google actually gave us the wrong result because it read a documented talked about all kinds of different paper sizes and it select the wrong paper size. Then when I repeated this experiment for last year's installment of the lecture. Google selected the same document, but now improved the capabilities of the extractive question answering system to actually provide the correct results of my query. 

*97.38 seconds*

### **6** Information Retrieval 
And now in this year where I repeated the same experiment, the user interface changed again, showing me the correct result, but also a table with different paper sizes in different formats. So it actually shows me an even clearer result of my question and this very small example, very simple example gives you an idea of the rapid progress that such extractive QA systems made in the last few years. 

*42.72 seconds*

### **7** Machine Learning 
The key to these and many other advances, is Machine Learning and in this lecture we are going to talk about Machine Learning and not AI because as this tweet very funny shows, if you write in Python, it's probably Machine Learning, but if you make a marketing slide and write it in PowerPoint, it's probably AI. So in this lecture I want to emphasize, then we focus on. Machine Learning and not AI. 

*34.32 seconds*

### **8** Machine Learning 
I also want to caution you that once you start doing Machine Learning it most of the time it looks like that you wait for a long time until your training and evaluation is finished and only at the end you know if something is wrong. 

*20.36 seconds*

### **9** Recommended Prerequisites 
Let's talk about the Recommended prerequisites, and I want to emphasize that these are only recommended but not required. So if you don't have those recommended skills, it shouldn't be a problem as long as you're motivated to learn some of them as we go along. It would be good if you have some basic Machine Learning and know-how about basic concepts and maybe also experience with one of the neural network frameworks. And of course it's good if you have Experience in reading academic papers as they have most of the time a different and more technical writing style than other sources. If you visited our Basic IR course, that's of course awesome, but it's not necessary as we will revisit the basics in a crash course. Good programming skills are always nice, but again, if you are in a group where you can support each other, it shouldn't be necessary to have a long Experience with programming. One necessity is to have an NVIDIA GPU available, but if you don't have one, you can use a free GPU from Google Colab and this service is completely free with limits in terms of how long you can use the GPU. But we designed the exercise to keep you within those limits. 

*106.81 seconds*

### **10** Some pointers to get you started … 
And if you want to get started on some background, here are some pointers to get you started. There is a great book about neural Network Methods in NLP by Yoav Goldberg and it also Contains a very good introduction to machine learning as well if you want to have an overview of the strong improvements in the paradigm shift in IR I can only recommend the recent Survey paper by Jimmy Lin at all about Bert and Beyond in Text ranking. And of course various Tutorials sites. 

*49.34 seconds*

### **11** 
Alright, let's talk about our organization and what we plan to do and how we plan to do it. 

*8.96 seconds*

### **12** Syllabus 
Now to the exciting part. The contents of our advanced information Retrieval lecture, which is divided into three blocks. First, we look at a Crash Course and revisit some of the basics in IR. Then we look at Representation Learning and finally the large block on Neural methods for IR. In the crash course we will first revisite the fundamentals and traditional indexing and scoring methods. Then we look at how we evaluate ranked lists in the information retrieval context, both with binary and gradient relevance. And finally, we look at how we create and analyze test collections to be able to conduct the evaluations. 

*64.44 seconds*

### **13** Syllabus 
In the Representation Learning block, we first start off with the basic building blocks and that is word embeddings and how we can actually represent words in the vector space and what that means for the future going forward. Then we look at different Sequence Representation methods to be able to contextualize vectors with convolutional neural networks, recurrent neural networks, and of course the currently (inaudible) pre trained Transformers architecture and from the NLP field we will do a small detour and look at Extractive QA and how we can find answer Locations in a given text. 

*59.86 seconds*

### **14** Syllabus 
In our main block of Neural IR we first look at the so called re ranking approach where we start from some early beginnings that were promising but not yet in paradigm shift where we started from straining from scratch. And then we are going to look at both efficient transformer architectures that already improved a lot on top of those early models. And then of course we will look at large BERT based language models to be able to understand the state of the art in re-ranking currently. Then we will take a look at some domain specific aspects and Caveats that come from those task changes between short passage and long document web ranking, but also I'm changing the domain altogether and look at some legal and patent domain retrieval. Then we will look at a quite new but very, very promising aspect of Neural IR, and that is to encode passages independently from a query. Index them in a nearest neighbor Index and directly retrieve the embedded query representation from this vector index. And a way to train those retrieval models is using knowledge distillation. Basically, a silver bullet technique that improves your more efficient models with the help of slower but more effective models. 

*125.39 seconds*

### **16** Lectures / Content 
The lectures are both in two well as well as my teaching repository on GitHub, where you will find additional lecture notes in form of automatically created closed captions. So if you want, please go ahead and start the repository so more people will see it and add your Content via issues and pull requests. For example, if you fix automatic closed caption errors you will get bonus points. If you add lecture notes, summaries, bugfixes, etc. of course you will also be rewarded with generous bonus points. 

*45.56 seconds*

### **17** Exercises 
We will have two exercises this semester: The first is data annotation exercise to understand the task we tried to model, but more importantly to create a test and analysis data set that all of you will use in exercise 2, which will be the larger exercise of course. And here we have two parts that are implemented using Python And PyTorch with Neural re-ranking where you will Implement and train Neural re-ranking models from scratch. And then the second part where we use a an up and coming workflow to download already trained models from HuggingFace and create an Extractive QA system with that. 

*60.44 seconds*

### **18** Exercise 1 
So for the first exercise will see that creating annotations is time consuming, and of course if we split the task it's easier and we create more data with less time spent per person. So Each student will spend only a few hours, something like 4 to 5 hours, for approximately 500 annotations and with that we will create a fine grained passage retrieval and extractive QA data set that's based on MSMARCO. And this fine grained data does not exist yet. So if we want to use something like that in Exercise 2, we have to create it ourselves. And the interesting part here is we of course pool together all the resources created in Exercise 1 for everyone to be able to use in Exercise 2. So you can then use your own work again in Exercise 2. And of course before we publish any data, we will completely anonymized it and remove any student IDs associated for the grading for exercised 1. 

*92.39 seconds*

### **19** Exercise 1 - FiRA 
For Exercise one, we actually created a specialized and simple and hopefully fun to use tool called FIRA for both mobile and desktop use and each registered student will receive a pre created account with username and password via email and if you like the annotation task, you can continue after reaching your target, so the more annotations you create, the more bonus points you get. We're going to award you 4% of the total grade per 100 extra annotations. And if you reach as a bonus above 1000 annotations in total, we will remove the minimum point requirements for Exercise 2 end the exam. 

*59.47 seconds*

### **20** Exercise 2 
In Exercise 2, first we're going to implement and train non state of the art neural re-ranking models. They are not state of the art, but they teach you how to use PyTorch, how it works. And it allows you to learn the inner workings of training, loop loss functions and tensor operations without having to wait endless amounts of time that you would have for state of the art models. And then we will use pre-trained models from the HuggingFace model hub to create an extractive QA system that now changes the aspect. So here we don't actually train the model ourselves anymore because we'll download some large BERT instance. But we only use it for inference. And if we put those two together, we create a pipeline that's necessary to get those Google like results that I showed you in the beginning. 

*75.63 seconds*

### **21** Exercise 2 
Exercise 2 will be conducted in groups of three persons managed via TUWEL and you will be evaluated together. You work in a private GitHub repository that we administer via GitHub classroom, which allows us to create a boilerplate template for you so you have an easier start. And of course we hope that this Exercise is interesting, engaging and fun, and for creative extra work we will award you Lots of bonus points to show you how we appreciate it if you have fun with the exercise. And of course, if you find and fix box in the starter code or the lecture slides, we will also give you bonus points for that. 

*58.7 seconds*

### **22** Online Exam 
The Online Exam will be a 24 hour take home Exam without any  supervision. And we will give you a paper to read and post some questions to be answered in the TUWEL Test format. This exam is completely open book, so you can go ahead and please go ahead, watch lectures, do background reading and read as much as possible. We will offer two dates which we now set on the 26th of May and the 16th of June, and both times the exam will start at noon and finish at noon on the day after. 

*56.73 seconds*

### **23** Grading 
The Grading Scheme looks as follows: For Exercise one you will get 10% of the total grade for Exercise 2 50% and for the exam 40%. You always have to pass a certain minimum requirement, except for Exercise 2 and the exam if you get enough bonus points in Exercise 1, although of course in Total you still have to get more than half the points to get a positive, great and once you're positive the grades will split as shown here. 

*41.31 seconds*

### **24** See you next week – virtually 👋  
I hope you're excited as I am for this course and if you have any feedback problems, questions, any other issues, please write me an email and we'll definitely figure it out, so hopefully see you next week, even though it's only virtual. 

*23.89 seconds*

### Stats
Average talking time: 58.59345738636364

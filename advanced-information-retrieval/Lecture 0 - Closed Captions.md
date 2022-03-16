# Lecture 0 - Course Introduction

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Hi and welcome to the 2022 edition of Advanced Information Retrieval. I'm super happy that you're interested in our lecture. My name is Sebastian and I'm happy to present this introduction today. 

*17.49 seconds*

### **2** AIR – 2022 
Let's start with some Cliff notes. So this year we're also doing our lecture completely remotely and our online format includes YouTube uploads of recorded lectures, which are approximately 45 minutes to an hour each. Each lecture comes with additional PDF slides, automatic and corrected full text transcripts, so if you don't want to listen to me, you can still read everything I said in a very nice way. We offer a flexible grading structure with various opportunities for bonus points, and we will also offer biweekly online office hours for exercises and lectures. Our lecture will include two exercises without any interviews and in 24 hour take home exam where we will offer you two different dates. Actress this year will be myself, Sebastian, as well as my colleague Sophia Althammer. 

*60.61 seconds*

### **3** AIR – 2022 
If you want to get in touch with us, we're trying out a new thing in this semester. So if you have questions regarding the content of lectures or the exercises then please open up a thread on our GitHub discussion section which you will find here and we hope this brings a better experience than just using the plain old Tool Forum. So we're happy to receive feedback on that as well. If you spot any errors or want to make corrections to the lectures or exercise assignments, then please just create an issue in our GitHub repository. We of course will award you with bonus points for spotting and fixing errors if you have any other problems, questions or feedback, please write an email to this email address here. And we will try to get back to you as soon as possible. 

*56.96 seconds*

### **4** 
Alright, so what's information retrieval information retrieval is at the heart of Internet technology, but in fact information retrieval has existed long before the Internet. Think of index cards in libraries. That was basically information retrieval, and now information retrieval is the core technology that underpins an enormous amount of activities that everyone. Of US conduct online from web search. Two curating feeds in Facebook, Instagram and Twitter information retrieval. Influences nearly everything we do online where we start out without a clear direction where we want to go. And therefore studying information retrieval is highly influential and very interesting, so I hope you also like it as much as we do. 

*68.89 seconds*

### **5** Information Retrieval 
This is just a short example of a running Google query that I started to try out in the year 2019 where I asked Google for the size of an A3 sheet of paper and here you can see it. Of course you will know that if you use Google then you don't get just a list of documents, but you get useful snippets already selected for you. Right and here we have this stronger component of language understanding, but in 2019 Google actually gave me the wrong results and showed me the size 4 and a four sheet of paper because the website they selected talked about a bunch of different paper sizes and now. They just selected the wrong one, then in 2020 they did fix their machine learning algorithms to now show the correct size in bold. It's still coming from the same website, interestingly and then. 

*74.09 seconds*

### **6** Information Retrieval 
In 2021 they changed their news in the face again. So the answer is still correct, but now you get a table that has been parsed and re displayed inside the Google layout, which gives you a couple of different size options both in millimeters and inches great.  

*25.94 seconds*

### **7** Information Retrieval 
And then in 2022. I repeated the same query. Now we get a different parse table. We still get the correct result on top, but somehow there is also now a sentence which kind of gives us the wrong result. So I would argue this even shows a slight regression. In terms of the performance that we want to see. But we still get most of the information that we requested in a very concise manner. 

*37.7 seconds*

### **8** Machine Learning 
The key to these and many other advances, is machine learning and in this lecture we are going to talk about machine learning and not AI because as this tweet very funny shows, if you write it in Python, it's probably machine learning, but if you make a marketing slide and write it in PowerPoint, it's probably AI. So in this lecture I want to emphasize that we focus. One machine learning and not. 

*34.32 seconds*

### **9** Machine Learning 
I also want to caution you that once you start doing machine learning it most of the time it looks like that you wait for a long time until your training and evaluation is finished and only at the end. You know if something is wrong. 

*20.36 seconds*

### **10** Recommended Prerequisites 
Let's talk about the recommended prerequisites, and I want to emphasize that these are only recommended but not required. So if you don't have those recommended skills, it shouldn't be a problem as long as you're motivated to learn some of them as we go along. It would be good if you have some basic machine learning and know how about the basic concepts and maybe also experience with one of the neural network frameworks. And of course it's good if you have experience in reading academic papers as they have most of the time a different and more technical writing style than other sources. If you visited our basic IR course, that's of course awesome, but it's not necessary as we will revisit the basics in a crash course. Good programming skills are always nice, but again, if you are in a group where you can support each other, it shouldn't be necessary to have. A long experience with programming. One necessity is to have an NVIDIA GPU available, but if you don't have one, you can use a free GPU from Google Call app and this service is completely free with limits in terms of how long you can use the GPU. But we designed the exercise to keep you within those limits. 

*106.81 seconds*

### **11** Some pointers to get you started … 
And if you want to get started on some background, here are some pointers to get you started. There is a great book about neural network methods in NLP by Yoav Goldberg and it also contains a very good introduction to machine learning as well if you want to have an overview of the. Strong improvements in the paradigm shift in IR. I can only recommend the recent survey paper by Jimmy Minette al about Bert and beyond in text ranking. And of course various tutorials sites. 

*49.34 seconds*

### **12** 
Alright, let's talk about our organization and what we plan to do and how we plan to do it. 

*8.96 seconds*

### **13** Syllabus 
Now until the exciting part, the contents of our advanced information Retrieval lecture, which is divided into three blocks. First we look at a crash course and revisit some of the basics in IR. Then we look at representation, learning and finally the large block on neural methods for IR in the crash course. We will first revisit the fundamentals. And traditional indexing and scoring methods. Then we look at how we evaluate. Ranked lists in the information retrieval context, both with binary and graded relevance. And finally, we look at how we create and analyze test collections to be able to conduct the evaluations. 

*64.44 seconds*

### **14** Syllabus 
In the representation learning block, we first start off with the basic building blocks and that is word embeddings and how we can actually represent words in the vector space and what that means for the future going forward. Then we look at different sequence representation methods to be able to contextualize vectors with convolutional neural networks, recurrent neural networks, and of course the currently all the rage pre trained Transformers architecture and from the NLP field. We will do a small detour and look at extractive QA. And how we can find answer locations in a given text? 

*59.86 seconds*

### **15** Syllabus 
In our main block of neural IR we first look at the so-called re ranking approach, where we start from some early beginnings that were promising but not yet a paradigm shift where we started from training from scratch. And then we are going to look at both efficient transformer architectures that are already improved a lot on top of those early models and then of course we will look at large birth based language models to be able to understand the state of the art. In re ranking currently. Then we will take a look at some domains specific aspects and caveats that come from those task changes between short passage and long document web ranking, but also. I'm changing the domain altogether and look at some legal and patent domains retrieval. Then we will look at a quite new but very, very promising aspect of neural IR, and that is to encode passages independently from a query. Index them in nearest neighbor index and directly retrieve the embedded query representation. From this vector index. And way to train those retrieval models is using knowledge distillation. Basically, a silver bullet technique that improves your more efficient models with the help of slower but. More effective models. 

*125.39 seconds*

### **16** Lectures / Content 
The lectures are both in two well as well as my teaching repository on GitHub, where you will find additional lecture notes in form of automatically created closed captions. So if you want, please go ahead and start the repository so more people will see it and add your content via issues and pull requests. For example, if you fix automatic. Close caption errors. You will get bonus points if you add lecture notes, summaries, bugfixes, etc. Of course you will also be rewarded with generous bonus points. 

*45.56 seconds*

### **17** Student Experience - YouTube 
At this point I want to highlight your experience in the lecture materials that we provide. So we we chose YouTube as our platform to provide lecture videos because we automatically create timestamps for each slide in our lectures. So now YouTube converts those timestamps into user interface improvements and those interface improvements work across devices and apps of. YouTube. And here you can now use YouTube to jump exactly to specific slight numbers, and if you don't like section you can just jump over it or go back to a specific slide if you want to repeat that. 

*48.75 seconds*

### **18** Student Experience - Transcript 
Furthermore, we also have very well flowing transcripts and again our aim is to enable fine grained navigation so every resource is slide based and our transcript. Again, allow you to jump to specific slides and then for each slide we have this running text of well formatted text where we use Azure speech recognition to output pretty much correct capitalization punctuation and we use your and your colleagues feedback on those texts, to improve and even further especially concerning acronyms. 

*56.09 seconds*

### **19** Feedback 2021 - YouTube Usage 
And we we did the same thing in 2021 and I just wanted to highlight what your colleagues from the year 2021 had to say about our use of YouTube, sorry very much liked it and I hope that this continues again this year and the timeline feature was also very noticeable and 2/3 of students actively used it. So that's why we're going to continue to provide all materials in this way. 

*34.05 seconds*

### **20** Feedback 2021 - GitHub & Transcript Usage 
Furthermore the usage of the transcripts provided some sort of puzzle for us, so we observed that a lot of students did not try out to look at the transcripts, but once they tried, most people tried it again and used more than one transcript. So please have a look at the transcripts and see if this is a format that works well for you. So once a transcript has been corrected by one of your colleagues it is really readable and provides a good alternative to listening to lectures because you then don't miss anything that we say. 

*52.42 seconds*

### **21** Exercises 
We will have two exercises this semester. The first is a data annotation exercise to understand the task we try to model, but more importantly to create a test and analysis data set that all of you will use in exercise two, which will be the larger exercise of course. And here we have two parts that are implemented using Python and PyTorch with neural re-ranking where you will implement and train neural re-ranking models from scratch. And then the second part where we use a an up and coming workflow to download already trained models from HuggingFace and create an extractive QA system with that. 

*60.44 seconds*

### **22** Exercise 1 
So for the first exercise we'll see that creating annotations is time-consuming, and of course, if we split the task, it's easier and we create more data with less time spent per person, so each student will spends only a few hours. Something like four to five hours, for approximately 500 annotations. And with that we will create a fine grained passage retrieval and extractive QA data set that's based on MSMARCO. And this fine grained data does not exist yet. So if we want to use something like that in exercise 2, we have to create it ourselves. And the interesting part here is we of course pool together all the resources created in exercise one for everyone to be able to use in exercise two, so you can then use your own work again in Exercise 2. And of course, before we publish any data, we will completely anonymize it and remove any student ID's associated for the grading of exercise 1. 

*92.39 seconds*

### **23** Exercise 1 - FiRA 
For exercise one, we actually created a specialized and simple and hopefully fun to use tool called FIRA for both mobile and desktop use and each registered student will receive a pre created account with username and password via email. And if you like the annotation task, you can continue after reaching your target, so the more annotations you create, the more bonus points you get. We're going to award you 4% of the total grade per 100 extra annotations. And if you reach as a bonus if you reach above 1000 annotations in total, we will remove the minimum point requirements for exercise two and the exam. 

*59.47 seconds*

### **24** Exercise 2 
In exercise two, first we're going to implement and train non state of the art neural re-ranking models. They are non state of the art but they teach you how to use PyTorch, how it works and it allows you to learn the inner workings of a training loop, loss functions, and tensor operations without having to wait. Endless amounts of time that you would have for state of the art models. And then we will use pre trained models from the HUGGINGFACE model hub to create an extractive QA system that now changes the aspect. So here we don't actually train the model ourselves anymore because we'll download some large bird instance. But, and we only use it for inference. And if we put those two together, we create a pipeline that's necessary to get those Google like results that I showed you in the beginning. 

*75.63 seconds*

### **25** Exercise 2 
Exercise two will be conducted as a group exercise and in contrast to previous years, this year we will need to have groups of four people because we have so many. Registrations in TISS that we can't use groups of three people as this would overwhelm our resources that we have available in this semester. I know it's not perfect and the larger group size, the more annoying the exercises get, but we still think this is a better option than to need to cut off the registration at some point. The groups will be evaluated together. So choose your teammates wisely, but we are very flexible and we know that sometimes things don't workout perfectly so we will accommodate students where team members might drop out at some point and we will please send us an email if that happens and we will try to accommodate this. You will work in a GitHub repository that's just private to your group, and this GitHub repository will be administered via GitHub classroom, which gives us great tools to give you a template repository to make the whole setup seamless. And of course we will offer you lots of bonus point opportunities, so we hope that you have fun with this exercise and for creative extra work we'll honor that and give you bonus points for that. And of course if you find and fix bugs, if you help your fellow students in the discussion forum, we will also award bonus points for that. 

*126.78 seconds*

### **26** Online Exam 
The online exam will be a 24 hour take home exam without any supervision. And we'll give you a paper to read and pose some questions to be answered in the TUWEL test format. This exam is completely open book, so you can go ahead and please go ahead, watch lectures, do background reading and read as much as possible. We will offer two dates which we now set on the 26th of May and the 16th of June and both times the exam will start at noon and finish at noon on the day after. 

*56.73 seconds*

### **27** Grading 
The grading scheme looks as follows. For exercise one you will get 10% of the total grade for exercise two 50% and for the exam 40%. You always have to pass a certain minimum requirement, except for exercise two and the exam if you get enough bonus points in exercise one, although of course in total you still have to get more than half the points to get a positive grade, and once you're positive the grades will split as shown here. 

*41.31 seconds*

### **28** See you next week – virtually 👋  
I hope you're excited as I am for this course and if you have any feedback problems, questions, any other issues, please write me an email and we'll definitely figure it out, so hopefully see you next week, even though it's only virtually. 

*23.89 seconds*

### Stats
Average talking time: 56.594725446428576

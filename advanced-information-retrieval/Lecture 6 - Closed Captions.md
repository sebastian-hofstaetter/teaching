# Lecture 6 - Transformer and BERT Pre-training

*Automatic closed captions generated with the Azure Speech API*

### **1** 
Welcome to this lecture on Transformer architecture and Bird pre training. Thanks for tuning in my name is Sebastian and if you have any questions feel free to contact me. Let's get started. 

*15.37 seconds*

### **2** Today 
Today we're going to talk about the transformer architecture and how this self attention works together with a positional encoding and then. We're going to talk about Bert Pre training in the Bert model with the masked Language modelling task, and in the very important way of sharing models and pre trained and trained models in the community. And then as one example of a downstream task of this versatile architecture, we're going to take a look at extractive question answering. 

*45.37 seconds*

### **3** Another Versatile Building Block 
The transformer architecture is just another versatile building block in our set of neural network model parts. The transformer architecture itself, similar to CNNs and R and ends in previous lectures is not task specific at all. It operates on a sequence of vectors and inputs a sequence of vectors and in outputs same sized. Sequence effectors, So what we do with it is our choice. And the transformer architecture quickly gained huge popularity and now especially pre trained models are ubiquitous in NLP and IR research and increasingly also in production systems with techniques such as quantization or knowledge distillation. Typical model sizes. Are not possible without. Current or previous generation GPU's. So if we want to run those models on GPUs that are 5 or 10 years old. We will have problems and. Because we Luckily have modern hardware, we can do those large matrix multiplications that Transformers are designed for and. Use the modern GPU's to their maximum capabilities.  

*103.82 seconds*

### **4** 
Alright, let's take a look at the transformer architecture and contextualization VR self attention. 

*10.03 seconds*

### **5** Contextualization via Self-Attention 
The general idea of contextualization is we have term representations, so we start again with some sort of word or subword embedding, and each word gets assigned A and embedding just as in the previous lectures, but now. We have a second layer underneath that and that's the contextualized representation, where each vector corresponds to a combination of all the vectors in the sequence, and Now the problem is how to learn this combination for our words. So that in this example here the word step if you only look at term representation of step on its own, it can have many different meanings and you probably are not going to pick this one. But with the contextualization? The word step here should correspond to the correct meaning. Of course, this is a very simplified example, but I hope you get the general idea. Context in this context, means also only the local sequence, so we're not looking at external knowledge bases or anything like that. And then. We also have to start right off the bat and say. The general idea here is very computationally intensive because for every. Contextualization output representation, you need to combine all other. Term representations, which means that you get an oh of N squared complexity. 

*128.66 seconds*

### **6** Transformer 
The transformer architecture contextualize is with multihead. Self attention and commonly just as with R and ends. We stack multiple layers of Transformers on top of each other. The transformer can be utilized as an encoder, but also in the encoder decoder combination. We saw in a previous lecture to generate natural language output. The. Large improvement over R and ends comes from the fact that Transformers do not require any recurrence, so the attention. Is done via a series of matrix multiplications over the sequence as one tensor, and we don't have to have any recurrence to go through the sequence, which of course makes the computation easier. There is a lot of boilerplate code that falls off. And it's much better to parallelize on modern tributs. And the transformer has initially been proposed in translation, but now it's the backbone of basically any. NLP advancement in the last year and last couple of years. 

*89.95 seconds*

### **7** Transformer – Architecture  
So. Before we run our embeddings through the self attention mechanism of the transformer architecture, we have to do one crucial step and this is we add a positional encoding. To every vector in our embedding, because now we don't have recurrence, we need some other mechanism for the model to know about the relative and absolute positions of the word in the sequence. And therefore we add this positional encoding as an elementwise addition to our embeddings, and how that looks will see in a minute. And after we added the positional encoding, each transformer layer. Has three projection layers to query key and values. So for the attention itself were looking at the query and key projection, each of those projections. It's basically a fancy way of saying linear layer and after we ran every element in our sequence through those linear layers. We matrix multiply the query and key by transposing one of them. So now. We have an interaction between the elements between each word. Inside the transformer. But we have A twist here. Now is the fun part where the multihead aspect comes in. So by convention we say that each. Mountain, each head of the query and key projections. Starts at a certain index. So basically if we have 10 heads, an hour dimension allati is 100 every 10th. Effort ends index. We start with a new head and we can. Do this by just reshaping the tensor inside the model so we're not changing the data, but we're just having a convention that yes, every 10th index we start a new head. So now we reshape our tenses to add this additional head dimension, and now everything runs basically in parallel and each head can focus on different things. After we did that, we receive our attention via a softmax, four each head. So we run the interactions through a softmax and. Then we receive a single attention value for each. Each input word for every other output word. Basically a matrix of attentions and those are multiplied with the. If the vectors that come out of the value projection. Then again, we also need to project back to the output dimensionality because we can have different total dimensions between the input and the sum of the multihead dimensions. And this output is then fed into the next transformer layer, and so on and so on until we get our contextualized representations back. That, in general, have the same dimensions as the input. And each word corresponds to a contextualized word. Of the output.  

*259.85 seconds*

### **8** Self-Attention Definition 
In the transformer paper, this attention mechanism is formally defined as follows, so we have. A again our QK&V values. Then are those projections so this formula hides quite a bit of complexity. But let's just assume we have our projections and then. We matrix multiply, query and key. And divided by a scaling factor and the scaling factor is the square root of the key embeddings dimensionality. After we ran it through the softmax, we also multiplied with a new value. So now we get out our attendant values. The thing here is we also have another part of complexity and that is sequence length that are uneven. So if we have uneven sequences, we actually apply a masked softmax. That masks out padding values so they receive zero attention so that in the end. The edge. Attention, so the sum of attention values sums up to one.  

*94.6 seconds*

### **9** Transformer in PyTorch 
Now the Transformers are as ubiquitous as they are. They have very good native support in all large libraries, and here is an example for PyTorch, where you define the type of layer and then you define how many layers you want to get your transformer module. And now with that transform module, you can run a, for example here a random. Input sequence with. Batch size, sequence length and vector sizes dimensionality through your Transformers. And those native supports are awesome because they bring a lot of speed and stability improvements. And of course they are highly tested by a lot of people. However they can in their raw form, especially in PyTorch, and can be a bit tricky to apply if you talk about masking padding and you have to transposed your input. And the transposing of the mask not always corresponds to the transposed of the input, but for that you can just follow the documentation or tutorials to get it started. 

*84.08 seconds*

### **10** Transformer – Positional Encoding 
Coming back to the positional encoding that I mentioned earlier. The initial transformer used sinusoid curves, and those look like that so each dimension. Receives as an addition. Those are in the value of one of those curves at its current location in the sequence, which means that, relatively speaking, the distance. If you measure the distance, for example with a cosine similarity, the distance on the relative distance is always the same. Between positions, but of course the absolute value is different.  

*60.16 seconds*

### **11** Transformer - Variations 
Oh well, so it would be kind of easy if we could say yeah, the one transformer paper solved all problems and we're all happy with that. Well, even though it's an incredible work and spurred a lot of new ideas. There is a lot we can fix or improve upon this initial architecture, and as you can see, a lot of people did so, so I would say this is a very non exhaustive list of different transformer variants and improvements. There is a lot of focus. On efficiency and how to handle long input to break this quadratic runtime and especially memory requirement. And if you look at the. Year values in each citation here and you can see that this speed of innovation is incredible and if you work on on an improvement to the transformer architecture, the chances are very high that someone else is doing almost the exact same thing and will publish before you. So there is an incredible rush going on at the moment.  

*89.56 seconds*

### **12** In-Depth Resources for Transformers 
The popularity of Transformers Luckily brings us also a lot of educational content, and of course it's much more than we could cover in one short lecture. So here are some pointers if you want to know more about Transformers and each of those again contains another large set of specific detailed analysis descriptions, etc. If you want to know more about it. 

*33.53 seconds*

### **13** 
We can use Transformers from scratch with a random initialization in. This usually much better if we take a pre train transformer model. 1st and then start off from this pre trained checkpoint. So now let's look at the workflows, tasks and models that make this possible. 

*25.19 seconds*

### **14** Pre-Training Motivation 
Usually we work with tasks that don't have a large training corpus, but we still want to use those large and high capacity transformer models which need a lot of data to work well. So the idea for pre training is to create a task agnostic training. That works on unsupervised sets of text, very similar to how work to make was trained. We teach the model about the meaning of words and patterns in the language, and we don't need labels for that. But we just make certain predictions about word or sentence positions. And. Once this model is pre trained with usually a lot of compute power, it can then be taken and fine tuned for a variety of tasks that work on this specific language. 

*64.78 seconds*

### **15** Masked Language Modelling 
One very commonly used technique for pretraining Transformers is the so called masked language modelling, and for that let's recall our example from before where we want to have a very good context dependent representation of the word steps. And. Pinpoint the exact meaning of the word steps in the vector space, so for that to sort of infuse that knowledge into the model, we do the unsupervised pre training with masked language modelling where we take a text sequence such as a sentence or a paragraph and we mask random words with a certain probability. And then we set up the task to try to predict the original work that has been masked. And based on the error of our model on that task, we update the weights.  

*70.94 seconds*

### **16** Masked Language Modelling 
So the training procedure for masked language modelling is as follows. In this case, we want to predict the word steps. For that we replace it with a mask token. We keep track that this position has been masked and run. Now our input sequence through the model to get contextualized representations, and those contextualized representations. Are used to predict the probability for each word in our vocabulary and based on that we can create a loss function and update the weights of the model to better match the target word. But this loss, this prediction over the full vocabulary is prohibitively expensive for large vocabularies. Because we need to create a softmax over every word, but models that use transformer models that use this pre training technique. Often use tokenizers, such as WordPiece or BytePair to split infrequent terms and into multiple tokens, and so reduced the vocabulary size. 

*81.66 seconds*

### **17** BERT 
Bert, the de facto standard big pre train transformer model that is out there right now. Very similar to how work to back a couple of years ago. Overrun everything. Bert is now doing the same. Coincidentally, both come from Google, So Bird stands for bidirectional encoder representations from Transformers. It showed large effectiveness gains out of the gate and. The more people work with it, the bigger the improvements get. So the general ingredients. That make all those gains possible are. WordPiece tokenization, so again we have a small vocabulary. Said covers infrequent terms by splitting them up into pieces, but. Very frequent terms get their own. Get their own representation so it's not like every word get split up, but only the infrequent ones get split up. Then birth models are very large, so we're looking at the base model alone has 12 transformer layers with 700 dimensions each for each vector. So you require a lot of memory to train and also to infer. Then Bert kind of uses a multi task approach where some parts of the model are reused between tasks and shared and Bert achieves that not by. Creating a special network architecture and connections inside the network but no bird achieves that by using specialized tokens and those special eyes tokens are just part of the vocabulary and only how they are employed in the pre training and fine tuning stages. Their meaning comes to life, so to speak, so the most important one is the CLS token aclass ification token. It is prepended to every sequence that's fed into bird, and it is used as a pooling operator to get a single vector per sequence. And when you have this single vector, you can just add a single linear layer. That reduces this vector to score, or a class probability. For any prediction classification or multi label task you might have based on your. Encoded words. Then the second token is the mask token, so Bert is pre trained with masked language modeling and for that it replaces some words with masked tokens and then the model is forced to predict those words. Next is separated token, so to. Allow for more. Tasks where you also need to compare two sentences together. Bird is actually trained always with two sentences together, so we have one sentence. Then the separate the token and then the other sentence. This is also augmented with additional sequence encoding's that are learned. A quick side note, Bert also learns positional encoding and does not use the does not use the fixed sinusoid curves as the original Transformers. And if you have the two sentences, you can do stuff like question answering for example, where the question is the first sentence in the fine tuning, and the answer is the 2nd sentence. But it is pre trained on. Random sentences taken out of documents. And Bert is pre trained very very long. So if you would do it on one GPU alone it would take weeks on end to do that. But of course the authors of birth didn't do that. They basically used a whole server rack full of TP use to train bird.  

*315.12 seconds*

### **18** BERT - Input 
Let's take a closer look at the input that gets fed into the bird model. So we have either one or two sentences, so we can also omit the second sentence if we want. Bert adds positional and segment embeddings on top of the token embeddings, and you can see that the position embedding differs based on position and the segment based on if it's part of the first or second sequence. And also in this example you can see if you look closely at the end of the input you can see the effects of the word piece. Tokenization were the word playing is actually split up into play and in and ING is prepended with this special double hashtag. That's word piece tokenization. Way of telling us that this is a connection. To the previous word, no other connection than that exists, so the model has to learn that on its own.  

*74.77 seconds*

### **19** BERT - Model 
The actual model is quite simple, so I don't think we need a formal definition for it. It's basically end layers of stacked Transformers. With some special things like layer normalization, GeLU activations which are like Relu activations but with as, I would say like a grace swing under zero that allows for a meaningful gradient if you have values that are negative but you don't really want to activate based on those values. Basically you want to have a way. Of pushing them into the positive range if needed. With relos you can't do that. So if Aurello is zero it. Basically is dead because the Flatline of the negative. Equals zero output. Does not allow for a gradient. Then Bert uses task so called task specific heads on top of the stacked Transformers to pool the CLS or individual token representations and this pooling most of the time means we have single linear layer that takes in the dimension of the CLS token and outputs a single score or a multi label. Score right every transformer layer receives as input the output of the previous one, just as you would in other transformer architectures. And as I said before, and I want to re emphasize this point, the CLS token is only special because we train it to be special in there is no mechanism inside the model that differentiates it from other tokens. And most in my opinion, most of the novel contributions in the Bert Model Center around pre training and especially the workflow of how other people then build up and interact with the model. 

*152.97 seconds*

### **20** BERT - Workflow 
The workflow. It's pretty simple. Someone who has a lot of hardware pre trains the model on a combination of masked language modelling and next sentence prediction. So this is something new that I haven't told you before. The next sentence prediction actually uses the CLS token to try to predict the next if the second sentence is actually. The next sentence in the document of the first sentence, or if it is randomly picked from another document. And once Bert is trained for a very long time, we can just download it and fine tuning on our task by switching the input and maybe switching the head on top of it. And to do that, there is a really awesome library called Transformers from hugging phase. Which incorporates numerous model variants. It incorporates a lot of pre trained models and a lot of simplicity to just get started with pre train transformer models. 

*83.94 seconds*

### **21** BERT++ 
And of course, just as with the transformer variance, there are now many Bert variants, so we have pre trained bird models for many languages. We have pre trained Bert models for domains like biomedical publications etc. And we also have different architectures that have a similar workflow, but maybe a different pre training regime. A little bit different. Architecture inside the Transformers to allow for bigger models, more efficient models and especially to allow for longer sequences. As Bert is capped at 512 tokens in total for both sentences. Which of course, especially in information retrieval, tends to become a problem if you want to. Score full documents with thousands of tokens, but you don't actually want to run Bert with longer sequences. As with Transformers, Burke also has a quadratic runtime in the sequence length.  

*83.2 seconds*

### **22** Pre-Training Ecosystem 
OK, so now our situation is that some research groups create those awesome models. But now the problem is of how to share them efficiently and how to share them with very low barriers in the community. Because if you have a simple one word, one vector embeddings such as more trabek, the sharing was as simples as a single text file containing both the vocabulary. And the weights of each vector, and we could simply load that format into any bigger models that we wanted, and they used mostly whitespace tokenization, which meant the choice of tokenizer also wasn't that important. But now with Bert and all the other models that are similar with Bert, we require exact model architecture's, which means the specific code. And configuration for hundreds of details. Then we need weights for 100 plus sub modules and layers that are used inside Berg. We need the specific tokenizer and support tokenization and special token handling that can be different from model instance to model instance, so a single text file doesn't work here anymore and we need another solution. 

*93.3 seconds*

### **23** HuggingFace: Transformers Library 
And one of these solutions is the hugging phase Transformers library, which started as a port of the TensorFlow implementation of Bert to PyTorch. But now it is quickly morphed into this multi use multi model multi framework library centered around Transformers. It provides awesome outside box support for tokenization. A lot of different Bert architectures, many NLP tasks, however, also note that. The support for re ranking and dense approval tasks is not there yet, but it focuses more on general NLP tasks and it's still expanding quite quickly. For example, in to speech recognition models it similar to transform model and hugging face library gained huge popularity and. In my opinion it is because in really is easy to use and the pre training ecosystem needs this for broad access to all those pre trained and trained models. 

*82.52 seconds*

### **24** HuggingFace: Model Hub 
In addition to the Transformer Library which gives you the code to run models, the HuggingFace team also created in Model Hub, where now everyone can create an account similar to GitHub and upload models to this hub, including a specific format for model definitions, an trained. Model weights. Which means that everyone can download them via the Transformers library and start using them very quickly. The data is hosted by HuggingFace, which especially for academics is. Very good as we don't have to worry about Public Storage space that is always iaccessible.  

*57.63 seconds*

### **25** HuggingFace: Model Hub 
Each model is packaged and uploaded via Git LFS, and you can also add a readme which is, like GitHub, displayed as a model card to be able to explain what you trained. And to show how easy it is, we actually also uploaded a couple of models. And if you want to know more about them you can check them out on the hugging face model Hub. 

*33.97 seconds*

### **26** HuggingFace: Getting Started 
If you want to get started, it's super easy, so here is a very short example that shows you. Yeah, you just point to the repository name and then create tokenizer, an model from the configuration store there and then you can simply. Tokenize a sentence. And pass that sentence to your model to get your encoded representations out of it. And that basically is it. For a full example, you can check out our notebook that shows you how to use one of our models. 

*50.87 seconds*

### **27** 
We know that we can easily obtain a pre trained birth model. Let's look at one example. A re downstream task out of the NLP suite of possible tasks that are usable with bird. 

*16.67 seconds*

### **28** Soooo many tasks are solvable with BERT 
Bert allows so many tasks to be solvable, so the original Bert paper evaluates on four different major tasks, but now with over 18,000 citations we can assume that some more have been evaluated. As long as your text input is lower than 512 tokens and you can pool the CLS token or learn E per term prediction. You can use Bert and of course you can also use bird as part of a larger model that does something before or after with the input. The HuggingFace model hub alone provides out of the box support for dozens of tasks with pre build pipelines and it lists over 200 datasets that are used by uploaded trained models. 

*61.02 seconds*

### **29** Extractive Question Answering 
Extractive QA is the task where your given a query and the passage or document and your task with selecting the words in the passage that answer the query, and we want to select at least one span with a start and end position that we then can extract from the original text and this extracted text can either be used. In a highlighted search UI with surrounding text also displayed or in a chat bot or audio based assistant as an answer to the given query. While the task itself is not perfectly solvable, in many cases where the query type must be specific to be answerable with a fixed text, many many types of queries are solvable with this extractive QA. And to solve the rest of the queries, we can use generative question answering where the models are tasked. To create new text with new words and a more natural, conversational style, which of course is more complex and potential for error's and biases. Because we generate new text is also higher. So in this lecture we're going to focus. On extractive QA. 

*100.29 seconds*

### **30** Extractive QA: Datasets 
There are a lot of datasets for extractive QA now popular datasets are SQuAD or NaturalQuestions and both are based on Wikipedia text. Where SQuAD contains artificially created queries based on a given passage. The NaturalQuestions set actually includes Google search queries that have been asked by real users. Both come with fairly large training and evaluation sets. And many pre trained models are available on the hugging Face Model Hub. So here in the right you can see. One example from the SQuAD data set where we have on the left side in the picture, we have our passage and on the right side we have a set of queries that are asked as part of this data set. For this given passage and the corresponding groundtruth answers as well. 

*68.45 seconds*

### **31** Extractive QA: Training 
To train and extractive keyway model we for example start off with a pre trained Bert model where we concatenate the query and passage with the pre trained special tokens. And then. We take the per term output, which is a vector out of Bert for the passage sight of the concatenation and we predicts if this token is the start or the end token of the answer, and this prediction is done by adding a single. For example a single linear layer on top of each single per term. Output and the single linear layer reduces the dimensionality from. 700 dimensions down to two dimensions given giving us a binary classification. If it's a start or an end token. And. And tokens are usually trained with gold label start positions and then. We can use beam search to find the best combination during inference of a start and end token, because if we. Create dependence from the end tokens 2D already predicted start tokens. We again have multiple paths we can take and the loss is usually based on a CrossEntropy of prediction vs ground truth label per term. And potentially this could also include a another output that. Binary says if the question or if the question is answerable with the given passage. So the second version of squat also contains examples into that direction. 

*134.49 seconds*

### **32** IR + QA = Open Domain QA 
And now. If we want to create a more realistic system where the passage is not yet given, but we have to. Look at a full collection of passengers. Given a query, we combine information retrieval and QA, and commonly this is referred to as open domain QA where we have a collection we need to index our collection and usually an IR system needs to generate candidates 1st and then our reading system that tries to extract. The answer is applied to the top candidates. And those two can be separate systems and separate models, or they can be jointly learned into a single model architecture. Having this open domain QA setting. Definitely makes the evaluation and analysis more complex as. We have many more moving parts and more decisions that we need to take both 40 evaluation as well as the model training. But this open domain QA fulfills the initial idea of this immediate answer that we showed in the search engine example. Presented in the first course introduction. 

*101.52 seconds*

### **33** Summary: Transformers & BERT 
The main key takeaway points from this lecture are that Transformers apply self attention to contextualize a sequence. The Bert Pre trained model uses Transformers and is pre trained to be easily used for downstream task and then finally. The. Very important piece of tying the ecosystem together in an open and accessible way lowers the barrier of entry for everyone and makes the use of all our work much more accessible and broadly available, which is awesome. 

*49.71 seconds*

### **34** Thank You  
With that I thank you very much for your attention and I hope you enjoy this talk and tune in next time. See ya. 

*9.36 seconds*

### Stats
Average talking time: 83.15818014705884
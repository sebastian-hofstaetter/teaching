# Hi there 👋 Welcome to my teaching materials!

I'm working on Information Retrieval at the Vienna University of Technology (TU Wien), mainly focusing on the [award-wining](https://twitter.com/s_hofstaetter/status/1446193420222050309) master-level Advanced Information Retrieval course. I try to create engaging, fun, and informative lectures and exercises &ndash; both in-person and online!

*Please feel free to open up an issue or a pull request if you want to add something, find a mistake, or think something should be explained better!*

### Contents

- [Advanced Information Retrieval 2021 & 2022](#air-2022)

- [Our content creation workflow](#workflow)

----

<a id="air-2022"></a>
## Advanced Information Retrieval 2021 & 2022

**🏆 Won the Best Distance Learning Award 2021 @ TU Wien**

Information Retrieval is the science behind search technology. Certainly, the most visible instances are the large Web Search engines, the likes of Google and Bing, but information retrieval appears everywhere we have to deal with unstructured data (e.g. free text).

**A paradigm shift.** Taking off in 2019 the Information Retrieval research field began an enormous paradigm shift towards utilizing BERT-based language models in various forms to great effect with huge leaps in quality improvements for search results using large-scale training data. This course aims to showcase a slice of these advances in state-of-the-art IR research towards the next generation of search engines. 

----

**New in 2022:** Use GitHub Discussions to ask questions about the lecture!

----

![Syllabus](advanced-information-retrieval/air-syllabus.png)
*The AIR syllabus overview*

### Lectures

In the following we provide links to recordings, slides, and closed captions for our lectures. Here is a complete playlist [on YouTube](https://www.youtube.com/playlist?list=PLSg1mducmHTPZPDoal4m59pPxxsceXF-y).

| Topic                  | Description                                                                     | Recordings  | Slides  | Text  |
| -------------          | -------------                                                                   |-------------                 | -----       | -----         |
| **0:**&nbsp;Introduction 2022        | Infos on requirements, topics, organization                                     | [YouTube](https://youtu.be/RBgyq7nYc3M) | [PDF](advanced-information-retrieval/Lecture&#32;0&#32;-&#32;Course&#32;Introduction.pdf)            | [Transcript](advanced-information-retrieval/Lecture&#32;0&#32;-&#32;Closed&#32;Captions.md)            |
| **1:**&nbsp;Crash&nbsp;Course&nbsp;IR&nbsp;Fundamentals     | We explore two fundamental building blocks of IR: indexing and ranked retrieval | [YouTube](https://youtu.be/ZC94KSDd4DM) | [PDF](advanced-information-retrieval/Lecture%201%20-%20Crash%20Course%20-%20Fundamentals.pdf)            | [Transcript](advanced-information-retrieval/Lecture&#32;1&#32;-&#32;Closed&#32;Captions.md)
| **2:**&nbsp;Crash&nbsp;Course&nbsp;IR&nbsp;Evaluation     | We explore how we evaluate ranked retrieval results and common IR metrics (MRR, MAP, NDCG) | [YouTube](https://youtu.be/EiDltQZ713I) | [PDF](advanced-information-retrieval/Lecture%202%20-%20Crash%20Course%20-%20Evaluation.pdf)            | [Transcript](advanced-information-retrieval/Lecture&#32;2&#32;-&#32;Closed&#32;Captions.md)
| **3:**&nbsp;Crash&nbsp;Course&nbsp;IR&nbsp;Test&nbsp;Collections     | We get to know existing IR test collections, look at how to create your own, and survey potential biases & their effect in the data | [YouTube](https://youtu.be/pRRveh3D0pI) | [PDF](advanced-information-retrieval/Lecture%203%20-%20Crash%20Course%20-%20Test%20Collections.pdf)            | [Transcript](advanced-information-retrieval/Lecture%203%20-%20Closed%20Captions.md)
| **4:**&nbsp;Word&nbsp;Representation&nbsp;Learning     | We take a look at word representations and basic word embeddings including a usage example in Information Retrieval| [YouTube](https://youtu.be/f3nM6DKVwug) | [PDF](advanced-information-retrieval/Lecture%204%20-%20Word%20Representation%20Learning.pdf)            | [Transcript](advanced-information-retrieval/Lecture%204%20-%20Closed%20Captions.md)
| **5:**&nbsp;Sequence&nbsp;Modelling     | We look at CNNs and RNNs for sequence modelling, including the basics of the attention mechanism. | [YouTube](https://youtu.be/7Bfj_UuJh38) | [PDF](advanced-information-retrieval/Lecture%205%20-%20Sequence%20modelling%20in%20NLP.pdf)            | [Transcript](advanced-information-retrieval/Lecture%205%20-%20Closed%20Captions.md)
| **6:**&nbsp;Transformer&nbsp;&&nbsp;BERT     | We study the Transformer architecture; pre-training with BERT, the HuggingFace ecosystem where the community can share models; and overview Extractive Question Answering (QA). | [YouTube](https://youtu.be/Mt7UJNKxscA) | [PDF](advanced-information-retrieval/Lecture%206%20-%20Transformer%20and%20BERT%20Pre-training.pdf)            | [Transcript](advanced-information-retrieval/Lecture%206%20-%20Closed%20Captions.md)
| **7:**&nbsp;Introduction&nbsp;to Neural&nbsp;Re&#8209;Ranking     | We look at the workflow (including training and evaluation) of neural re-ranking models and some basic neural re-ranking architectures. | [YouTube](https://youtu.be/GSixIsI1eZE) | [PDF](advanced-information-retrieval/Lecture%207%20-%20Introduction%20to%20Neural%20Re-Ranking.pdf)            | [Transcript](advanced-information-retrieval/Lecture%207%20-%20Closed%20Captions.md)
| **8:**&nbsp;Transformer&nbsp;Contextualized Re&#8209;Ranking     | We learn how to use Transformers (and the pre-trained BERT model) for neural re-ranking - for the best possible results and more efficient approaches, where we tradeoff quality for performance. | [YouTube](https://youtu.be/Fle-jKzV-Rk) | [PDF](advanced-information-retrieval/Lecture%208%20-%20Transformer%20Contextualized%20Re-Ranking.pdf)            | [Transcript](advanced-information-retrieval/Lecture%208%20-%20Closed%20Captions.md)
| **9:**&nbsp;Domain&nbsp;Specific&nbsp;Applications *Guest&nbsp;lecture&nbsp;by&nbsp;@sophiaalthammer*    | We learn how about different task settings, challenges, and solutions in domains other than web search. | [YouTube](https://youtu.be/rHXTpHIiq6U) | [PDF](advanced-information-retrieval/Lecture%209%20-%20Domain%20Specific%20Applications.pdf)            | [Transcript](advanced-information-retrieval/Lecture%209%20-%20Closed%20Captions.md)
| **10:**&nbsp;Dense&nbsp;Retrieval ❤&nbsp;Knowledge&nbsp;Distillation | We learn about the (potential) future of search: dense retrieval. We study the setup, specific models, and how to train DR models. Then we look at how knowledge distillation greatly improves the training of DR models and topic aware sampling to get state-of-the-art results. | [YouTube](https://youtu.be/EJ_7Gx6amt8) | [PDF](advanced-information-retrieval/Lecture%2010%20-%20Dense%20Retrieval%20and%20Knowledge%20Distillation.pdf)            | [Transcript](advanced-information-retrieval/Lecture%2010%20-%20Closed%20Captions.md)

### Neural IR & Extractive QA Exercise

In this exercise your group is implementing neural network re-ranking models, using pre-trained extractive QA models, and analyze their behavior with respect to our FiRA data.

📃 [To the 2021 assignment](advanced-information-retrieval/neural-ir-exercise-2021/Assignment.md)

📃 [To the 2022 assignment](advanced-information-retrieval/neural-ir-exercise-2022/Assignment.md)

----

<a id="workflow"></a>
## Our Time-Optimized Content Creation Workflow for Remote Teaching

Our workflow creates an engaging remote learning experience for a university course, while minimizing the post-production time of the educators. We make use of ubiquitous and commonly free services and platforms, so that our workflow is inclusive for all educators and provides polished experiences for students. Our learning materials provide for each lecture: 1) a recorded video, uploaded on YouTube, with exact slide timestamp indices, which enables an enhanced navigation UI; and 2) a high-quality flow-text automated transcript of the narration with proper punctuation and capitalization, improved with a student participation workflow on GitHub. We automate the transformation and post-production between raw narrated slides and our published materials with custom tools.

![Workflow Overview](workflow/workflow-overview.png)

Head over to our [workflow folder](workflow) for more information and our custom python-based transformation tools. Or check out our full paper for an in-depth evaluation of our methods published at the SIGCSE Technical Symposium 2022: 

*A Time-Optimized Content Creation Workflow for Remote Teaching*  Sebastian Hofstätter, Sophia Althammer, Mete Sertkan and Allan Hanbury https://arxiv.org/abs/2110.05601

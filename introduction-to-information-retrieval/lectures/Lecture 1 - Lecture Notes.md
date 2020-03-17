# Lecture 1 - Foundations of Information Retrieval - Lecture Notes

*These are some additional notes on the Foundations of IR lecture [slides](Lecture%201%20-%20Foundations%20of%20Information%20Retrieval.pdf). Currently this document contains mostly descriptions of slides with figures on them and does not cover the whole lecture. Please feel free to contribute anything you think is missing or fix anything incorrect!*

## Motivation

What is the essence of Information Retrieval?

**Is a document relevant to a given query and if so by how much**

The first challenge is that we want a fast answer over a set of many (million, billions..) documents. We cannot read them one-by-one and decide if they are relevant. We have to find a way to dramatically reduce the needed computation time. 
	
	"Find the needle in the haystack!"
	
For now lets assume we deal with a static set (collection) of text documents. And the documents only have one field that we want to search in. 

The next challenge: How can an algorithm decide/describe/model this relevance? E.g. Does a document satisfy the information need of the user and does it help complete the user’s task?

Intuition: If a query word appears more often in a document, the document is more relevant to the query. Additionally, computers are very good at counting things. In longer documents words tend to appear more, so we also take the document length into account among other statistics. 

Creating those statistics at query time would contradict the first challenge, however all common statistics can be computed query-independent at indexing time. The data structure we use to hold and access these statistics is called inverted index.


## Inverted Index

Inverted index allows us to access all needed statistics per term. It is not a relational database that stores the full data - the documents are stored in another part of the system (or just linked to).

The index stores two main parts (in its simplest form): document data and term data (depending on which key you use to retrieve stuff: document id or term id/characters). The term data houses a dictionary/mapping structure that maps a set of characters (the term) to a "posting list". This posting list contains frequency (and potentially positional info) for the keyed term for every document that the term appears in. A posting list in its most basic form is an array of integer tuples. Different posting lists have different lengths, depending on the document frequency count of the term.

The inverted index is built up document per document: Each document's metadata is registered and the contents go through a linguistic pipeline, which includes at least a tokenizer that splits up the stream of characters in a stream separated tokens (words we want to index). The pipeline commonly also includes case folding, stemming, and normalizations. But this can vary from domain to domain. Each token is added to the term dictionary and an entry in the posting list for the current document is created or updated.


## Search

Now that we have created our index we want to use it to search for documents! Now the challenges we talked about in the motivation and the inverted index data structure really come together 🙂. Given a query, we run it through the same pipeline as the documents during the index creation to get a list of tokens, then we retrieve the statistics (e.g. posting lists) for these tokens and compute a score for each document that is in the posting lists via a scoring model. Now we have a sortable list of document ids and scores. We sort it by the score value to get the most relevant documents. Now we can either leave it at that and return the user just the document ids or we can for example read the documents and display the contents to the users. We can then build functionality on top of that: snippets, highlighting, paging etc.. 


### Dictionary data structures

**Finite state transducers**

Dictionary data structure used by Lucene (and subsequently used by Solr + Elasticsearch)


Additional readings:

- FSTs in Lucene by Lucene committer Michael McCandless http://blog.mikemccandless.com/2010/12/using-finite-state-transducers-in.html

- Code documentation: https://lucene.apache.org/core/7_5_0/core/org/apache/lucene/util/fst/package-summary.html

- Examples using Lucene's FST (Automata) by Doug Turnbull https://opensourceconnections.com/blog/2013/02/21/lucene-4-finite-state-automaton-in-10-minutes-intro-tutorial/

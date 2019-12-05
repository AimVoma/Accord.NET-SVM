Accord.NET-SVM Repo

###### Project Description

Machine Learning Application that performs Sentiment Classification(Fine-Grained, emotions)
with sparse text representations(TFIDF) or pre-trained dense word vectors(Word2Vec),
on Supervised Linear Model(SVM). The Classification result is later dumped
in local storage as a Confusion Matrix(CM) Analysis. The implementation of the project 
was based on Accord.NET, a Machine Learning Framework written completely in Csharp for
production-grade application development.

###### App.Config
Basic Configurations of IO operations
Text Representations(TFIDF-W2V)
SVM model setup

###### Forms of Text Representations
* TFIDF - Sparse
* Word2Vec(W2V) - Dense

###### Forms of Linear Optimization Function/Solver
* SMO(Sequential Minimization Optimization)
* LCD(Linear Coordinate Descent)

###### Prerequisite:[Accord.NET](http://accord-framework.net/)

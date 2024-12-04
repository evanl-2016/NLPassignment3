# NLP Assignment 3

## Sub-task 1 - HMM POS tagging

Call the constructor for the HMM tagger to begin the training process.  Call with these parameters:
- conlluFile - The directory of the conllu file from which the corpus can be read.
- label - Either "xpos" or "upos" depending on the type of lable you would like the tagger to train on and generate. See https://universaldependencies.org/format.html for more information
- unknown_threshold - Some float indicating the frequency threshold for words in the corpus that will be dropped. 
- convergence-threshold - Some float indicating the maximum difference between iterations that can be called convergence, therefore stopping the algorithm
- test - A boolean that, when true, stops the training process at roughly 10% of the corpus, this can be used for testing and evaluation purposes.

## Sub-task 2 - K-means clustering

Call the kMeansClustering function and pass in these parameters:
- path - The path where the conllu file containing the corpus can be read
- k - The desired number of clusters
- test - A boolean, which, if true runs the algorithm on the first 100 sentences.  This is handy for testing the algorithm without running it for the full time BERT needs to tokenize the entire dataset.
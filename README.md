# Applied-ML-Final-Project

This project is aimed at the problem of genre classification of books using their summaries. The methodology uses both supervised and unsupervised techniques, and is based on the CMU Book Summary Database.

The data can be found at: https://www.cs.cmu.edu/~dbamman/booksummaries.html

For the use of GloVe embeddings, you would also need to download the pre-trained embeddings from https://nlp.stanford.edu/projects/glove/ and load the 50 D one. 

1 - Data Ingestion - this script is focused on loading the data, cleaning it, and summary statistics

2- Embeddings - this script is focused on training and loading the various embeddings

3 - Model Training - this script is focused on training all the models across every embedding

4 - plots - This script is focused on outputting the plots of the results of the embedding + model pairs

5 - kmeans - this script is focused on the unsupervised k-means clustering

# ImageIntelligence ML Tech Test
My solutions to the ImageIntelligence ML Tech Test.

My solutions are based on the observation that the provided face embeddings are obtained by Facenet, making the embeddings themselves very representative and discriminative. In particular, the Facenet is trained using triplet loss and finetuned using carefully chosen hard negatives. This means that the embeddings have a plausible and desirable property: in the 512-dimensional space where the embeddings live in, the distance between the embeddings of the same persons are much smaller compared to that of the different persons, with respect to the distance measure of Squared Euclidean Distance. 

Below are my solutions to the three tasks. If not otherwise specified, all the distance measure used here are Squared Euclidean Distance.

## Task 1
For this standard classication task I present 3 solutions:
1. Simple Nearst Neighbour. The distance between every testing sample and every training sample is computed. Every testing samples are simply assigned a label same as its nearst neighbour across all training samples. There does not exist a explicit 'training' phase nor does there exist a explicit 'classifier'. The downside of this method is that it requires computation bwtween every testing and training samples. It does not scale well with the size of the dataset. The classification accuracy is 0.99 on randomly chosen train-test splits.

2. Cluster Center. Find the cluster center for every class using training samples by calculating the mean. The process of finding the centers is defined as the training phase and the obtained cluster centers together form the classifier. During testing phase, for a unknown sample, simply calculate the distance between every cluster centers and assign the cluster with the smallest distance as the label to the sample. This method scales well with the size of the dataset. The classification accuracy is 0.99 on randomly chosen train-test splits.

3. Simple Neural Network. Construct a simple 2 layer fully connected neural network. Good scalability. The classification accuracy is 0.99 or 1 on randomly chosen train-test splits.

## Task 2
Based on the observations in Task 1, I decided to adapt the Cluster Center method for this task, because it is fast, accurate, and avoids the parameter tuning of training a neural network. The method try to find a optimal threhold through training, which is applied during testing to determin if a sample belongs to one of the known classes or is unknown. The classification accuracy is 0.98-0.99 on randomly chosen train-test splits (unknown is treated as a new class).

## Task 3
I chose the use agglomerative clustering algorithm to better explore the characterristic of the given embeddings. The algorithm is distance based bottom-up method, which is suitable in this situation. The clustering accuracy is 0.95 using all samples. The figure below shows how the number of samples per person affect the grouping accuracy.

Using all samples, the algorithm takes approximately 5 seconds to run. 

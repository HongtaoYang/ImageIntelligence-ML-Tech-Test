# ImageIntelligence-ML-Tech-Test
My solutions to the ImageIntelligence ML Tech Test.

My solutions are based on the observation that the provided face embeddings are obtained by Facenet, making the embeddings themselves very representative and discriminative. In particular, the Facenet is trained using triplet loss and finetuned using carefully chosen hard negatives. This means that the embeddings have a plausible and desirable property: in the 512-dimensional space where the embeddings live in, the distance between the embeddings of the same persons are much smaller compared to that of the different persons, with respect to the distance measure of Squared Euclidean Distance.

Below are my solutions to the three tasks.

## Task 1

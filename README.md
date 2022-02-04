# IMMMF-Recommender-system
Iterative Maximum Matrix Factorization - Movie Recommender System

## Recommender Systems
![image](https://user-images.githubusercontent.com/58035908/152573224-d60e83e9-854f-4bf0-bc62-58d8311f4187.png)

**Content-Based Recommendation System**
If a user is watching a movie, the device will look for other films with similar content or that are in the same genre as the one they are watching. There are a number of fundamentals attributes that are used to compute similarity when looking for material that is identical.
The method computes distances between the movies to determine their similarity

![image](https://user-images.githubusercontent.com/58035908/152573623-f1885ffa-522d-4999-bcb5-7d6679e930d1.png)

**Collaborative Filtering** 
It examines the preferences of similar users and makes recommendations. Having a large amount of knowledge about users and products, will make the system more effective. 
Collaborative Filtering can be divided into Memory-Based Collaborative Filtering and Model-Based Collaborative filtering:
Memory-based algorithms approach the collaborative filtering problem by using the entire database of user - item interactions. It tries to find users that are similar to the active user, and uses their preferences to predict ratings for the active user. They are not always as fast and scalable, in the context of very large datasets.
Model-based recommendation systems involve building a model based on the dataset of ratings.
Ratings are extracted from the dataset, and it is used to build a “model” to make recommendations without having to use the complete dataset every time. This approach potentially offers the benefits of both speed and scalability.

**Motivation**
The proposed method draws motivation from research on multi-class classification.
The approach is used to decompose a multiclass problem into multiple, independently trained, binary classification problems and to combine them appropriately so as to form a multiclass classifier.


Using the proposed Hierarchical Maximum Margin Matrix Factorization (HMMMF) method to:
Elucidate the matrix completion problem by overcoming the challenges of sparse data.
Handle the noise present in the training data.


Example:
![image](https://user-images.githubusercontent.com/58035908/152573859-65dcfb6d-efc5-4e53-8156-dda051acba9f.png)

Features:
![image](https://user-images.githubusercontent.com/58035908/152573898-8967127d-3e10-4de9-a08d-cbe87818509b.png)

Ratings depend on features:
![image](https://user-images.githubusercontent.com/58035908/152573990-5218116f-1693-4f1a-9fd4-2f2e20179157.png)

Why Matrix Factorization?
![image](https://user-images.githubusercontent.com/58035908/152574105-8808c2fa-0139-450f-b1a6-57ea11961c59.png)

![image](https://user-images.githubusercontent.com/58035908/152574135-96d0ad42-5fd8-4853-84c0-38dce81433e8.png)

What is Maximum Margin MF?
![image](https://user-images.githubusercontent.com/58035908/152574218-92b1f2b3-8724-49db-95c1-7a7dbcf15948.png)

![Alt Text](https://machinelearningjourney.com/wp-content/uploads/2020/01/Linear3D.gif)





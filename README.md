# titanic_dataset

This is one of the projects I work on as I follow Aurelien Geron's Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow, Copyright 2019 Aurélien Géron, 978-1-492-03264-9. You can get yourself a copy of the book [here](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291).

This project looks at the a supervised learning task, classification. On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
In this challenge, we had complete the analysis of what sorts of people were likely to survive. In particular, we applied the tools of machine learning to predict which passengers survived the tragedy. 

## The Dataset

The titanic.csv file contains data for 887 of the real Titanic passengers. Each row represents one person. The columns describe different attributes about the person including whether they survived (S), their age (A), their passenger-class (C), their sex (G), the fare they paid (X) etc.

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn 
**Aurelien Geron's Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow:** https://github.com/ageron/handson-ml2
**Ken Jee's notebook:** https://www.kaggle.com/code/kenjee/titanic-project-example/notebook

## Project overview

* A machine learning classifiacation project that is trained on Kaggle Titanic - Machine Learning from Disaster Dataset (https://www.kaggle.com/competitions/titanic/overview).
* We are doing supervised learning here and our aim is to do predictive analysis
* We had to deal with the missing values
* During our journey we'll understand the important tools needed to develop a powerful ML model
* We'll also perform some feature engineering to obtain new features that will make out model more powerful
* Our aim is to play with tools like cross validation, GridSearchCV, RandomizedSearchCV, Logistic Regressions, Decison Trees, Random Forests, Support Vector Machines, K-Nearest Neighbors,Naive Bayes, but also combining models using Voting Classifier and Pipelines to reach our goal
* We'll evaluate the performance of each of our model using RMSE and also tune hyper parameters to further optimize our model
* We'll validate our predictions against our test dataset and conclude our learnings


 ## Training Process
  * Naive Bayes 
  * Decison Trees
  * Logistic Regression
  * Random Forests
  * Support Vector Machines
  * K-Nearest Neighbors

   
 ## Performance Boost
   * Cross Validation  
   * Grid Search 
   * Randomize Search
   * Voting Classifier

## Results 
After trying many models, tuning them and combining, we were able to achive a **78,947% (top 9%)** on the **Titanic dataset competiton** on Kaggle with **the SVC,Random Forest and KNN combined model**.

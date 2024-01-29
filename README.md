# NUS-Datathon-2023
![Topic](https://img.shields.io/badge/Topic-MachineLearning-blue)
![Language](https://img.shields.io/badge/Language-Python-green)

Team:  
Team Zero
<br>

Notebook:  
[Link to Google Colab](https://colab.research.google.com/drive/1ydC7IRMoiWxvoopWF1zFcO3YInFqifj-?usp=sharing)


# Overview  
Over a span of 2 days, we built machine learning models to predict the outcomes of the target `f_purchase_lh` using Python. There were a total of 304 columns in the parquet file provided by Singlife, which contained 3 different dtypes: float64(44), int64(46) and object(214) (a significant number of columns with object dtype actually contained numerical values). Our team conducted in-depth EDA (in `NUS Datathon 2024_second best iteration` notebook), then used a 12-step cleaning process to reduce the number of features for model training. However, we decided to adopt the SelectFromModel method with a less rigorous data cleaning process as it gave a better F1-score. 

**Evaluation Metrics:**  
1. Precision  
2. Recall  
3. F1 Score

These metrics are particularly useful in scenarios where classes are imbalanced, which is the case for our dataset. 


# Our Approach
1. Data Cleaning 
2. Feature Engineering 
3. Imputation Techniques (SimpleImputer, IterativeImputer)
4. RandomUnderSampler
5. SMOTE 
6. XGClassifier + SelectFromModel feature selection method
7. Optuna
8. Other Models (Balanced RF, logistic regression, KNN, SVM)

# Data Cleaning  

**Function 1: ```clean_data(data, target)```**  
1. Null Value Analysis: It calculates and displays the count and percentage of null values per column
2. Column Removal: Columns with 100% null values are removed
3. `hh_20`, `pop_20`, `hh_size_est` are also removed because we observed that `hh_size` =  `hh_20`/ `pop_20`, and  `hh_size` is more meaningful than `hh_20` and `pop_20`, and is more granular than `hh_size_est`

**Function 2: ```clean_data_v2(data)```**  
1. `None` Entries Handling: Counts and percentages of `None` entries per column are calculated and sorted. Rows where `min_occ_date` or `cltdob_fix` are `None` are removed 
2. Data Type Optimization: Converts all float64 columns to float32 for efficiency
3. Column Dropping: The `clntnum` column, a unique identifier, is dropped as it does not contribute to the analysis

**Function 3: ```clean_target_column(data, target_column_name) ```**  
This function is dedicated to preprocessing the target column of the dataset. 


# Data Pre-Processing / Feature Engineering

### client_age
We believe that the age of clients influences their purchasing decisions, hence we added a new column to contain values calculated by subtracting ```min_occ_date``` by ```cltdob_fix```
  
Total percentage of null values in the DataFrame: 22.6%  
Our dataset contained many values, with 32 columns with > 90% null values and 83 columns with > 50% null values.

### Median Data Imputation  
Our data contained many features with right-skewed distributions, which is why we used median imputation as it is robust in the presence of outliers and skewed data distributions. Unlike the mean, which can be heavily influenced by extreme values, the median provides a more representative value of the central tendency and helps in preserving the original distribution of the dataset. 

### Under-Over Sampling Technique  
We implemented a combined under-over sampling strategy to address the class imbalance in our dataset. 

**Under-Sampling**  
We first applied Random Under-Sampling to reduce the size of the overrepresented class. This approach helps in balancing the class distribution and reducing the training dataset size, which can be beneficial for computational efficiency.

**Over-Sampling with SMOTENC**  
After under-sampling, we used SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous data) for over-sampling the minority class. Unlike basic over-sampling techniques, SMOTENC generates synthetic samples for the minority class in a more sophisticated manner, considering both nominal and continuous features. This leads to a more balanced and representative training dataset.

**Combining Both Techniques**  
By combining under-sampling and over-sampling, we aimed to create a more balanced dataset without losing significant information. This combination helps in improving the model's performance, especially its ability to predict minority class instances, leading to more reliable and generalized outcomes.  
<br>
By carefully addressing the class imbalance using this combined approach, we enhanced the model's ability to learn from a more representative dataset, thereby improving its predictive performance on unseen data.

# Feature Selection / ML Model 

**Feature Selection**  
**Understanding Key Influencers in High-Dimensional Data**  
One of our primary challenges was to decipher the most influential factors from a dataset that originally contained over 300 columns (200+ after data cleaning).

### Integrating XGBClassifier with SelectFromModel

**Utilizing a Strong Classifier:**
We employed the XGBClassifier, renowned for its effectiveness in classification tasks and its capability to rank feature importance. 

**SelectFromModel Methodology:**
The SelectFromModel method was applied in tandem with the XGBClassifier. This method analyzes the feature importance scores generated by the classifier and retains only the most significant features. We chose to keep the top 40 features. This threshold was thoughtfully selected to ensure that we retain enough features to capture the diverse aspects of customer behavior while avoiding the pitfalls of model over-complexity and potential overfitting.

**Why SelectFromModel Over RFE or PCA?**

**Computational Efficiency:**  
Recursive Feature Elimination (RFE) is inherently iterative and computationally demanding, especially with a large number of features. In contrast, SelectFromModel offers a more computationally efficient alternative.

**Preserving Interpretability with PCA Limitations:**  
While PCA is effective for reducing dimensionality, it transforms the original features into principal components, which can be challenging to interpret, especially in a business context where understanding specific feature influences is crucial. SelectFromModel maintains the original features, making the results more interpretable and actionable.

**Outcome and Impact:**  
By implementing this feature selection strategy, we were able to significantly reduce the feature space from over 200 to 40, focusing on the most relevant variables that influence customer behavior. 


# XGBClassifier with Optuna 
Optuna is a hyperparameter optimization framework that employs a Bayesian optimization technique to search the hyperparameter space. Unlike traditional grid search, which exhaustively tries all possible combinations, or random search, which randomly selects combinations within the search space, Optuna intelligently navigates the search space by considering the results of past trials.

For our XGBoost classifier, a gradient boosting framework renowned for its performance and speed, the following key hyperparameters were considered:  

- n_estimators: The number of trees in the ensemble.
- max_depth: The maximum depth of the trees.
- learning_rate: The step size shrinkage used to prevent overfitting.
- subsample: The fraction of samples used to fit each tree.
- colsample_bytree: The fraction of features used when constructing each tree.

The optimization process resulted in a set of hyperparameters that achieved an improved F1 score compared to the baseline model. We observed a 10% improvement in the F1 score, indicating an improvement in the harmonic balance of precision and recall for the model.

# Other models (Balanced RF, KNN, SVM, neural networks)
We have built other models but the results are not ideal in terms of F1-score, which is our focus. For example, 

# Next Steps given more time in future implementations  
Since we only had 2 days to work on this datathon, there are some approaches we would like to take if given more time. 
- Experiment with more variations in data preprocessing / feature engineering 
- Use more advanced Optuna features like pruners and samplers for further refinement
- Ensembling methods


### Model 2 and 3. KNN and SVM
We've run the SVM (Support Vector Machine) and KNN (K-Nearest Neighbors) models on our dataset and here are the results interpreted in layman's terms:

- ROC AUC Score (0.671): This score represents the model's ability to distinguish between the classes. A score of 1 means perfect distinction, while a score of 0.5 means the model is no better than random guessing. Our score is 0.671, which is better than a random guess but shows there is significant room for improvement.

- Log Loss (0.174): This number tells us about the uncertainty of the model's predictions. Lower values are better, with 0 representing absolute certainty. Our model's log loss is 0.174, indicating that our predictions are fairly certain, but there could be some misclassifications.

- Precision (0.636): Of all the instances where the model predicted the positive class, 63.6% were actually correct. This is a measure of accuracy for the positive predictions.

- Recall (0.082): Out of all the actual positive instances, the model only correctly identified 8.2%. This means we're missing a lot of true positive instances.

- F1 Score (0.148): This score is a balance between Precision and Recall. It's particularly useful when the class distribution is uneven. The low F1 score suggests the model is not very effective; it's neither precisely identifying true positives nor is it catching a high number of the actual positives.

#### What This Means:
- The model does an average job of ranking predictions but tends to be overly confident in its wrong predictions, indicated by a decent ROC AUC but a high Log Loss.
- It's quite precise—if it says something is likely, it's worth checking out. However, it misses a lot of actual positive cases (low Recall), and thus, the F1 Score is low.
- Essentially, the model is a conservative predictor, only flagging something as positive if it's really sure, but it is missing a lot of true positive cases in the process.




### Model 4. Balanced Random Forest  

![image](https://github.com/reina-peh/NUS-Datathon-2024-Team-Zero/assets/63966022/c319a205-f942-4191-a480-1caa4830fb31)  
The Balanced Random Forest has helped in terms of Recall, possibly because it has been designed to better handle imbalanced classes by adjusting the training algorithm to focus more on the minority class. However, the trade-off here is Precision, leading to many false positives.

ROC AUC Score (0.8320): This is quite a good score. It means that the model has a high chance of correctly distinguishing between the positive and negative classes. In other words, it can identify which cases are likely to be true positives versus true negatives.  

Log Loss (0.5418): Log Loss is a measure of uncertainty where lower values are better. Your model has a moderate log loss, indicating some uncertainty in the predictions it's making.  

Precision (0.1124): Precision is low, which tells us that when the model predicts a case as positive, it is correct only about 11.24% of the time. This suggests that there are quite a few false positives – instances where the model predicted the outcome would occur, but it didn't.  

Recall (0.7682): This metric is quite high, indicating the model is very good at finding the true positive cases. In practical terms, it's catching most of the instances it should.  

F1 Score (0.1961): Despite the high recall, the F1 score is still low because it takes into account both precision and recall. The low precision drags this score down, indicating that the model is not very balanced in its predictive performance.  






![image](https://github.com/reina-peh/NUS-Datathon-2024-Team-Zero/assets/63966022/b7876757-fe1a-44df-a1aa-d33bad565e1a)  
*XGBClassifier metrics*
ROC AUC Score: Improved from 0.8320 with Balanced Random Forest to 0.8693 with XGBoost, indicating that the XGBoost model has a better overall ability to distinguish between the classes.

Log Loss: Decreased from 0.5418 to 0.1263, showing that the XGBoost model has greater confidence in its predictions and a lower rate of uncertainty.

Precision: Significantly increased from 0.1124 to 0.7619. This means that the XGBoost model is much more accurate when it predicts a positive class; there are fewer false positives.

Recall: Decreased from 0.7682 to 0.1059. This suggests that while the XGBoost model is more precise, it's now missing a large number of actual positive cases. It has become more conservative, only predicting positives when it's very sure, leading to many false negatives.

F1 Score: Decreased slightly from 0.1961 to 0.1860, which, despite the high precision, indicates a worse balance between precision and recall due to the drop in recall.

In summary, the XGBoost model has improved in terms of being able to accurately predict the majority class, as indicated by the increased precision and reduced log loss. However, it has become less effective at identifying the minority class, which is usually the more important class to predict in imbalanced datasets, as shown by the lower recall and slightly lower F1 score.

This comparison highlights a common trade-off in machine learning models between precision and recall. Improving one often comes at the expense of the other, especially in imbalanced datasets. The challenge is to find a balance that maximizes both, or to prioritize one over the other based on the specific needs of your application. If predicting the minority class is crucial, you might need to adjust the XGBoost model further or consider techniques like resampling, tailored loss functions, or ensemble methods that can maintain a higher recall without losing the gains in precision


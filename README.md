# NUS-Datathon-2023
![Topic](https://img.shields.io/badge/Topic-MachineLearning-blue)
![Language](https://img.shields.io/badge/Language-Python-green)

Team:  
Team Zero
<br>

Notebook:  
[Link to Google Colab](https://colab.research.google.com/drive/1ydC7IRMoiWxvoopWF1zFcO3YInFqifj-?usp=sharing)

Exploratory Data Analysis:  
`NUS Datathon 2024_second best iteration` notebook in past_iterations folder

# Overview  
Over a span of 2 days, we built machine learning models to predict the outcomes of the target `f_purchase_lh` using Python. There were a total of 304 columns in the parquet file provided by Singlife, which contained 3 different dtypes: float64(44), int64(46) and object(214). Our team conducted in-depth EDA (refer to `NUS Datathon 2024_second best iteration` notebook in past iterations folder), then used a 12-step cleaning process + SelectFromModel to reduce the no. of features for model training. However, we decided to adopt a less rigorous data cleaning process + SelectFromModel method to reduce the no. of features as it produced a better F1-score (our evaluation metric priority). 

**Evaluation Metrics:**  
1. Precision  
2. Recall  
3. F1 Score

These metrics are particularly useful in scenarios where classes are imbalanced, which is the case for our dataset (because of huge differences between Precision and Recall values in past iterations). 


# Our Approach
1. Data Cleaning 
2. Feature Engineering 
3. Imputation Techniques (SimpleImputer, IterativeImputer)
4. RandomUnderSampler
5. SMOTENC  
6. XGClassifier
7. SelectFromModel feature selection method
8. Optuna for hyperparameter optimization  
9. Other Models (Balanced RF, logistic regression, KNN, SVM)

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
  
### Median Data Imputation  
Total percentage of null values in the DataFrame: 22.6%  
32 columns with > 90% null values  
83 columns with > 50% null values  

Since our data contained many features with right-skewed distributions, we used median imputation as it is robust in the presence of outliers and skewed data distributions. Unlike the mean, which can be heavily influenced by extreme values, the median provides a more representative value of the central tendency and helps in preserving the original distribution of the dataset. 

### Under-Over Sampling Technique  
We implemented a combined under-over sampling strategy to address the class imbalance in our dataset. 

**Under-Sampling**  
We first applied Random Under-Sampling to reduce the size of the overrepresented class. This approach helps in balancing the class distribution and reducing the training dataset size, which can be beneficial for computational efficiency.

**Over-Sampling with SMOTENC**  
After under-sampling, we used SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous data) for over-sampling the minority class. Unlike basic over-sampling techniques, SMOTENC generates synthetic samples for the minority class in a more sophisticated manner, considering both nominal and continuous features.

**Combining Both Techniques**  
By combining under-sampling and over-sampling, we aimed to create a more balanced dataset without losing significant information. This combination helps in improving the model's performance, especially its ability to predict minority class instances.  
<br>
By carefully addressing the class imbalance using this combined approach, we have enhanced our model's ability to learn from a more representative dataset, thereby improving its predictive performance on unseen data.

# Feature Selection / ML Model  
One of our primary challenges was to decipher the most influential factors from a high-dimensional dataset that originally contained over 300 columns (200+ after data cleaning).

#### Integrating XGBClassifier with SelectFromModel

**Utilizing a Strong Classifier:**  
We employed the XGBClassifier, renowned for its effectiveness in classification tasks and its capability to rank feature importance. 

**SelectFromModel Methodology:**  
The SelectFromModel method was applied in tandem with the XGBClassifier. This method analyzes the feature importance scores generated by the classifier and retains only the most significant features. We chose to keep the top 40 features, so that we retain enough features to capture the diverse aspects of customer behavior while avoiding the pitfalls of model over-complexity and potential overfitting.

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

The optimization process resulted in a set of hyperparameters that achieved a 10% improvement in the F1 score from the baseline model, indicating a more harmonic balance of precision and recall for the model.  

# Other models  
We have built other models (Balanced RF, KNN, SVM, neural networks) but the results were not ideal in terms of F1-score, which is our focus. For instance, when we ran a Balanced Random Forest model, Recall improved significantly but Precision worsened significantly too (meaning there are more false positives)  
```
ROC AUC Score: 0.8324309322229221
Log Loss: 0.5365502384839375
Precision: 0.12374581939799331
Recall: 0.7350993377483444
F1 Score: 0.21183206106870228
```

# Comparision with Best and Second-Best Iterations  
Even though the Precision (and ROC AUC & logloss) were better in the second-best iteration, we submitted the iteration with the better F1-score as it indicates a more balanced performance in predicting both the majority and minority classes (which is what we want to achieve). 

Second-best iteration  
```
Average ROC AUC Score: 0.8495514162418356
Average Log Loss: 0.1246570067320931
Average Precision: 0.7382839768926726
Average Recall: 0.13814695972129856
Average F1 Score: 0.23216271250192205
```
Best iteration (Submitted)  
```
Precision: 0.43037974683544306
Recall: 0.384180790960452
F1 Score: 0.4059701492537313
```

# Next Steps  
Since we only had 2 days to work on this datathon, there are some approaches we would like to take if given more time. 
- Experiment with more variations in data preprocessing / feature engineering 
- Use more advanced Optuna features like pruners and samplers for further refinement
- Try other oversampling techniques like ADASYN, which adaptively generates minority samples according to their distributions
- Ensembling methods

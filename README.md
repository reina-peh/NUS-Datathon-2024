# NUS-Datathon-2023
Team Name : **TeamZero**
<br>

Notebook: [On Google Colab](https://colab.research.google.com/drive/1ydC7IRMoiWxvoopWF1zFcO3YInFqifj-?usp=sharing)

## Problem Statement
Singlife has observed a concerning trend in the customer journey: potential policyholders are expressing hesitation and eventual disengagement during the insurance acquisition process. To address this, Singlife seeks to leverage its dataset. The objective is to derive actionable insights from this data to enhance the customer experience. The challenge is to dissect the dataset to uncover the critical touchpoints that contribute to customer drop-off and identify opportunities to streamline the application process and personalise communication. The ultimate goal is to predict customer satisfaction and conversion rates, thereby bolstering Singlife's market position.

### Datadriven Approach
In our project, we extensively tested with various data cleaning methods and models to dive deep in the analysis of Singlife's dataset to enhance the customer experience and improve conversion rates. As such the notebook below along with the explanation depicts our efforts in creating the best dataset to be used to train our model, and making sure not to over or underfit the training data such that we can create the best outcome for the test results. 

**Data Analysis:** We delve into the dataset to explore and understand its key features, gaining insights into the underlying patterns and distributions.

**Handling Imbalanced Data:** Strategies are implemented to effectively manage data imbalance, ensuring our model's robustness and accuracy.

**Feature Selection:** Through careful analysis, we identify and select the most impactful features that contribute significantly to our model's performance.

**Model Training & Tuning:** We develop a predictive model, followed by meticulous tuning to enhance its predictive capabilities and ensure optimal performance.

### Methodological Justifications and Detailed Explanations
In this project, we have employed a variety of data preprocessing and modeling techniques. Each method was selected based on specific characteristics of our dataset and the objectives of our analysis. To ensure clarity and transparency, we have provided detailed justifications for our methodological choices. These include, but are not limited to:
1. Imputation Techniques
2. Feature Selection
3. Model Selection
4. Parameter Tuning


### 1. Data Exploration and Cleaning
### **Data Cleaning**

### Function 1: ```clean_data(data, target)```
This function targets null values and specific columns for removal. The steps include:

1. Null Value Analysis: It calculates and displays the count and percentage of null values per column.
2. Column Removal: Columns with 100% null values are removed, except for a specified target column. Additional columns deemed redundant ('hh_20', 'pop_20', 'hh_size_est') are also dropped.

### *Function* 2: ```clean_data_v2(data)```
The focus here is on handling 'None' entries, data type conversion, and removing a unique identifier column.

1. 'None' Entries Handling: Counts and percentages of 'None' entries per column are calculated and sorted.
Row Removal: Rows where 'min_occ_date' or 'cltdob_fix' are 'None' are removed, indicating the importance of these fields.
2. Data Type Optimization: Converts all float64 columns to float32 for efficiency.
3. Column Dropping: The 'clntnum' column, a unique identifier, is dropped as it does not contribute to the analysis.

### Function 3: ```clean_target_column(data, target_column_name) ```
This function is dedicated to preprocessing the target column of the dataset. The primary focus is to handle missing values and ensure the data type consistency of the target variable, which is crucial for the accuracy and effectiveness of the model.



### 2. Feature Engineering
###Feature Engineering

We have identified that age at which a customer purchases an insurance policy is important, hence we created a new column to calculate their age using ```min_occ_date``` and ```cltdob_fix```

Train Test Split to divide a dataset into training and testing subsets for model training and evaluation**


### 3. Data Processing
### **Data Imputation: Addressing NaN or None Values**

**Challenges with Missing Data:** Our dataset contained NaN or None values, which posed a significant challenge for performing accurate statistical analyses and data visualizations. The choice of imputation technique was critical to preserve data integrity and maintain the reliability of our analyses.

**Why Median Imputation?**

- **Robustness Against Outliers and Skewed Distributions:** Median imputation was chosen for its robustness in the presence of outliers and skewed data distributions. Unlike the mean, which can be heavily influenced by extreme values, the median provides a more representative value of the central tendency in such cases.

- **Maintaining Data Integrity:** The median imputation helps in preserving the original distribution of the dataset. This is crucial for maintaining the structural integrity of the data, ensuring that subsequent analyses are reflective of the true nature of the underlying data.

**Alternatives Considered and Their Limitations:**

- **Mean Imputation:** Simple imputer mean was considered; however, its susceptibility to outliers made it less suitable for our dataset. Mean imputation could potentially introduce bias, especially in skewed distributions, leading to distorted analyses.

- **Simplicity and Efficiency:** Given the size and nature of our dataset, median imputation offered a balance between simplicity, computational efficiency, and effectiveness. This method allowed us to quickly and effectively address missing values, enabling us to proceed with our analyses without introducing significant bias.

### Under-Over Sampling Technique

In our data preprocessing phase, we implemented a combined under-over sampling strategy to address the class imbalance in our dataset. Class imbalance is a common issue in machine learning, where some classes are underrepresented compared to others. This imbalance can lead to biased models that don't perform well on minority classes.

**Why We Chose Under-Over Sampling:**

- **Under-Sampling**: We first applied Random Under-Sampling to reduce the size of the overrepresented class. This approach helps in balancing the class distribution and reducing the training dataset size, which can be beneficial for computational efficiency.

- **Over-Sampling with SMOTENC**: After under-sampling, we used SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous data) for over-sampling the minority class. Unlike basic over-sampling techniques, SMOTENC generates synthetic samples for the minority class in a more sophisticated manner, considering both nominal and continuous features. This leads to a more balanced and representative training dataset.

- **Combining Both Techniques**: By combining under-sampling and over-sampling, we aimed to create a more balanced dataset without losing significant information. This combination helps in improving the model's performance, especially its ability to predict minority class instances, leading to more reliable and generalized outcomes.

By carefully addressing the class imbalance using this combined approach, we enhanced the model's ability to learn from a more representative dataset, thereby improving its predictive performance on unseen data.

### 4. Model Building and Evaluation
### **Feature Selection**

**Understanding Key Influencers in High-Dimensional Data:** In our project, the primary challenge was to decipher the most influential factors from a dataset that originally contained over 200 columns. Such a high-dimensional dataset can obscure crucial insights, especially when analyzing complex customer behaviors. To navigate this, a strategic approach to feature selection was essential.

**Integrating XGBClassifier with SelectFromModel:**

- **Utilizing a Strong Classifier:** We employed the XGBClassifier, renowned for its effectiveness in classification tasks and its capability to rank feature importance. XGBoost, with its gradient boosting framework, excels in handling various types of data and uncovering complex patterns. Its intrinsic feature importance metric provides a reliable basis for feature selection.

- **SelectFromModel Methodology:** The SelectFromModel method was applied in tandem with the XGBClassifier. This method analyzes the feature importance scores generated by the classifier and retains only the most significant features. For our project, we chose to keep the top 40 features. This threshold was thoughtfully selected to ensure that we retain enough features to capture the diverse aspects of customer behavior while avoiding the pitfalls of model overcomplexity and potential overfitting.

**Why SelectFromModel Over RFE or PCA?**

- **Computational Efficiency:** Recursive Feature Elimination (RFE) is inherently iterative and computationally demanding, especially with a large number of features. In contrast, SelectFromModel offers a more computationally efficient alternative.

- **Preserving Interpretability with PCA Limitations:** While PCA is effective for reducing dimensionality, it transforms the original features into principal components, which can be challenging to interpret, especially in a business context where understanding specific feature influences is crucial. SelectFromModel maintains the original features, making the results more interpretable and actionable.

- **Balancing Feature Set and Model Complexity:** The goal was to distill the dataset to a manageable number of features without losing critical information. SelectFromModel, coupled with XGBClassifier, provided a more nuanced approach to achieving this balance compared to the broad dimensionality reduction offered by PCA or the intensive feature elimination process of RFE.

**Outcome and Impact:**

By implementing this feature selection strategy, we were able to significantly reduce the feature space from over 200 to 40, focusing on the most relevant variables that influence customer behavior. This not only enhanced the model's performance by reducing noise and complexity but also aided in interpretability, allowing for more straightforward insights and decision-making based on the model's outputs.

### 5. Evaluation Metrics: Precision, Recall, and F1 Score

**Precision** measures the accuracy of positive predictions. It is the ratio of true positive predictions to the total number of positive predictions (true positives + false positives). A higher precision score indicates that the model is more accurate in predicting positive cases.

**Recall** (Sensitivity) measures the model's ability to correctly identify all positive cases. It is the ratio of true positive predictions to the actual number of positive cases (true positives + false negatives). Higher recall indicates that the model is better at catching all positive cases.

**F1 Score** is the harmonic mean of precision and recall. It is a balance between precision and recall, providing a single metric that summarizes the model's accuracy. An F1 score reaches its best value at 1 (perfect precision and recall) and its worst at 0.

These metrics are particularly useful in scenarios where classes are imbalanced or when the costs of false positives and false negatives are very different.



# Previous Iterations 


### **Model Training**
In the training stage, we opted for the XGBoost classifier, guided by a comparative performance analysis against models like Logistic Regression, SVM, and KNN. The `pretrain_model` function is responsible for initializing the XGBClassifier with chosen hyperparameters and fitting it to the training data. Below are the key theoretical reasons for selecting XGBoost:

1. **Ensemble Learning Method**: XGBoost leverages an advanced form of gradient boosting, an ensemble technique where multiple weak learners (decision trees) are combined in a sequential manner. Each tree in the sequence corrects the errors of its predecessors, culminating in a robust overall model.

2. **Regularization Techniques**: It includes L1 and L2 regularization, which significantly reduce the risk of overfitting, a crucial advantage over traditional models, particularly in complex datasets.

3. **Versatile Data Handling**: XGBoost efficiently processes a mix of categorical and numerical features, outperforming models like Logistic Regression and KNN that often require extensive preprocessing for different data types.

4. **Optimized Performance and Speed**: Designed for high performance and speed, XGBoost optimizes computational resources and memory usage, making it more scalable and faster than conventional algorithms, especially in handling large datasets.

These factors combined — the ensemble approach, regularization, versatility in data handling, and optimized performance — make XGBoost an excellent choice for our dataset, leading to superior accuracy and efficiency compared to the alternatives.

Previously considered steps:


### **Hyperparameter Optimization with Optuna for XGBoost Model**

We focused on optimising the hyperparameters of an XGBoost classifier using the Optuna library. The goal is to find the best combination of hyperparameters that maximizes the F1 score of the model on the validation dataset. This is achieved through a systematic and efficient search across a specified range of hyperparameter values.

### 1. Data Exploration and Cleaning
*To reduce the number of features to help us with our model building process, we identified a few issues:*
- Null Values: Check for missing or null values in the dataset.
- Data Types: Ensure each column has the correct data type (e.g., dates, categorical data, numerical values).
- Outliers: Identify any outliers in numerical data that might skew analysis.


*And utilised methods targeted at them:* 
1. Dropped the columns that were not relevant to the problem statement.
     - 'clntnum', a unique identifier for each customer, not relevant to the problem statement.
2. Checked for duplicates (none)
3. Found the number and type of unique values in each column
4. Identified columns with an 'object' data type.
    - For columns with names matching patterns like 'n_months_since_lapse_' or 'n_months_last_bought_',convert them to a numeric data type.
    - Check if columns contain values of type Decimal or float and converts them accordingly.
    - Categorial data: certain columns are designated as categorical.
      - ie. ('clntnum', 'race_desc', 'ctrycode_desc', 'clttype', 'stat_flag', 'cltsex_fix', 'annual_income_est')
    - Date formatting: Other non-categorial columns are converted to datetime format 
5. Calculated the null values in each column 
      - drop the columns with 100% null values (except the target column)
  



### 2. Feature Engineering
- Further exploring the dataset, we realised that there are 5 types of policies:
1. general insurance (gi)
2. group policies (grp)
3. investment-linked policies (inv)
4. life or health insurance (lh)
5. long-term care insurance (ltc)

The suffixes (e.g. 42e115, 1280bf) are unique identifiers for the specific insurance products.

- We proceed to find unique identifiers of each policy type
  - Unique identifiers for gi policies: {'29d435', 'claim', '856320', '058815', 'a10d1b', '42e115'}
- Unique identifiers for grp policies: {'e91421', 'de05ae', '9cdedf', 'caa6ff', '70e1dd', '22decf', '94baec', 'fe5fb8', 'e04c3a', '6a5788', 'fd3bfb', '1581d7', '6fc3e6', '945b5a'}
- Unique identifiers for inv policies: {'e9f316', 'dcd836'}
- Unique identifiers for lh policies: {'d0adeb', 'e22a6a', '839f8a', '507c37', 'f852af', '947b15'}
- Unique identifiers for ltc policies: {'43b9d5', '1280bf'}



### 3. Data Processing
Examining the relationship between clients identified as 'at risk' and their history of insurance claims

#### New Claims Flag Creation:
A new binary flag called flg_has_claims is created in the demo_data DataFrame. This flag is set to 1 for clients who have made a claim in either health, life, or general insurance (as indicated by the respective flags flg_has_health_claim, flg_has_life_claim, flg_gi_claim). This unifies different types of claims into a single column for simplified analysis.

#### Subset DataFrames:
Two separate subsets of the demo_data DataFrame are created. One (at_risk_df) includes clients who are marked as 'at risk' (flg_at_risk equals 1), and the other (not_at_risk_df) includes those not marked as 'at risk'. Both DataFrames include only the new flg_has_claims column.

#### Calculating Percentages:
The code calculates the percentage of 'at risk' clients who have made claims and the percentage of 'not at risk' clients who have made claims. The result is 25.38% for 'at risk' and 6.58% for 'not at risk'.

#### Analysis Interpretation:
This indicates that a significant minority (about a quarter) of the 'at risk' clients have made claims on their policies. In contrast, a much smaller proportion of clients not identified as 'at risk' have made claims. This suggests that the 'at risk' flag may be a good indicator of higher claim activity and could be an area to focus on to prevent policy disengagement.

### Suggestions to Singhealth based on the analysis:
Based on this analysis, Singlife can take the following actions to address the customer journey issues:

1. Enhanced Communication: Tailor communication strategies for 'at risk' clients, especially around the claims process, to improve their experience and possibly prevent disengagement.
2. Risk Management: Review the criteria for labeling clients as 'at risk' and assess whether the current model effectively identifies clients who may need additional support.
3. Customer Support: Implement proactive support for 'at risk' clients who have made claims to guide them through the process and encourage continued engagement.
By addressing these points, Singlife can aim to improve customer satisfaction, reduce drop-off during the acquisition process, and potentially increase conversion rates.

### 4. Model Building and Evaluation

Processing the train_test_split:
- Imported the train_test_split function from the sklearn.model_selection module, which is a part of the scikit-learn library, a popular machine learning library in Python.

Splitting the Dataset:
- Called the train_test_split function with the following parameters:
  - data_cleaned
  - test_size=0.2: 20% of the datareserved for the validation set (val). The remaining 80% of the data will be used as the training set (tr).
  - random_state=42: A seed for the random number generator to ensure reproducibility 

Training and Validation Sets:
- The function returns two subsets of the data:
    - tr: The training set, which contains 80% of the data. This subset is used to train the machine learning model.
    - val: The validation set, which contains 20% of the data. This subset is used to evaluate the model's performance and to fine-tune the model's hyperparameters.


### 5. Model Deployment: (Best Model is the XGBClassifier)
Utiised 4 types of models; XGBClassifier (with and without optuna),  KNN, SVM, Logistic Regression and Random Forrest 


### 6. Success metrics used ROC AUC score, logloss, precision, recall & f1-score

### Model 1: XGBClassifier

- The XGBClassifier is based on the gradient boosting algorithm, which is an ensemble method that combines multiple weak learners to produce a more powerful model.
- Is a part of the XGBoost library, which is a popular open-source library for gradient boosting.
- use_label_encoder=False: Disables the automatic label encoding inside XGBoost for version 1.3.0 and above.
- eval_metric='logloss': Sets 'logloss' as the evaluation metric for binary classification.
- device='gpu': Utilizes GPU for model training, enhancing speed.
- enable_categorical=True: Allows direct processing of categorical features in the dataset.
- objective='binary:logistic': Specifies the learning objective for binary classification outputting probabilities.
- n_estimators=100: Number of boosting rounds or trees to build.
- learning_rate=0.05: The factor by which to shrink the feature weights after each boosting round.
- reg_alpha=0.1: Applies L1 regularization on weights, aiding in feature selection.
- reg_lambda=1.0: Applies L2 regularization on weights to combat overfitting.

#### XGBClassifier with Optuna 
Optuna is a cutting-edge hyperparameter optimization framework that employs a Bayesian optimization technique to search the hyperparameter space. Unlike traditional grid search, which exhaustively tries all possible combinations, or random search, which randomly selects combinations within the search space, Optuna intelligently navigates the search space by considering the results of past trials.

For our XGBoost classifier, a gradient boosting framework renowned for its performance and speed, the following key hyperparameters were considered:

- n_estimators: The number of trees in the ensemble.
- max_depth: The maximum depth of the trees.
- learning_rate: The step size shrinkage used to prevent overfitting.
- subsample: The fraction of samples used to fit each tree.
- colsample_bytree: The fraction of features used when constructing each tree.

##### Execution
Optuna creates a "study" which is a series of trials to evaluate the performance of the XGBoost model with different hyperparameters. For each trial, Optuna suggests a new set of hyperparameters, the model is trained on the training dataset, and then evaluated on the validation dataset. The F1 score is used as the evaluation metric because it provides a balance between precision and recall, which is particularly useful when dealing with imbalanced datasets.

##### Results
The optimization process resulted in a set of hyperparameters that achieved an improved F1 score compared to the baseline model. We observed an increase in the F1 score, indicating a significant improvement in the harmonic balance of precision and recall for the model.

##### Conclusion
- Before and after using Optuna, we observe a 10% improvement in ROC accuracy, from F1 score of 0.31 to 0.35
The application of Optuna for hyperparameter tuning of the XGBoost model has proven to be effective. It not only enhanced the model's predictive accuracy but also streamlined the hyperparameter tuning process. This approach can be replicated for other machine learning models to improve their performance on various tasks.

#### Next Steps given more time in future implementations
- Implement the optimized model in a production environment to assess the improvements in a real-world setting.
- Explore the incorporation of additional hyperparameters and the use of more advanced Optuna features like pruners and samplers for further refinement.
- Conduct additional optimization rounds as more data becomes available or as the characteristics of the data evolve over time.





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




### Model 4. Random Forrests and its comparison with our current best XGBClassifier 

![image](https://github.com/reina-peh/NUS-Datathon-2024-Team-Zero/assets/63966022/c319a205-f942-4191-a480-1caa4830fb31)
*balanced random forest model metrics*

ROC AUC Score (0.8320): This is quite a good score. It means that the model has a high chance of correctly distinguishing between the positive and negative classes. In other words, it can identify which cases are likely to be true positives versus true negatives.

Log Loss (0.5418): Log Loss is a measure of uncertainty where lower values are better. Your model has a moderate log loss, indicating some uncertainty in the predictions it's making.

Precision (0.1124): Precision is low, which tells us that when the model predicts a case as positive, it is correct only about 11.24% of the time. This suggests that there are quite a few false positives – instances where the model predicted the outcome would occur, but it didn't.

Recall (0.7682): This metric is quite high, indicating the model is very good at finding the true positive cases. In practical terms, it's catching most of the instances it should.

F1 Score (0.1961): Despite the high recall, the F1 score is still low because it takes into account both precision and recall. The low precision drags this score down, indicating that the model is not very balanced in its predictive performance.

The Balanced Random Forest has helped in terms of Recall, possibly because it has been designed to better handle imbalanced classes by adjusting the training algorithm to focus more on the minority class. However, the trade-off here is Precision, leading to many false positives.




![image](https://github.com/reina-peh/NUS-Datathon-2024-Team-Zero/assets/63966022/b7876757-fe1a-44df-a1aa-d33bad565e1a)
*XGBClassifier metrics*
ROC AUC Score: Improved from 0.8320 with Balanced Random Forest to 0.8693 with XGBoost, indicating that the XGBoost model has a better overall ability to distinguish between the classes.

Log Loss: Decreased from 0.5418 to 0.1263, showing that the XGBoost model has greater confidence in its predictions and a lower rate of uncertainty.

Precision: Significantly increased from 0.1124 to 0.7619. This means that the XGBoost model is much more accurate when it predicts a positive class; there are fewer false positives.

Recall: Decreased from 0.7682 to 0.1059. This suggests that while the XGBoost model is more precise, it's now missing a large number of actual positive cases. It has become more conservative, only predicting positives when it's very sure, leading to many false negatives.

F1 Score: Decreased slightly from 0.1961 to 0.1860, which, despite the high precision, indicates a worse balance between precision and recall due to the drop in recall.

In summary, the XGBoost model has improved in terms of being able to accurately predict the majority class, as indicated by the increased precision and reduced log loss. However, it has become less effective at identifying the minority class, which is usually the more important class to predict in imbalanced datasets, as shown by the lower recall and slightly lower F1 score.

This comparison highlights a common trade-off in machine learning models between precision and recall. Improving one often comes at the expense of the other, especially in imbalanced datasets. The challenge is to find a balance that maximizes both, or to prioritize one over the other based on the specific needs of your application. If predicting the minority class is crucial, you might need to adjust the XGBoost model further or consider techniques like resampling, tailored loss functions, or ensemble methods that can maintain a higher recall without losing the gains in precision


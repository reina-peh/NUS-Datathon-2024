# NUS Datathon 2024
![Topic](https://img.shields.io/badge/Topic-MachineLearning-blue)
![Language](https://img.shields.io/badge/Language-Python-green)

> **Achievement 🏆 :**  
> 2nd Place (out of over 900 participants)

This competition is the National University of Singapore's annual Data Science hackathon where participants build machine learning models to tackle real-life business cases of corporate partners. Our team was judged by both NUS Statistics and Data Science Society and lead data scientists from Singlife.

**Team Number:** 219  

**Team Name:** Team Zero  

**Team Members / Contributors:**
* Reina Peh ([LinkedIn](https://www.linkedin.com/in/reinapeh/))
* Ryan Tan ([LinkedIn](https://www.linkedin.com/in/ryantzr/))  
* Zhang Bowen ([LinkedIn](https://www.linkedin.com/in/bowen-zhang-2b5617249/))
* Claudia Lai ([LinkedIn](https://www.linkedin.com/in/claudialaijy/))  

# Predicting Singlife Clients' Purchasing Behaviors With Python  
Our goal is to predict the outcomes of the target `f_purchase_lh` using the dataset provided by Singlife, which contained 304 columns and 17,992 rows. Our Exploratory Data Analysis can be found in the `NUS Datathon 2024_EDA_Past_Iteration` file. 3 main model performance evaluation metrics were used as they are useful in scenarios where classes are imbalanced, which is the case for our dataset (because the minority class took up only 3-4% of the target column). 

**Evaluation Metrics:**  
1. Precision  
2. Recall  
3. F1-Score (our priority) 

# Our Approach
1. Data Cleaning 
2. Feature Engineering 
3. Imputation Techniques (SimpleImputer, IterativeImputer)
4. RandomUnderSampler
5. SMOTENC  
6. XGClassifier Model  
7. SelectFromModel Feature Selection Method
8. Optuna
9. LIME (Explainable AI)

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


# Data Pre-Processing  

### client_age
We believe that the age of clients influences their purchasing decisions, hence we added a new column to contain values calculated by subtracting ```min_occ_date``` by ```cltdob_fix```
  
### Median Data Imputation  
Total percentage of null values in the DataFrame: 22.6%  
32 columns with > 90% null values  
83 columns with > 50% null values  

Skewed Distributions  
<img src="https://github.com/reina-peh/NUS-Datathon-2024/assets/75836749/ba88e5ba-5dbb-4059-950c-f58180cd24cc" width="500">  
Since our data contained some features with right-skewed distributions, and many features with binary values (0s and 1s), we adopted median data imputation to fill the null values. This is because median imputation provides more representative values for features with only 0 and 1 values, and is also robust in the presence of skewed data distributions and outliers.

### Under-Over Sampling Technique  
We implemented a combined under-over sampling strategy to create a more balanced dataset to improve our model's ability to predict the minority class instances without losing significant information.  

**Under-Sampling**  
We first applied Random Under-Sampling to reduce the size of the overrepresented class. This approach helps in balancing the class distribution and reducing the training dataset size, which can be beneficial for computational efficiency.

**Over-Sampling with SMOTENC**  
After under-sampling, we used SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous data) for over-sampling the minority class. Unlike basic over-sampling techniques, SMOTENC generates synthetic samples for the minority class in a more sophisticated manner, considering both nominal and continuous features.  

# ML Model & Feature Selection  
One of our primary challenges was to decipher the most influential factors from a high-dimensional dataset that originally contained over 300 columns (200+ after data cleaning).

### Integrating XGBClassifier with SelectFromModel

**Utilizing a Strong Classifier:**  
We employed the XGBClassifier, a gradient boosting framework renowned for its effectiveness in classification tasks and its capability to rank feature importance.  

**SelectFromModel Methodology:**  
The SelectFromModel method was applied in tandem with the XGBClassifier. This method analyzes the feature importance scores generated by the classifier and retains only the most significant features. We chose to keep the top 40 features, so that we retain enough features to capture the diverse aspects of customer behavior while avoiding the pitfalls of model over-complexity and potential overfitting.

### Why SelectFromModel Over RFE or PCA?

**Computational Efficiency:**  
Recursive Feature Elimination (RFE) is inherently iterative and computationally demanding, especially with a large number of features. In contrast, SelectFromModel offers a more computationally efficient alternative.

**Preserving Interpretability:**  
While PCA is effective for reducing dimensionality, it transforms the original features into principal components, which can be challenging to interpret, especially in a business context where understanding specific feature influences is crucial. SelectFromModel maintains the original features, making the results more interpretable and actionable.  

# XGBClassifier with Optuna  
Optuna is a hyperparameter optimization framework that employs a Bayesian optimization technique to search the hyperparameter space. Unlike traditional grid search, which exhaustively tries all possible combinations, or random search, which randomly selects combinations within the search space, Optuna intelligently navigates the search space by considering the results of past trials.  

For our XGBoost classifier, the following key hyperparameters were considered:  

- `n_estimators`: The number of trees in the ensemble.
- `max_depth`: The maximum depth of the trees.
- `learning_rate`: The step size shrinkage used to prevent overfitting.
- `subsample`: The fraction of samples used to fit each tree.
- `colsample_bytree`: The fraction of features used when constructing each tree.  

### Our Results  
The optimization process resulted in a set of hyperparameters that achieved an approximately 10% improvement in the F1-score from the baseline model, indicating a more harmonic balance of precision and recall for the model.  

<img src="https://github.com/reina-peh/NUS-Datathon-2024/assets/75836749/6eed42b1-40da-43ed-a73e-2147b91192b9" width="500">  

# Local Interpretable Model-Agnostic Explanations (LIME) 
Local Interpretable Model-Agnostic Explanations (LIME) elucidates the decision-making of complex machine learning models by approximating them with simpler, interpretable models around specific instances. This technique demystifies "black box" models, making their predictions transparent and aiding in better decision-making.  
<img src="https://github.com/reina-peh/NUS-Datathon-2024/assets/75836749/6a20fe01-15ff-4b14-925c-3c1d6f4aa8af" width="500">  
Reference:  
Papers with Code - LIME Explained. (2016). https://paperswithcode.com/method/lime  
<br>  
### Applying LIME  
<img src="https://github.com/reina-peh/NUS-Datathon-2024/assets/75836749/c8519048-2d9a-49b3-a1ee-8aca78a0d976" width="500">  
<img src="https://github.com/reina-peh/NUS-Datathon-2024/assets/75836749/d0afbca7-0758-49af-9236-86f0f113e458" width="500">  

In our study, we applied LIME 100 times across different instances, averaging the explanation weights to counteract the randomness in data perturbation and ensure more consistent insights. This method helps pinpoint the features most impactful to the model's predictions, providing a stable and clear understanding of its behavior. The averaging process across multiple LIME iterations reduces variability, allowing us to identify and rely on the consistent influence of specific features on the model's decisions.  

# Next Steps  
Since we only had 2-3 days to work on this datathon, here are some approaches we would like to take if given more time:   
* Use more advanced Optuna features like pruners and samplers for further refinement
* Use other oversampling techniques like ADASYN, which adaptively generates minority samples according to their distributions

# Machine Learning Project

Customer Churn Prediction - Car Insurance Industry

# Introduction – Problem Statement

It is no secret that churn rate is a health indicator for any subscription-based company. The ability to identify customers that are not happy with provided solutions allows businesses to learn about product or pricing plan weak points, operation issues, as well as customer preferences and expectations, in order to reduce reasons for churn. For that reason, churn prediction and customer retention is a top priority for many companies, as acquiring new customers can be several times more expensive than retaining existing ones. Meanwhile, customer churn poses a significant challenge in the insurance industry, reports show a frequent and increasing trend of customers switching companies in order to take advantage of a competitors&#39; offer. For that reason, it is important to implement a churn model, which will ultimately lead to the design of a data-driven retention strategy, by providing insights into the reasons behind customers&#39; churn and risk estimations associated with individual customers.

With that been said, this project focuses on the design of a prediction model for car insurance churn (contract-oriented), which identifies the customers/contracts with a high likelihood of leaving the company.

# Data Acquisition

The main dataset used for the training and test of the churn model is extracted from the Enterprise Data Warehouse (DWH) environment of an insurance company. Within the DWH environment, data from various sources such as CRM or billing systems are integrated into comprehensive data structures, following the main principles of star-schema modelling architecture. Data are refreshed daily via an ETL procedure and results are stored in an Oracle Database.

In this context, the main source will be a consolidated table, containing historical information of the contracts, contract damages and receipt lines, which allow for the computation of many aggregated fields such as total years insured, total accidents for a specific period, coverages per contract number etc.

This consolidated view is later joined with other tables of the data warehouse in order to combine and bring together other set of attributes that are considered to influence churn prediction in car insurance branch, according to bibliography and insurance business logic.

In the table below you may find a comprehensive view regarding the attributes that are being evaluated:

|   | Description | Type |
| --- | --- | --- |
| **CURRENT\_POLICY\_STATUS** | This is the target variable. Policy status for the specified month (ex. Canceled, Active) | Categorical (Binary) |
| MO\_KEY | Month of renewal | Categorical |
| ACCEPT\_EMAIL | Identifies whether or not the customer has agreed to communication via email. | Categorical (Binary) |
| INS\_PKG\_KEY | Code specifying the insurance package coverage (example: BASIC, BASIC PLUS etc.) | Categorical |
| AGENT\_CTGR\_KEY | Code specifying the insuree&#39;s Agent Category (example: Insurance Consultant, Insurance Broker etc.) | Categorical |
| AGENT\_KEY | Code specifying the insuree&#39;s Agent. | Categorical |
| BONUS\_MALUS | If there are any discount according to bonus/malus level. The term bonus-malus (Latin for good-bad) is used for a number of business arrangements which alternately reward (bonus) or penalize (malus). | Categorical |
| CAR\_AGE | The Insured Vehicle&#39;s/S&#39; Lifetime | Numerical |
| CAR\_BRAND\_KEY | Brand Of The Insured Vehicle/S | Categorical |
| CAR\_CAPACITY | Cubic Capacity Of The Insured Vehicle/S | Numerical |
| CAR\_INVC\_ZONE\_KEY | Car Invoice Zone (for example big city centers are usually costly). | Categorical |
| CAR\_MODEL\_KEY | Model Of The Insured Vehicle/S | Type |
| CAR\_PROD\_YEAR | Year Of Construction Of The Insured Vehicle/S | Categorical |
| CUST\_AGE | Age Of The Insured Person | Categorical |
| IS\_STANDING\_ORDER | Method of paying the premium | Categorical |
| LOYALTY\_PROGRAM | Loyalty program that customer belongs to (0 No, 1Yes) | Categorical |
| NUM\_ACCIDENTS\_INVOLVED | Number Of Total Accidents That The Insuree Was Invlolved In | Numerical |
| NUM\_ACCIDENTS\_RESP | The number of total accidents | Numerical |
| NUM\_ACCIDENTS\_RESP\_FR | The number of accidents caused by the insured person (Responsible Fault) | Numerical |
| NUM\_ACCIDENTS\_RESP\_NOT\_FR | The number of accidents caused by the insured person (Not Responsible Fault) | Numerical |
| NUM\_ACTIVE\_SYMB | Number Of Policies That Are Active for this customer | Numerical |
| NUM\_COVERAGES | Sum Of All The Premiums That The Insured Is Paying | Numerical |
| PREMIUM\_PRICE | Premium that The Insured Is Paying | Numerical |
| SYMB\_DURATION | Maximum Duration Of All Policies Owned By The Same Customer | Numerical |
| TAXK\_INC\_ZONE\_KEY | Code specifying the insuree&#39;s income level (i.e. Low Income, Mediume Income etc.) | Ordinal |
| INS\_PARTY\_TENURE\_RANGE\_KEY | The customer&#39;s tenure as calculated by the organization | Numerical |
| INS\_SYMB\_TENURE\_RANGE\_KEY | The customer&#39;s tenure as calculated by the organization | Numerical |
| INS\_ALLPARTY\_TENURE\_RANGE\_KEY | The customer&#39;s tenure as calculated by the oragnization | Numerical |
| LIFE\_SYMB\_AFM | Which kind of company policies the customer has. For example if he/she has life insurance (per AFM)? | Categorical |
| FIRE\_SYMB\_AFM | Which kind of company policies the customer has. For example if he/she has fire insurance (per AFM)? | Categorical |
| LIFE\_SYMB\_PARTY | Which kind of company policies the customer has. For example if he/she has life insurance (per party)? | Categorical |
| FIRE\_SYMB\_PARTY | Which kind of company policies the customer has. For example if he/she has fire insurance (per party)? | Categorical |
| CLAIMS\_AMOUNT | The Amount The Company Will Pay To Reimburse The Lost Item That Is Equal To How Much The Item Is Worth Today | Numerical |
| HAS\_RETURNED | If The Customer Has Previously Left The Company And Then Returned | Categorical |
| NEW\_PREMIUM\_PRICE | New Premium Price after Renewal (is always 0 for churn cases --> remove) | Numerical |
| NUM\_COMPLAINTS | Total number of complaints made from the insuree (if any) | Numerical |
| NEW\_MONTHLY\_PREMIUM\_PRICE | New Monthly Premium Price after Renewal (is always 0 for churn cases --> remove) | Numerical |
| LOSS\_RATIO | The ratio of losses to premiums earned | Numerical |
| ASFAL\_AMOUNT | Sum of the amount to be paid for an insurance policy. | Numerical |

All attributes mentioned above, were consolidated in a single view. In this view, each record is associated with a specific contract number for the corresponding month. Note that both churners and existing customers&#39; profiles are taken into account. Also note that since the main source of extraction is the company&#39;s DWH, no complicated pre-processing needs to be performed, as problems such as data inconsistency and null values are taken care of during the ETL process.

However, since the final view contains historical monthly snapshots of contracts, in order to avoid the use of redundant information in the final dataset several manipulation techniques were incorporated in the final extraction logic, in order to provide a comprehensive view of the data.

1. Data extraction logic: Exclude repetitive or misguiding records from the dataset. For example in this case, the company is only interested in contracts that are about to be renewed. So in the final dataset only contracts that are recognized as possible for renewal in the next period are collected. Meanwhile, other business rules were applied, such as excluding contracts with zero duration since these data do not correspond to reality. The following SQL code gives an overview of the rules that were applied:
```
INSERT INTO ET_ML_CHURN_FI_FNL
SELECT * 
FROM ML_CHURN_FI
WHERE 
MO_KEY>20190501 AND  MO_KEY<20210701 AND --Only take into account last years' data.
TOTAL_YEAR_INSURED_SYMB>=0 AND TOTAL_YEAR_INSURED_SYMB<=10 AND --Only a specific insurance period is valid according to the business logic specified by the company.
POLICY_STATUS=1  AND --Take into account only active snapshots of contracts.
CURRENT_POLICY_STATUS!=-1 AND --Do not take into account null values for our target variable.
CURRENT_POLICY_STATUS=0 AND
IS_RENEWAL=1 AND  -- Only take into account monthly snapshots corresponding to the month of renewal for each contract.
SYMB_DURATION!=0  -- Do not take into account contract duration outliers.
AND INS_PKG_KEY!=-1 -- Do not take into account null attributes.
AND AGENT_CTGR_KEY!=-1-- Do not take into account null attributes.
AND TAXK_INC_ZONE_KEY!=-1-- Do not take into account null attributes.
```



2. Resampling: Given the nature of the use case it makes sense that the majority of our records represent the NOT\_CHURN class (~80% of the dataset), while the CHURN class is only present in the ~20% of the dataset. For that reason, the following methods of resampling were applied:
  A. Random sub-sampling of dominant class (NOT\_CHURN)
  B. Avoid repetitive records. Our dataset contains monthly snapshots that are bound to present multiple views of the same attributes, if the customer never churns or makes any distinctive changes in the contract rules. For that reason, after concluding to the main features that were going to be examined for the ML task, only the distinct values made it to the final dataset.
  C. Over-sampling under-represented class (CHURN): Enrich the dataset with views of records on customers that churned in past months (not included in the original dataset).

The final dataset above was extracted to data frames using Python library cx\_Oracle. The present code makes use of the pre-extracted data frames which were saved in the form of pickle dumps.

Note that the details above corresponds to the details applied for the acquisition of the training dataset. A similar process was applied for the extraction of the final testing dataset (containing similar data post April 2021).

| **Category** | **Train** | **Test** |
| --- | --- | --- |
| **Dataset Name** | ET\_ML\_CHURN\_FI\_FNL | ET\_ML\_CHURN\_FI\_TEST |
| **No of records** | 1.069.994 | 85522 |
| **Churn records** | 750.503 | 74.089 |
| **Not Churn Records** | 486.895 | 11.433 |
| **No of features** | 88 | 88 |

# Feature Selection

Feature Selection is one of the core concepts in machine learning which hugely affects the performance of our model. The data features that are used to train machine learning models have a huge influence on their performance, as irrelevant or partially relevant features can negatively affect model performance.

As a rule of thumb, fewer attributes is desirable because it reduces the complexity of the model and a simpler model is simpler to understand and explain.

**Benefits of performing feature selection:**

- **Reduces Overfitting** : Less redundant data means less opportunity to make decisions based on noise.
- **Improves Accuracy** : Less misleading data means modeling accuracy improves.
- **Reduces Training Time** : fewer data points reduce algorithm complexity and algorithms train faster.

## Selected Methods

1. **Pearson Correlation Matrix**

Correlation states how the features are related to each other or the target variable. Features with high correlation are more linearly dependent and hence have almost the same effect on the dependent variable. Therefore, when two features have high correlation, we can drop one of the two features.

Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable).

The Pearson correlation is also known as the &quot;product moment correlation coefficient&quot; (PMCC) or simply &quot;correlation&quot;. Pearson correlations are suitable only for metric variables and should therefore be used appropriately.

The correlation coefficient has values between -1 to 1:

- A value closer to 0 implies weaker correlation (exact 0 implying no correlation)
- A value closer to 1 implies stronger positive correlation
- A value closer to -1 implies stronger negative correlation

![](RackMultipart20210905-4-2p5joe_html_8ba4f1f8b90a546a.png)

2. **Mutual Info**

This method utilize the [mutual information](https://en.wikipedia.org/wiki/Mutual_information). It calculates mutual information value for each of independent variables with respect to dependent variable, and selects the ones, which has most information gain. In other words, it measures the dependency of features with the target value. The higher score means more dependent variables.

![](RackMultipart20210905-4-2p5joe_html_c4b514b64530bfbb.png)

3. **K-Best**

This class is actually a more general approach compared to the above-mentioned classes, since it takes an additional scoring function parameter which states, which function to use in feature selection. Actually, it as a kind of wrapper. We can also use f\_classif or mutual\_info\_class\_if inside this object. This object returns [p-values](https://www.r-bloggers.com/chi-squared-test/) of each feature according to the chosen scoring function as well.

Note about **f\_classif** method: It uses the ANOVA f-test for the features, and take into consideration only linear dependency unlike mutual information based feature selection, which can capture any kind of statistical dependency. Notice that the score, which are produced by different methods, are very different.

![](RackMultipart20210905-4-2p5joe_html_ce956f9bce51f544.png)

4. **Recursive Feature elimination**

**Recursive Feature Elimination** , or RFE for short RFE is a wrapper-type feature selection algorithm. This means that a different machine-learning algorithm is given and used in the core of the method, is wrapped by RFE, and used to help select features. This is in contrast to filter-based feature selections that score each feature and select those features with the largest (or smallest) score.

Technically, RFE is a wrapper-style feature selection algorithm that also uses filter-based feature selection internally. RFE works by searching for a subset of features by starting with all features in the training dataset and successfully removing features until the desired number remains. This is achieved by fitting the given machine-learning algorithm used in the core of the model, ranking features by importance, discarding the least important features, and re-fitting the model. This process is repeated until a specified number of features remains.

Note: The results output may vary given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.

# Model Selection &amp; Classification Algorithms

Model selection is the process of selecting one final machine-learning model from among a collection of candidate models for a training dataset. This process can be applied both across different types of models (e.g. logistic regression, SVM, KNN, etc.) and across models of the same type configured with different model hyperparameters (e.g. different kernels in an SVM).

In this project of binary classification, the following algorithms were examined:

- Decision Trees
- Random Forests
- Logistic Regression
- Linear Discriminant Analysis
- k-Nearest Neighbors
- Support Vector Machine
- Naive Bayes

However, in the final version of the project only: Decision Trees, Logistic Regression and Linear Discriminant Analysis were evaluated. The reason for this is that other algorithms were deemed unsuitable given the nature of the problem and the volume of the data. For example, algorithms like Naïve Bayes had distinctively low performance in terms of accuracy, while others like k-NN, SVM, and Random Forests had low performance within the training process due to the size of the dataset.

**K-Fold Cross Validation**

As there is never enough data to train the model, removing a part of it for validation poses a problem of under-fitting. By reducing the training data, we risk losing important patterns/ trends in data set, which in turn increases error induced by bias. Therefore, what we require is a method that provides ample data for training the model and leaves ample data for validation. K Fold cross validation does exactly that.

In K Fold cross validation, the data is divided into k subsets. Now the holdout method is repeated k times, such that each time, one of the k subsets is used as the test set/ validation set and the other k-1 subsets are put together to form a training set. The error estimation is averaged over all k trials to get total effectiveness of our model. As can be seen, every data point gets to be in a validation set exactly once, and gets to be in a training set k-1 times. This significantly reduces bias as we are using most of the data for fitting, and significantly reduces variance as most of the data is also being used in validation set. Interchanging the training and test sets also adds to the effectiveness of this method. As a general rule and empirical evidence, K = 5 or 10 is generally preferred, but nothing is fixed and it can take any value.

On top of the above, the script compares a group of classification algorithms (Logistic Regression, Linear Discriminant Analysis, Random Forest Classifier, Decision Trees Classifier, Gaussian Naïve Bayes and K neighbors Classifier) using different scoring functions such as accuracy, balanced accuracy, precision, recall, f1 score, cohen cappa score and roc auc score. Each algorithm can be defined in models list as shown below with or without parameters. When the brackets are empty means that the algorithm run with the default parameters as they provided by scikit-learn documentation: [https://scikit-learn.org/stable/supervised\_learning.html](https://scikit-learn.org/stable/supervised_learning.html). Each one of the models evaluated in turn using a five K Fold cross validation technique. The process for later use stores the mean scores for each metric and each iteration. Relying on the balanced accuracy score, the functions sorts the models from the highest ranked to the lowest one in an ordered dictionary. Finally, the best models were saved in order to be trained on the dataset and be further configured.

After the parameter tuning, the feature selection using RFE method (described above) is invoked in order to score each attribute importance. Then, only the attributes that are marked as important for the particular algorithm are selected in the final model that is examined.

Finally, the script continues with the model finalization, which aims to train the final model and store it in the server as a .sav file, waiting the prediction process to read it and make new predictions.

# Model Predictions

The final part of our script predicts whether a contract will churn or not, given a previously unseen test dataset, which is described above (see Data Acquisition). The main task of the script is to make the new predictions along with their probabilities for each contract number that is in renewable state.

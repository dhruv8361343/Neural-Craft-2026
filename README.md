# Neural-Craft-2026
 ## THE KAGGLE NOTEBOOK CONTAINS THE CODE FOR SOLVING THE PROBLEM . 
 ### The approach was to use XGBoost  Classifier as ML model for all the 3 tasks and then used primary category as a feature for secondary category and secondary category,primary category as feature for severity . 

 the various parameters of model used are as follows --
 # model for primary category
  ```python
  eval_metric="mlogloss",
    n_estimators=1200,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    tree_method="hist",
    n_jobs=-1
```

# model for secondary category
```python

secondary_model = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_estimators=1500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2,
    reg_alpha=0.5,
    tree_method="hist",
    n_jobs=-1
)
 ```
# model for severity

```python
severity_model = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_estimators=1200,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    tree_method="hist",
    n_jobs=-1
)

```

The workflow includes:

Text preprocessing and cleaning

Feature extraction using TF-IDF vectorization

Handling class imbalance

Training models using XGBoost

Evaluating performance using standard classification metrics

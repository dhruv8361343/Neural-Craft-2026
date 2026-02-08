# Neural-Craft-2026
 ## THE KAGGLE NOTEBOOK CONTIANS THE CODE FOR SOLVING THE PROBLEM . 
 ### The approach was to use XGBoost  Classifier as ML model for all the 3 tasks and then used primary category as a feature for secondary category and secondary category,primary category as feature for severity . 

 the various parameters of model used are as follows --
  ```

  eval_metric="mlogloss",
    n_estimators=1200,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    tree_method="hist",
    n_jobs=-1
```

The workflow includes:

Text preprocessing and cleaning

Feature extraction using TF-IDF vectorization

Handling class imbalance

Training models using XGBoost

Evaluating performance using standard classification metrics

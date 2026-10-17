# Credit Card Fraud Detection

Detecting fraudulent credit card transactions with logistic regression, a neural network, a decision tree, a random forest and XGBoost, on data where only 0.17% of transactions are fraud.

## The problem

Out of 284,807 transactions, only 492 are fraud. A model that says "normal" for every transaction is right 99.83% of the time and catches zero frauds, so accuracy is useless here. This project scores every model by how many frauds it catches (recall) and how many of its flags are real frauds (precision).

## Data

Credit card transactions made by European cardholders over two days in September 2013, collected by Worldline and the Machine Learning Group of the Université Libre de Bruxelles. 284,807 transactions, of which 492 are fraud. The columns are `V1` to `V28` (scrambled for privacy, so their meaning is unknown), `Amount`, and `Class` (1 for fraud).

The copy used here comes from [OpenML](https://www.openml.org/d/1597) and has no `Time` column. The original is on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The data file is not included in this repository. The notebook downloads it automatically. Please check the licence on the source page before reusing the data.

## Method

- **Three groups:** the data was split into a fit group (170,883 transactions, 295 frauds) for training, a validation group (56,962, 99 frauds) for choosing settings, and a test group (56,962, 98 frauds) used only for the final scores. The split keeps the same fraud share in every group.
- **Scaling:** `Amount` was rescaled using the average and spread of the fit group only.
- **Choosing the cut-off:** each model gives a chance of fraud for every transaction. Instead of using 0.5, the cut-off was chosen on the validation group by one rule: catch as many frauds as possible while at least half of the flagged transactions are real frauds. That rule is this project's choice. A bank could pick differently.
- **Models:** logistic regression, a neural network (two hidden layers of 32 and 16 units, trained for 10 passes), a decision tree (depth 6), a random forest (100 trees, depth 8) and XGBoost (early stopping on the validation group). The tree-based models and the network used class weights so that a fraud counts for more than a normal transaction.

## Results

Scores on the test group (98 frauds), each model at its own cut-off chosen on the validation group:

| Model | Frauds caught | Frauds missed | False alarms | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 86 | 12 | 102 | 0.878 | 0.457 |
| Neural network | 86 | 12 | 101 | 0.878 | 0.460 |
| Decision tree | 79 | 19 | 81 | 0.806 | 0.494 |
| Random forest | 84 | 14 | 109 | 0.857 | 0.435 |
| XGBoost | 85 | 13 | 113 | 0.867 | 0.429 |

No model is clearly better. With only 98 frauds in the test group, differences of a few frauds are within noise.

### What choosing the cut-off changed

| Setup | Frauds caught | False alarms |
| --- | --- | --- |
| Logistic regression, default 0.5 cut-off | 63 | 13 |
| Logistic regression, class weights, default 0.5 cut-off | 90 | 1,439 |
| Neural network, class weights, default 0.5 cut-off | 89 | 1,117 |
| Logistic regression, cut-off chosen on validation | 86 | 102 |

Choosing the cut-off carefully caught almost as many frauds as the class-weight versions, with about 11 to 14 times fewer false alarms.

### Overfitting and learning rate

Training the neural network for more passes made it worse on unseen data. The validation loss was lowest around pass 3 and then rose, while the training loss kept falling. With early stopping, training stopped after pass 6 (best pass 3) with a validation average precision of 0.683, against 0.693 for the 10-pass network, so the same quality in fewer passes.

Validation average precision for four learning rates, same network and seed:

| Learning rate | Final training loss | Validation average precision |
| --- | --- | --- |
| 0.0001 | 0.2674 | 0.572 |
| 0.001 | 0.0769 | 0.693 |
| 0.01 | 0.0395 | 0.689 |
| 0.1 | 0.2157 | 0.644 |

The smallest rate was too slow to learn in 10 passes, and the largest was unstable. Anything from 0.001 to 0.01 worked about equally well.

### How much does one split matter?

The whole procedure was repeated for logistic regression and the decision tree with five different random shuffles:

| | Frauds caught | False alarms |
| --- | --- | --- |
| Logistic regression | 83 to 88 (average 85.6) | 63 to 100 (average 83.6) |
| Decision tree | 74 to 83 (average 80.0) | 26 to 115 (average 69.4) |

Logistic regression caught more frauds in every shuffle and was steadier. The tree's false alarms swung widely, and in 2 of the 5 shuffles it never reached the precision rule, so a backup rule (the best balance of precision and recall) was used for those. A single split can mislead when there are this few frauds.

## Conclusions

- Accuracy hides everything in this data, so recall and precision are the scores that matter.
- Choosing the cut-off on a separate validation group mattered more than the choice of model.
- A simple logistic regression did as well as a neural network, a random forest and XGBoost, and was more consistent than a single decision tree.

## Limitations

- Two days of data from 2013, from European cardholders. The models may not work on other periods or regions.
- The columns are scrambled, so the models cannot be interpreted, and the original `Time` column is missing from this copy.
- The test group has only 98 frauds, so small differences between models are noise. Only logistic regression and the decision tree were repeated across shuffles. The other three models were run once per setting.
- Even the best setup makes about one false alarm for every fraud it catches. This is a learning project, not a system a bank could deploy.

## Run it yourself

The notebook was run in Google Colab, which has everything installed. To run it elsewhere:

```
pip install pandas scikit-learn tensorflow xgboost matplotlib seaborn
```

## Files

- `credit_card_fraud_detection.ipynb`: the full notebook, from loading the data to comparing the models

## Author

James-Mawuena Abraham, Computer Engineering, Federal University of Technology, Akure. [LinkedIn](https://www.linkedin.com/in/jamesmawuena)

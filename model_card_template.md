# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
**Model Type:** Classification Model
**Model Architecture:** The model uses a supervised machine learning algorithm trained on census data to predict whether an individual earns more than $50,000 per year. A Random Tree Forest is the chosen decision model.

## Intended Use
This model is intendede to predict whether an individual's income exceeds $50,000 annually based on census data. This model can be used in applications that require income prediction for individuals in similar demographic catgories.

## Training Data
The model is trained on the **census.csv** dataset, which contains census data.

The data includes various demographic attributes such as age, education level, marital status, and occupation.

The training dataset consists of approximately 32,000 samples. The datasets has both numerical and categorical features.

The categorical features were one-hot encoded, and the target variable, **salary** was binarized using a label binarizer. Data was split into training and test datasets.

## Evaluation Data
The model was evaluated on a separate test set, which consists of 20% of the original data. Precision, Recall, and F1-score were used to assess the model's performance on the test dataset.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
**Precision:** 0.7419
**Recall:** 0.6384
**F1-Score:** 0.6863

These metrics reflect the model's performance in predicting the positive class, which are individual's whose income is greater than $50K, with a focus on balancing false positives and false negatives.

## Ethical Considerations
The model may inherit biases present in the census data. Some demographic groups may be underrepresented, which could affect the fairness of predictions for those groups.

## Caveats and Recommendations
The model's performance is dependent on the **quality and accuracy of the input data**. If the input data is biased or unrepresentative ofthe population, the model's predictions may not generalize well.

Some features, such as education level and occupation, are likely more relevant in predicting income, but this should be verified through **feature importance analysis**.

This model should be used in a research or policy-making context where predictions can help inform decisions, but it should not be used for high stakes decisions such as hiring, law enforcement, or lending, where fairness and transparency are critical.
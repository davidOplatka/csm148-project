# Appendix

In this section, we discuss the methods we applied to reach our final model, which you can read about in our main report.

## Exploratory Data Analysis

Our project's overall goal was to use information about students to predict their final exam score.
The first variable we explored was the outcome variable, `Exam_Score`. 
By plotting a histogram of this variable, we say that the distribution of exam scores was skewed to the right, with most values centered around the median score of 67.
Without these outliers, the distribution of exam scores was roughly symmetric, so we believed that a linear model may be appropriate for predicting a student's final exam score.

To investigate this further, we calculated the correlation between `Exam_Score` and all of the numeric variables in our dataset and plotted a correlation heatmap.
It quickly became clear that `Attendance` and `Hours_Studied` had the strongest influence on a student's exam score, with correlation coefficients of 0.581 and 0.445, respectively.
While neither of these coefficients are particularly high, they were strong enough to indicate that there may be a linear relationship between `Exam_Score` and the other features in our dataset.
We confirmed this by plotting scatterplots of `Exam_Score` against the other numeric features, and we could visually see a linear relationship in many of these pairwise plots.

To investigate the categorical variables, we created side-by-side boxplots of `Exam_Score` against the categorical variables.
The variable that stood out to us the most was `Access_to_Resources`, in which students with higher access to resources earned higher exam scores, on average, than students with lower access.

For training and testing our models, we randomly split our data set, using 80% of the data for training and validation (with cross validation) and the remaining 20% for testing.

## Data Preprocessing and Feature Engineering

The original dataset for our project contained 3 columns with missing values. These columns were `Teacher_Quality`, `Parental_Education_Level`, and `Distance_from_Home`.

* The `Distance_from_Home` variable had 3 levels (Near, Moderate, and Far), and we noticed almost no difference in the distribution of the `Exam_Score` (the variable we are trying to predict)
across these three levels. Given this, our team made the decision to simply drop this column from the dataset, allowing us to preserve as many data points as possible without data imputation.

* As for the other two variables, `Teacher_Quality` and `Parental_Education_Level`, we believed that these variables were more likely to influence a student's `Exam_Score`,  	
so rather than dropping these columns, we dropped the rows for which these variables were missing. In total, this meant dropping 164 observations of the original 6,607 (About 2.5%).

The next thing we checked were the actual values of the data itself. Almost all values seemed reasonable, with a single `Exam_Score` exceeding 100 (101).
While this may imply that a student earned extra credit on their exam, our team wanted to keep all exam scores in the range of 0 to 100, so we simply changed this score to 100.

Now that the data was clean, we needed to prepare the data for our models. The most important step here was dealing with the categorical variables. We had several binary categorical variables as well as ordinal categorical variables.

For the binary categorical variables, we encoded the original values as 0s and 1s:

* `Extracurricular_Activities` - `'No'`: 0, `'Yes'`: 1
* `Internet_Access` - `'No'`: 0, `'Yes'`: 1
* `School_Type` - `'Private'`: 0, `'Public'`: 1
* `Learning_Disabilities` - `'No'`: 0, `'Yes'`: 1
* `Gender` - `'Female'`: 0, `'Male'`: 1

Many of the ordinal categorical variables were measured on a scale of Low-Medium-High. We encoded these variables as `'Low'`: 0, `'Medium'`: 1, `'High'`: 2.
These variables were `Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Family_Income`, and `Teacher_Quality`.

Finally, we encoded:

* `Peer_Influence` - `'Negative'`: 0, `'Neutral'`: 1, `'Positive'`: 2
* `Parental_Education_Level` - `'High School'`: 0, `'College'`: 1, `'Postgraduate'`: 2

## Regression Analysis

Linear Regression was used as a classification model to predict hours studied vs. exam score for a student. In our final model, we used more predictors to better predict exam score. With Linear Regression, we perform it on a training and validation set, then a test set to evaluate our performance. When comparing true to predicted values in our validation, we get the following metrics in our primary lab.

Key Metrics:
* R^2: 0.20152
* Correlation: 0.45096

With a low R^2 followed by not a strong correlation, we saw that our model is underfitting. Thus, through ridge and lasso regression, we use 10-fold cross-validation to first select the best lambda and alpha values and then loop through to compute scores. Fitting the ridge and lasso model helps minorly, but there were not greate improvements from the linear regression model above. 

Here are some metrics:
* Lasso (Min) RMSE: 3.4267618338079435
* Lasso (Min) Correlation Coefficient: 0.444495780036996
* Lasso (Min) R^2: 0.19689257769408797

* Lasso (1SE) RMSE: 3.472772895242439
* Lasso (1SE) Correlation Coefficient: 0.4444957800369967
* Lasso (1SE) R^2: 0.17518117640547648

* Ridge (Min) RMSE: 3.4267618338079435
* Ridge (Min) Correlation Coefficient: 0.444495780036996
* Ridge (Min) R^2: 0.19689257769408797

* Ridge (1SE) RMSE: 3.4267618338079435
* Ridge (1SE) Correlation Coefficient: 0.444495780036996
* Ridge (1SE) R^2: 0.19689257769408797

Since there were little improvements both metrically and visually through a model, we concluded that ridge and lasso regression did not significantly help considering the use of simply one factor, that being hours studied. Given the lack of datapoints and variability, it made it difficult for the graph to use regression. Given in our final model we use multiple predictors, ridge and lasso regression can serve as more useful indicators towards performance improvement.

## Logistic Regression

Logistic regression was used as a binary classification model to predict whether a student would attend a public or private school based on various predictors. The feature selection technique used was Recursive Feature Elimination (RFE). With RFE, the model is iteratively trained on the data, and at each step the least important features (as indicated by the model’s coefficients) are removed. This process continues until only the most influential features remain, helping to simplify the model and potentially improve its performance. From this, the key predictors were: parental involvement, access to resources, and parental education level.

Since these predictors were all categorical variables, they needed to be encoded. We converted categorical features into numeric representations in two primary ways. For some variables, each category was turned into its own binary column (one-hot encoding), indicating presence (1) or absence (0) of a category. For others, the categories were replaced with integers that represent different categories, effectively putting them on a numeric scale (label encoding).

Regularization was implicitly applied through the default “l2” penalty in LogisticRegression. This helped keep coefficients more stable and reduced overfitting, even though the dataset and feature set were relatively simple. Without regularization, certain predictors might have dominated the model, leading to poorer generalization. Thus, yes, regularization was needed and did support the interpretability and stability of the logistic regression model.

Here are some key metrics:
* Prediction Accuracy: 0.5120
* Prediction Error: 0.4880
* True Positive Rate (Recall): 0.4797
* True Negative Rate (Specificity): 0.5263

The logistic regression model wasn’t the best because a prediction on whether a student would attend a public or private school was hard to predict given possible confounding variables and there were not suitable predictors.

## KNN

TODO: Henry tk

## PCA and Clustering

Our dataset had six numeric features to which we applied principal component analysis.
After standardizing the variables, we applied PCA but found no meaningful results.
Each principal component explained about the same proportion of the total variance, so reducing the dimensionality of these numeric features is not practical.
These results are unsurprising because we only have 6 numeric features and would hope that they all have meaningful contributions independently to the total variance in the data.

As for clustering, we performed K means clustering with k=2, k=3, and k=4.
With k=3, we were able to see good separation of the data across the number of tutoring sessions and the class attendace.
In general, these methods were fairly unsuccessful, achieving silhouette scores of 0.128, 0.122, and 0.119 for k=2, 3, and 4, respectively.
Thus, this method was not helpful in predicting the `Exam_Score` from our features.
Even though we saw some separation across a few of our variables, this can also be captured by the distribution of the variable values themselves as inputs to a model.

Ultimately, PCA was more useful for our projet because it demonstrated that most, if not all, of our numeric features were contributing significant variance to the overall data.
This told us that our final model would likely include most of these variables (in the end, 5 out of 6 numeric variables were included in our final model).

## Neural Network

The final method we implemented to predict students' Exam Scores was a Multi-Layer Perceptron (MLP) neural network. Below are the key steps of our process:

We used the cleaned dataset `StudentPerformanceFactorsCleaned.csv` and performed further preprocessing before feeding it into the neural network including standardizing all numerical columns to ensure all features were on the same scale, preventing any one feature from dominating the model; one-hot ecoding all categorical columns into numeric form with values of 0 and 1; splitting the dataset into training, validation, and test sets in proportions of 60%, 20%, and 20%, respectively; finally converting the preprocessed data into PyTorch tensors for compatibility with the neural network.

Next we create defined the **Multi-Layer Perceptron**: with two hidden layers:
  * *First Layer*: 128 neurons with ReLU activation and a dropout rate of 30%.
  * *Second Layer*: 64 neurons with ReLU activation and a dropout rate of 30%.

During the training, we use *MSE* for the loss function and Adam optimizer with a learning rate of 0.01, selected after testing lternative rates (0.01, 0.005, and 0.0001). We choose to set *batch size* to be 32, balacing performance and computational efficiency and choose the number of *epochs* to be 2000.

The process of Validation and Evaluation is:
At first we have overfiiting issue since we have phenomenon of low loss and high test MSE, therefore, we implement the **early stopping mechanism** which monitored validationed loss with a patience theshold of 35 epochs. Specifically, if the validation loss did not improve for 35 consecutive epochs, training stopped early to prevent overfitting.

As the result, evaluation stops at *epoch 318*, we get a training loss of *33.5123*. The test loss *MSE* of *3.8128578662872314*.



## Hyperparameter Tuning


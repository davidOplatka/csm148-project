# Main Report

In this section, we discuss an overview of our project, including the data set our team used, the problem we wanted to address, our key methodologies, and results.
We also offer instructions for how to run our code to reproduce our results.

## The Dataset

TODO

## Problem Overview

The primary ojective of this project is to identify the key factors that influence a students' academic performance, as measured by their final exam grades. By analyzing the available dataset, our team sought to develop a predictive model that can forcast students' exam scores.

This model is valuable as it offers important implications. It can help students identify some changes (such as attending class or tutoring sessions) that may improve their grades and overall academic performance. It can also offer educators and administrators insights on how to build a more supportive environment to enhance academic performance. Finally, we might gain some insight into some inequities such as how students' varying access to resources or their parents' education level may influence their academic success.

Therefore, our team hoped to find a simple, yet effective solution that provided highly interpretable results into the drivers of student success.

## Key Methodologies

After exploring many different relationships between numeric and categorical variables, our team concluded that a linear model would be the best at predicting a student's exam score.
In our exploratory data analysis, we observed moderately strong linear associations between a student's exam score and their attendance history (r=0.569) and hours studied (r=0.435).
We also observed differences in students' median exam scores based on categorical variables, such as students' access to resources (students with higher access to resources earned higher median exam scores).
With these discoveries, our team believed that a linear model would be a simple but effective model for this problem, allowing us to achieve a high overall accuracy and avoid overfitting an unnecessarily complex model.

To train this model, we first standardized the numeric features by converting the raw values into z-scores (mean-centering and scaling).

We then constructed a linear model to predict exam scores using all of the features in our dataset.
We observed that nearly all features were statistically significant at a significance level of $\alpha = 0.05$.
The only features that were not significant were:

* whether the student attended public or private school
* the student's gender (male or female)
* the student's average daily sleep hours

We removed these features and fit the model again, achieving a higher F-statistic.

To further reduce the chance of overfitting, we then applied regularization techniques to determine if any other variables should be dropped from our model and to try to make the model more generalizable to a test data set.
After applying LASSO regularization with cross validation, we determined that all the remaining features in our linear model were important.
Finally, we applied ridge regression with cross validation to decrease the magnitude of the coefficients and reduce any overfitting to the training data.
This resulted in a model with a low test mean squared error and a high $R^2$ value.

We will further discuss the results of our model in the next section.

## Results

We utilized 5-fold CV for both our ridge and lasso models, resulting in the following alpha and model coefficient values. We choose to use K-fold CV because it helped us find the best regularization coefficients without overfitting to the training data.

Lasso Alpha and Coefficients:
* Best Alpha is 0.0022
* Best Coefficients are [ 1.719  2.287  1.025  1.038   0.530  0.717  0.493  0.806  0.638  0.516  0.529  0.501  0.195 -0.895  0.467]
* Best Intercept is 61.623

Ridge Alpha and Coefficients:
* Best Alpha is 9.0
* Best Coefficients are [ 1.719  2.286  1.026  1.039  0.535   0.719  0.496  0.817  0.639  0.519  0.533  0.504 0.197 -0.902  0.469]
* Best Intercept is 61.595

Note: the variables corresponding to the coefficients in these models are ... INCLUDE FEATURES
  
Thus, in our results we notice that our ridge regression model performs exceptionally well, making predictions that are close to the actual exam scores. When plotting a scatterplot of the actual vs. predicted scores, nearly all of the points are very close to the line `y=x`, demonstrating the strong accuracy of our model. With a `R^2` of about 0.72, we notice that this model captures the variance in the test scores quite accurately as well.

Key Metrics:
* `Ridge CV MSE`: 3.066807
* `Ridge CV R^2`: 0.719789
* `LASSO CV MSE`: 3.067022
* `LASSO CV R^2`: 0.719646

With a nice MSE and R^2 value, our ridge regression model performs well, which is why we picked it as our best and final model. While most predicted scores mostly align with our actual scores, we notice some outliers that are present. A small number of students scored significantly higher than what was predicted, possibly showing factors that influenced score but weren't a feature in the dataset. What this could suggest is that while the model performs overall well, there are opportunities to incorporate additional features or further refine preprocessing, while preventing overfitting as well.

## How to Reproduce Results

Note: refer to the code blocks in main.ipynb that correspond to these instructions.

1. Have Python installed with following libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, and statsmodels. Install packages using:
```
$ pip install <package_name>
```
2. Download StudentPerformanceFactors.csv dataset. Use df.head(), df.info(), and df.describe() to confirm data is loaded correctly.
3. We can drop unnecessary columns, remove rows with missing values, and correct any out-of-range values. Features like distance from home, sleep hours, school type public, and gender male were found to hinder the model, so we can drop these features. Set exam scores > 100 to be 100.
4. Create boxplots, heatmaps, and scatterplots to confirm the relationships and patterns match those described in the report.
5. Use the provided preprocessing functions to transform categorical variables into numeric form so that models can properly interpret all input variables. We used binary and sequential label encodings.
6. Split the data into training and testing sets. We used a 80-20 train-test split. Then, run feature scaling code to standarize numeric values for all predictors before applying the models.
7. Run the OLS regression model and the corresponding analysis. Then, proceed to Lasso and Ridge regression models. The code includes the necessary hyperparameters.
8. Review metrics like MSE and R-squared. The Ridge model should yield slightly better generalization performance. Also, create a scatter plot comparing actual vs. predicted exam scores. If the points align closely with y=x, the model is performing as expected.

# Main Report

In this section, we discuss an overview of our project, including the data set our team used, the problem we wanted to address, our key methodologies, and results.
We also offer instructions for how to run our code to reproduce our results.

## The Dataset

TODO

## Problem Overview

The goal of this project is to predict students' final exam grades based on various features and data provided in the dataset.

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

In our results we notice that our Ridge model performs exceptionally well, close to the actual exam scores when comparing actual vs. predicted scores, fitting a `y=x` line almost perfectly. With a `R^2` of 0.72, we notice that this model captures our variance quite accurately as well.

Key Metrics:
* `Ridge CV MSE`: 3.066807
* `Ridge CV R^2`: 0.719789
* `LASSO CV MSE`: 3.067022
* `LASSO CV R^2`: 0.719646

With a nice MSE and R^2 value, our model performs well, especially better than our Lasso model, which is why we pick our Ridge model over our Lasso model. While most predicted scores mostly align with our actual scores, we notice some outliers that are present. A small number of students scored significantly higher than what was predicted, possibly showing factors that influenced score but weren't a feature in the dataset. What this could suggest is that while the model performs overall well, there are opportunities to incorporate additional features or further refine preprocessing, while preventing overfitting as well.

## How to Reproduce Results

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

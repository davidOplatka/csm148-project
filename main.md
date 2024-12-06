# Main Report

In this section, we discuss an overview of our project, including the data set our team used, the problem we wanted to address, our key methodologies, and results.
We also offer instructions for how to run our code to reproduce our results.

## The Dataset

TODO

## Problem Overview

TODO 

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

TODO: Maddox

## How to Reproduce Results

TODO: Joseph
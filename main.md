# Main Report

In this section, we discuss an overview of our project, including the data set our team used, the problem we wanted to address, our key methodologies, and the results.
We also offer instructions for how to run our code to reproduce our results.

## The Dataset

We used the student performance dataset, found on Kaggle at:
* https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

This dataset tracks student performance on a final exam, as well as several potential contributing factors to that performance.

We chose this dataset because we were interested in student performance, being students ourselves who want to optimize our own performance. We wanted to see if there was anything we could do to improve our own performance. Distance from home, parental education level, and teacher quality had a few missing values, while none of the others did. Next, we have a description of each of the features of the data.

### Numeric Features

* The number of hours studied
* Percentage of Classes Attended
* Hours slept per night
* Previous exam scores
* Tutoring sessions attended per month
* Amount of physical activity
* Final exam score

### Categorical Features

* Parental involvement (Low, Medium, High)
* Access to educational resources (Low, Medium, High)
* Whether or not they participated in extracurricular activities
* The student's motivation level (Low, Medium, High)
* Whether the student had access to the internet
* The student's family income (Low, Medium, High)
* Teacher Quality (Low, Medium, High)
* Whether they attended public or private school
* How peers influenced academic performance (Positive, Neutral, Negative)
* Whether the student had learning disabilities
* The student's parents' education level (High School, College, Postgraduate)
* The distance from the student's home to their school (Near, Moderate, Far)
* The student's gender (Male, Female)

## Problem Overview

The primary objective of this project is to identify the key factors that influence students' academic performance, as measured by their final exam grades. By analyzing the available dataset, our team sought to develop a predictive model that can forecast students' exam scores.

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
After applying LASSO regularization with cross-validation, we determined that all the remaining features in our linear model were important.
Finally, we applied ridge regression with cross-validation to decrease the magnitude of the coefficients and reduce any overfitting to the training data.
This resulted in a model with a low test mean squared error and a high $R^2$ value.

We will further discuss the results of our model in the next section.

## Results

We utilized a 5-fold CV for both our ridge and lasso models, resulting in the following alpha and model coefficient values. We chose to use K-fold CV because it helped us find the best regularization coefficients without overfitting to the training data.

Lasso Alpha and Coefficients:
* Best Alpha is 0.0022
* Best Coefficients are [ 1.719  2.287  1.025  1.038   0.530  0.717  0.493  0.806  0.638  0.516  0.529  0.501  0.195 -0.895  0.467]
* Best Intercept is 61.623

Ridge Alpha and Coefficients:
* Best Alpha is 9.0
* Best Coefficients are [ 1.719  2.286  1.026  1.039  0.535   0.719  0.496  0.817  0.639  0.519  0.533  0.504 0.197 -0.902  0.469]
* Best Intercept is 61.595

Note: the variables corresponding to the coefficients in these models are `Hours_Studied`, `Attendance`, `Parental_Involvement`, `Access_to_Resources`, `Extracurricular_Activities_Yes`, `Previous_Scores`, `Motivation_Level`, `Internet_Access_Yes`, `Tutoring_Sessions`, `Family_Income`, `Teacher_Quality`, `Peer_Influence`, `Physical_Activity`, `Learning_Disabilities_Yes`, `Parental_Education_Level`.
  
Thus, in our results we notice that our ridge regression model performs exceptionally well, making predictions that are close to the actual exam scores. When plotting a scatterplot of the actual vs. predicted scores, nearly all of the points are very close to the line `y=x`, demonstrating the strong accuracy of our model. With an `R^2` of about 0.72, we notice that this model captures the variance in the test scores quite accurately as well.

Key Metrics:
* `Ridge CV MSE`: 3.066807
* `Ridge CV R^2`: 0.719789
* `LASSO CV MSE`: 3.067022
* `LASSO CV R^2`: 0.719646

With a nice MSE and R^2 value, our ridge regression model performs well, which is why we picked it as our best and final model. While most predicted scores mostly align with our actual scores, we notice some outliers that are present. A small number of students scored significantly higher than what was predicted, suggesting the existence of factors that influenced scores but weren't features in the dataset. What this could suggest is that while the model performs overall well, there are opportunities to incorporate additional features or further refine preprocessing, while preventing overfitting as well.

Since we standardized our numeric features before fitting our models, we can use the magnitudes of the coefficients to compare the importance of the features. As we can see, `Attendance` (2.286) and `Hours_Studied` (1.719) have the greatest magnitudes among all features, so these are the most important features in our predictive model. We also see large coefficients for `Access_to_Resources` (1.039) and `Learning_Disability_Yes` (-0.902), which may highlight some of the social challenges or inequities that put some students at a disadvantage when it comes to evaluation through exam scores. This gives us some insight into the strongest contributors in our model. However, we must be clear that we cannot conclude a causal link between these students' exam scores and these features because this is an observational study, not an experiment. Thus, this evaluation must be taken with extreme caution, and further studies will be needed to assess the true importance of these individual features on students' testing performance.

## How to Reproduce Results

### Method 1

Note: refer to the code blocks in main.ipynb that correspond to these instructions.

1. Have Python installed with the following libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, and statsmodels. Install packages using:
```
$ pip install <package_name>
```
2. Download StudentPerformanceFactors.csv dataset from the kaggle link in the "The Dataset" section. Use df.head(), df.info(), and df.describe() to confirm data is loaded correctly.
3. We can drop unnecessary columns, remove rows with missing values, and correct any out-of-range values. Features like `Distance_from_Home`, `Sleep_Hours`, `School_Type`, and `Gender` were found to be insignificant to the final model, so we can drop these features. Set any exam scores > 100 to 100.
4. Create boxplots, heatmaps, and scatterplots to confirm the relationships and patterns match those described in the report.
5. Use the provided preprocessing functions to transform categorical variables into numeric form so that models can properly interpret all input variables. We used binary and sequential label encodings.
6. Split the data into training and testing sets. We used an 80-20 train-test split. Then, run the feature scaling code to standardize numeric values for all predictors before applying the models.
7. Run the OLS regression model and the corresponding analysis. Then, proceed to Lasso and Ridge regression models. The code includes the necessary hyperparameters.
8. Review metrics like MSE and R-squared. The Ridge model should yield slightly better generalization performance. Also, create a scatter plot comparing actual vs. predicted exam scores. If the points align closely with y=x, the model is performing as expected.

### Method 2

Alternatively, simply run all cells in order in our provided [main.ipynb](https://github.com/davidOplatka/csm148-project/blob/main/main.ipynb) file, installing any required packages in a new cell at the top beforehand. This code snippet will install all needed packages and can be copied and pasted into a cell at the beginning of the linear modeling file.

```
$ pip install numpy
$ pip install pandas
$ pip install matplotlib
$ pip install seaborn
$ pip install scikit-learn
$ pip install statsmodels
```

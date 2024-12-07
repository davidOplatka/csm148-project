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

This is disucssed in the main document.

## Logistic Regression

TODO: Joseph

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

### 1. Dataset and Preprocessing
We used the cleaned dataset `StudentPerformanceFactorsCleaned.csv` and performed further preprocessing before feeding it into the neural network:
- **Standardization**: Standardized all numerical columns to ensure all features were on the same scale, preventing any one feature from dominating the model.
- **One-Hot Encoding**: Encoded all categorical columns into numeric form with values of 0 and 1.
- **Data Splitting**: Split the dataset into training, validation, and test sets in proportions of 60%, 20%, and 20%, respectively.
- **PyTorch Tensors**: Converted the preprocessed data into PyTorch tensors for compatibility with the neural network.


### 2. Model Architecture
We defined a **Multi-Layer Perceptron (MLP)** with the following layers:
- **Input Layer**: Accepts the number of features in the dataset.
- **Two Hidden Layers**:
  1. **First Layer**: 128 neurons with ReLU activation and a dropout rate of 30%.
  2. **Second Layer**: 64 neurons with ReLU activation and a dropout rate of 30%.
- **Output Layer**: A single neuron to output a continuous value for regression tasks.

---

### 3. Training Process
- **Loss Function**: Used **Mean Squared Error (MSE)**, suitable for regression tasks.
- **Optimizer**: Adam optimizer with a learning rate of 0.001, selected after testing alternative rates (0.01, 0.005, and 0.0001).
- **Batch Size**: Set to 32, balancing performance and computational efficiency.
- **Epochs and Early Stopping**:
  - Initial epochs: 2000.
  - **Early Stopping Mechanism**: Monitored validation loss with a patience threshold of 35 epochs. Training stopped early at **epoch 318** when the validation loss stopped improving.

---

### 4. Validation and Evaluation
During training:
- At each epoch, the validation loss was calculated and compared to the best validation loss observed so far.
- If the validation loss did not improve for 35 consecutive epochs, training stopped early to prevent overfitting.

---

### 5. Results
- At **epoch 318**, the training loss was **33.5123**.
- The final **Mean Squared Error (MSE)** on the test dataset was **3.8128578662872314**.



## Hyperparameter Tuning


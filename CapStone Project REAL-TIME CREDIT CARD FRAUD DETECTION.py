#!/usr/bin/env python
# coding: utf-8

# Our model is a prediction classification which aims to predict the default of credit card utilizing features such as age, pay, sex, marital status, etc.
# 
# Documentation will be broken down into the following groups…
# 
# - Visual Problem Statement 
# - Visual Data Dictionary
# - Troubles that occurred during the project
# -  EDA /  Exploratory Data Analysis
# -  Feature Engineering
# -  Choice of Models and Evaluation Metrics
# -  Step by Step Instructions to reproduce solution from scratch
# -  Key Insights
# -  Suggestions or Recommendations
# -  5-minute video Presentation
# 
# Data cleaning
# 
# -  Find missing values.
# -  Find NaN and 0 values.
# -  Do all columns have the same dtypes?
# -  Convert dates to datetime types.
# -  You can use the python package arrow or datetime.
# -  Convert categorical variables to type 'category' if working with pandas.
# -  Convert strings to ints or floats if they represent numbers.
# -  Standardize strings
# -  Convert them to lower case if possible.
# -  Replace spaces with underscores or dashes.
# -  Remove white spaces around the string this is very critical.
# -  Check for inconsistent spellings typically done manually.
# -  Look for duplicate rows or columns.
# -  Look for preprocessed columns; example: A categorical column that has been duplicated with categorical labels.
# 
# 
# Data preparation
# 
# - Convert categorical features to dummy indices if you are doing regression or assign numerical labels if you are doing classification
#  
# - Do test train split to generate a test set. Further do a train validation split, you will need to run the test train split function from sklearn twice for this purpose
# - 
# 
# Exploratory Data Analysis and Data Visualization
# 
# -  Identify outliers in the datasets. Keep track of them, we want to run to train the model with the outliers and without them to see their effect.
# - Check for imbalance in the target variable. Quantify the imbalance.
# - Pairplot if possible to check the relationship between all the features and the target.
# - Look at the histogram for each variable, try to identify if you have a symmetric or normal distribution.
# - If possible plot a QQ plot to check the normality of the data. If you want more information, refer to this.
# - If it's a classification problem, run a chi-square test between each categorical feature and the target to check for correlation and run ANOVA between the continuous/discrete features and the target to check for correlations.
# - If it's a regression problem, get Pearson correlations between the continuous features and target and run ANOVA between each categorical variable and target.
# - Check for correlations between individual features; use similar approaches as you did with the target.
# 
# 
# Key Insights from EDA
# 
# At the end of this section, you should have written down the following:
# 
# - A bullet point list of the relationship between features and target and between individual features.
# - A written summary of the conclusions of the exploratory data analysis.?The first point involves writing down what type of correlations observed between each feature and target. For regression, you can state the correlation value ( r-squared value) between the feature and the target for the classification; you can state the p-value of the chi-square test with the conclusion of whether you are accepting or reject the null hypothesis. The same should be done for between individual features.?The second part involves writing a small summary of part 1. It would be best if you mentioned in words the conclusion that you reached.
# - There may be situations where you may be compelled to drop a variable because you have either too many outliers or strong correlations with the output. You may even want to create new variables using external data that you have imported. You should document and discuss these changes in this section.
# - If you are working on a problem that has industry relevance, this is also a place where you can discuss the results in the context of the industry.
# 
# Data visualizations
# 
# - For each pair of variables (i.e, feature vs. feature and feature vs. target) you need to have a separate section with visualizations. Based on the type of target and feature, you may need to choose an appropriate plot (Histogram, time-series plots, scatter plots, etc.)
# - 
# - Make sure that you label the plots properly: Each plot should have the following:
# - 
# - Readable and descriptive axis labels.
# - Font size of at least 12 for the x-y axis tick labels.
# - A title.
# - A legend that labels a curve. Even if you have a single curve.
# - If you are using a scatter plot, no fancy points, use circles, triangles, and other simple geometric shapes.
# - Make sure that everything is easy to read plot. Use colors meaningfully; most plots do not need several colors.
# - Try to make sure that you start at the origin (x=0, y=0). Be aware of the scale of your data.
# 
# 
# -  Ensure you save your final visualizations in the /images/ folder and call them from the images folder. Make sure you are given the images meaningful names. For example, you can name an EDA image between two features as "eda_scatter_feature1_feature2.png". Everyone has their naming convention. Make sure that you stick to that convention.
# 
# Model Building
# 
# -  sure you train the model then get predictions on the training set. We want to keep track of this since we want to check for overfitting. We will compare the results from the training set to the testing set to see if we are overfitting.
# 
# - You should have a plot that shows the training set points, and training set predictions overlaid
# 
# - Typically, if you have high training accuracy (or low r-square value for regression) and low testing set accuracy ( or high r-squared value for regression), it means you are overfitting. In such a situation, you need to go to a more complex model. Perhaps use regularization or get more data.
# 
# - So far, you have learned- Linear regression, Polynomial regression, Logistic regression and Naive Bayes. Hence you get to choose from these algorithms based on the type of problem, classification/regression, that you have chosen.
# 
# - If possible, you want to utilize grid search to find parameters. Grid search is great for finding parameters, especially when you have many of them. https://scikit-learn.org/stable/modules/grid_search.html
# 
# 
# Model Evaluation
# 
# Once you train your model, you need to evaluate how good your model is with the test set. For this section, you need to do the following:
# 
# 
# - Run predictions on the test set.
# - Clearly state the difference between the prediction metrics of the test set and training set
# - Discuss the difference between test set predictions and training set predictions. Is there overfitting?
# - If there is overfitting, you may have to go back to the previous step and try other approaches like cross-validation.
# - If you do have overfitting, you should document how you dealt with it. [ ] In the case of classification, you should discuss the confusion matrix's implications if you are dealing only with two classes.
# - You should have a plot that shows the test set points, and predictions overlaid.
# - Are there specific features that substantially affect the outcome? Look at correlations between the features and the predictions.
# - In some cases, you might be able to use feature importance (especially for tree methods) feature/ set of features contribute heavily to the model's success. State and plot those here. You will discuss them in the next section
# 
# 
# Key Insights from Predictive analysis
#  https://3.basecamp.com/3945211/buckets/24864192/todos/6532669867
# 
# 
# 
# Step by Step Like:
# https://app.colaberry.com/app/network/network/927/projectsteps
# 
# 
# What to figure out in the project:
# 
# - Dataset and Problem Statement - What is the problem I am trying to solve? Why is it important? Why should I use ML/DS to solve this problem? (Why DS should be a question in Stress Test 1 and Level 1 Approval)
# - EDA - How does my data look? How is it distributed? How might the distribution affect my analysis? Why is a specific class of models appropriate for the data at hand?
# - Feature Engineering - What am I trying to predict? What am I using to predict my target (what are my predictors)? What are the relationships and nature of my predictors? Which predictors should I use for modeling? Which influences the target the most? How should I transform the predictors so they can be incorporated in the model?
# - Modeling and Evaluation
# - Deployment
# 
# 
# How to select a project to solve
# - Industry to choose from (Might be provided to you)
# - Type of problem Supervised/Unsupervised, Classification/regression (Will be decided by the dataset that you choose)
# 
# 
# Problem Statement:   https://3.basecamp.com/3945211/buckets/24864192/todos/6532669814
# 
# -  Clearly state your data source.
# -  What type of data do you have?
# -  Structured/Unstructured.
# -  What are you predicting?
# -  What are your features?
# -  What is your target?
# -  What type of problem is it?
# -  Supervised/Unsupervised?
# 
# 
# 

# In[76]:


import pandas as pd
import numpy as np

# Define the number of rows and columns
n_rows = 1000
n_cols = 9  # Increased to accommodate the "date" column and the new "Credit Card Type" column

# Define the column names
col_names = ["transaction_id", "date", "amount", "card_number", "merchant_id", "location", "time", "fraud", "Credit Card Type"]

# Define the date range
start_date = pd.to_datetime("2018-01-01")
end_date = pd.to_datetime("2020-12-31")

# Define the ranges and probabilities for each column (including "date" and "Credit Card Type")
col_ranges = {
    "transaction_id": (1, n_rows),
    "date": pd.date_range(start=start_date, end=end_date, freq='D'),  # Generate daily dates
    "amount": (1, 1000),
    "card_number": (1000000000, 9999999999),
    "merchant_id": (100, 999),
    "location": ["Toronto", "Montreal", "Calgary", "Ottawa", "Edmonton", "Vancouver", "Winnipeg", 
                 "Moncton", "St. John's", "Halifax", "Charlottetown", "Saskatoon"],
    "time": (0, 23),
    "fraud": [0, 1],
    "Credit Card Type": ["Visa", "Mastercard", "American Express"]
}

col_probs = {
    "transaction_id": None,
    "date": None,  # No probability needed for dates
    "amount": None,
    "card_number": None,
    "merchant_id": None,
    "location": [0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Adjusted for 4 choices
    "time": None,
    "fraud": [0.9, 0.1],  # Adjusted for 10% fraud in 2018
    "Credit Card Type": [0.4, 0.3, 0.3]  # Adjusted for the distribution of Visa, Mastercard, and American Express
}

# Create an empty dataframe
df = pd.DataFrame(columns=col_names)

# Fill the dataframe with random values
for col in col_names:
    # Get the range and probability for the column
    col_range = col_ranges[col]
    col_prob = col_probs[col]

    # Generate random values for the "date" and "Credit Card Type" columns
    if col == "date":
        date_values = pd.date_range(start=start_date, end=end_date, freq='D')
        col_values = pd.Series(date_values)  # Date values directly
    elif col == "Credit Card Type":
        col_values = pd.Series(np.random.choice(col_range, size=n_rows, p=col_prob))
    elif col_prob is None:
        # Use uniform distribution for other columns
        col_values = pd.Series(np.random.randint(col_range[0], col_range[1] + 1, size=n_rows, dtype="int64"))
    else:
        # Use categorical distribution for other columns
        col_values = pd.Series(np.random.choice(len(col_range), size=n_rows, p=col_prob))
        col_values = col_values.apply(lambda x: col_range[x])  # Map index to values

    # Assign the values to the dataframe
    df[col] = col_values

# Create the "Fraud detected Details" column based on the "fraud" column
fraud_details = []
fraud_percentage_by_year = {2018: 0.10, 2019: 0.22, 2020: 0.28}  # Adjusted for the specified percentages

for idx, row in df.iterrows():
    if row["fraud"] == 0:
        year = row["date"].year
        detection_probability = fraud_percentage_by_year.get(year, 0)
        if np.random.rand() <= detection_probability:
            details = "Attempted Fraud Detected"
        else:
            details = ""
    else:
        details = np.random.choice([
            "An IP address that doesn't match the card address on file",
            'Difference between shipping and billing addresses',
            'Unusual or spammy email accounts',
            'Series of declined transactions in a row',
            'Multiple attempts to incorrectly enter the card number',
            'Same address, different cards'
        ], p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    fraud_details.append(details)

df["Fraud detected Details"] = fraud_details

# Create the "Credit Limit" column based on the specified conditions
credit_limit = []
for idx, row in df.iterrows():
    if row["fraud"] == 0 and row["Fraud detected Details"] == "Attempted Fraud Detected":
        credit_limit.append("$20,000.00")
    elif row["fraud"] == 0 and row["Fraud detected Details"] == "":
        credit_limit.append(np.random.choice(["$10,000.00", "$15,000.00", "$5,000.00"]))
    elif row["fraud"] == 1 and row["Fraud detected Details"] in [
        "An IP address that doesn't match the card address on file",
        'Difference between shipping and billing addresses',
        'Unusual or spammy email accounts',
        'Series of declined transactions in a row',
        'Multiple attempts to incorrectly enter the card number',
        'Same address, different cards'
    ]:
        credit_limit.append(np.random.choice(["$20,000.00", "$30,000.00"]))
    else:
        credit_limit.append("")

df["Credit Limit"] = credit_limit

# Save the dataframe as a CSV file
df.to_csv("debit_card_fraud_detection_dataset.csv", index=False)

# Set the max_rows option to None to display all rows
pd.set_option('display.max_rows', None)

# Print the entire dataframe
df


# In[77]:


df.dtypes


# In[75]:


df.dtypes


# In[79]:


import pandas as pd

url = 'https://raw.githubusercontent.com/OwenDataScience/REAL-TIME-CREDIT-CARD-FRAUD-DETECTION/main/debit_card_fraud_detection_dataset.csv'
ef = pd.read_csv(url)
ef


# In[80]:


ef.info()


# In[81]:


ef.describe()


# In[ ]:





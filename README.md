# Student Loan Repayment Prediction Challenge

---

## Overview

This repository contains the solution to the Student Loan Repayment Prediction Challenge, which focuses on training and evaluating a deep neural network to predict whether a borrower is likely to repay their student loan. The project uses TensorFlow and Keras to build, train, and evaluate a binary classification model.

---

## Key Tasks

The analysis consists of the following major tasks:

1. Data Retrieval:

   - Importing the student loan dataset from a provided online source.
   - Loading the data into a Pandas DataFrame for analysis.

2. Data Cleaning and Preparation:

   - Creating the target labels (y) from the `credit_ranking` column, where 1 represents good credit and 0 represents poor credit.
   - Defining the features (X) from the remaining columns.
   - Reviewing and scaling the feature data using StandardScaler.

3. Data Splitting and Feature Scaling:

   - Splitting the dataset into training and testing sets using a 70/30 ratio.
   - Standardizing the feature data to improve neural network performance.

4. Model Building and Evaluation:

   - Building a deep neural network using Keras with two hidden layers.
   - Training the model using 50 epochs with binary cross-entropy loss and the Adam optimizer.
   - Evaluating the model’s performance on test data.

5. Results Interpretation:

   - Printing loss and accuracy of the trained model.
   - Making predictions and generating a classification report.
   - Discussing ideas for building a student loan recommendation system based on context-based filtering.

---

## How to Run

Clone the Repository:  
`git clone https://github.com/CCIEMikeG/neural-network-challenge-1.git`  
`cd neural-network-challenge-1`

1. Set Up the Environment:

   - Ensure Python 3.x is installed.
   - Install required libraries:  
     `pip install pandas scikit-learn tensorflow`

2. Run the Notebook:

   - Open the `student_loans_with_deep_learning.ipynb` notebook using Jupyter Notebook or VS Code.
   - Execute each cell sequentially to train the model and analyze the results.

---

## Dependencies:

Ensure the following Python libraries are installed:

- pandas  
- scikit-learn  
- tensorflow

You can install these libraries with:  
`pip install pandas scikit-learn tensorflow`

---

### Results

The analysis successfully:

- Loaded and processed the student loan dataset.
- Standardized features to optimize model performance.
- Built and trained a binary classification model using a deep neural network.
- Evaluated model accuracy:
  - Final Model Accuracy: ~73%
- Generated a classification report for precision, recall, and f1-score.
- Proposed a context-based filtering strategy for a future student loan recommendation system.

---

## License

This project is licensed under the MIT License.

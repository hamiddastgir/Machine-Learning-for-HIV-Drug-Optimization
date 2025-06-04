# Optimizing HIV Treatment Strategies Using Machine Learning

This repository contains code and resources for analyzing and optimizing HIV treatment strategies using machine learning techniques, specifically Logistic Regression and Q-Learning. The project involves preparing and cleaning the dataset, performing logistic regression analysis to classify patients based on treatment regimens, and applying Q-Learning to optimize treatment policies for reducing viral load.

## Contents

	•	data_cleaning.ipynb: Data preparation and logistic regression analysis.
	•	q_learning.ipynb: Implementation of the Q-Learning algorithm.
	•	mgddll6.xlsx: Original dataset (ensure this file is placed correctly for the notebooks to access).
	•	merged.csv: Processed dataset ready for logistic regression.
	•	q_table_with_og_states.csv: Output Q-table from the Q-Learning process.

## Project Overview

## Data Preparation and Logistic Regression (data_cleaning.ipynb)

### Purpose

	•	Data Cleaning: Prepare the dataset by filtering and restructuring data relevant to the analysis.
	•	Feature Engineering: Calculate average biomarker values per patient.
	•	Classification: Assign patients to classes based on dominant treatment regimens.
	•	Logistic Regression: Use the prepared data to build a logistic regression model for predicting patient classes.

### Key Steps

1. Data Loading and Cleaning
	•	Load the dataset from mgddll6.xlsx.
	•	Drop unnecessary columns (e.g., ‘Unnamed: 0’).
	•	Filter out visits without NRTI treatment since they constitute only about 3% of the data.
2. Creating Treatment Categories
	•	Define categories based on combinations of NNRTI and PI treatments:
	•	Neither NNRTI nor PI: Patients not on these treatments.
	•	NNRTI Only: Patients on NNRTI but not PI.
	•	PI Only: Patients on PI but not NNRTI.
	•	Both NNRTI and PI: Patients on both treatments.
	•	Count the number of patients in each category for analysis.
3. Biomarker Averaging
	•	Select relevant biomarkers for analysis.
	•	Calculate the mean of these biomarkers for each patient (CASEID).
4. Class Assignment
	•	Create binary classes:
	•	Class 1: Patients primarily on NNRTI only.
	•	Class 0: Patients primarily on PI only.
	•	Assign each patient to a dominant class based on the sum of their treatment occurrences.
	•	Exclude patients who do not fit clearly into either class.
5. Data Merging
	•	Merge the averaged biomarker data with the class assignments.
	•	The resulting dataset (merged.csv) is ready for logistic regression analysis.

### Next Steps

#### Logistic Regression Model
	•	Use merged.csv to train a logistic regression model.
	•	Predict patient classes based on their averaged biomarkers.
	•	Evaluate the model’s performance and adjust as necessary.

 ## Q-Learning Implementation (q_learning.ipynb)

 ### Purpose

 	•	Reinforcement Learning: Apply Q-Learning to determine optimal treatment strategies.
	•	Policy Optimization: Learn policies that minimize the viral load over time.

### Key Steps

1. Data Loading and Preprocessing
	•	Load the dataset from mgddll6.xlsx.
	•	Drop unnecessary columns.
	•	Normalize clinical metrics to create the state space.
2. Defining State and Action Spaces
	•	State Variables: Select clinical metrics (HGB_LC, MCV_LC, PLATLC, WBC_LC, HSRAT) as state features.
	•	Action Variables: Define treatment combinations using NRTI, NNRTI, PI, and OTHER.
	•	Generate all possible action combinations and create mappings between actions and indices.
3. State Discretization
	•	Discretize the normalized state variables into bins to handle continuous data in the Q-Learning algorithm.
4. Initializing the Q-Table
	•	Initialize a Q-table with dimensions corresponding to the number of unique states and actions.
5. Setting Q-Learning Parameters
	•	Define learning rate (alpha), discount factor (gamma), and exploration rate (epsilon).
6. Implementing the Q-Learning Algorithm
	•	Iterate over multiple episodes.
	•	For each step:
	•	Choose an action based on the exploration-exploitation trade-off.
	•	Calculate the reward based on the change in viral load.
	•	Update the Q-table using the Bellman equation.
7. Saving Results
	•	Save the final Q-table along with the corresponding states to q_table_with_og_states.csv for analysis.

### Outcomes

**Optimal Policy Identification**
	•	The Q-table contains Q-values indicating the expected utility of taking certain actions in given states.
	•	Analyze the Q-table to derive policies that optimize treatment strategies for patients.


 ## Requirements

	•	Python 3.x
	•	Libraries:
	•	pandas
	•	numpy
	•	scikit-learn
	•	itertools
	•	random
	•	openpyxl (for reading Excel files)


## Contact

For questions or further information, please contact:

Muhammad Hamid Ahmed Dastgir
Email: hamiddastgirwork@gmail.com
LinkedIn: linkedin.com/in/hamiddastgir

## Acknowlegements

**Supervisor:** Dr. Choudur Lakshminarayan
**Institution:** Stevens Institute of Technology, School of Business



 

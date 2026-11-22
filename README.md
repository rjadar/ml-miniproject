# Application of Machine Learning Methods to Evaluate Collectibles

A machine learning project exploring the classification of Funko Pop collectibles into price categories using product characteristics and search volume.

## Dataset
- 359 Funko Pop collectibles
- Five price categories derived from average sale price
- Features include search volume, category, material, and special version

## Models
- Linear SVM (One-vs-Rest)
- Linear SVM (One-vs-One)
- RBF SVM
- Custom Kernel SVM
- Decision Tree
- Bagging Classifier
- Class-Balanced Linear SVM

## Methodology
- Data preprocessing and feature encoding
- Model comparison
- Nested 4-fold / 3-fold cross-validation
- Hyperparameter tuning
- Evaluation using weighted precision, recall, and F1-score

## Setup

\\\ash
pip install -r requirements.txt
\\\

## Run

\\\ash
python main.py
\\\

Run the complete experiment:

\\\ash
python main.py --full
\\\

## Report

See [ml_mini_writeup.pdf](ml_mini_writeup.pdf) for the project summary.

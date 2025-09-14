SVM & Deep Learning Experiments
This repository contains experiments and code samples for comparing Support Vector Machine (SVM) models and deep learning approaches on high-dimensional datasets.

⚠️ Note
SVM is not recommended for this project.

Due to the large size and high dimensionality of the data, SVM does not perform optimally. Please use more suitable algorithms such as XGBoost for machine learning tasks on this dataset.

Repository Contents
svm/ — SVM experiments and scripts (for reference only)
deep_learning/ — Deep learning models, training scripts, and utilities
data/ — Data samples and preprocessing scripts
xgboost/ — Recommended classification models and examples
notebooks/ — Jupyter notebooks for analysis and visualization
Getting Started
Clone this repository
git clone https://github.com/JACKisREAL/svm-deeplearning-testing.git
Install dependencies
pip install -r requirements.txt
Run experiments
See notebooks/ or the scripts in xgboost/ for recommended workflows.
Recommended Approach
For large, high-dimensional data, use the code in the xgboost/ directory. XGBoost generally provides better performance, scalability, and training speed in these scenarios compared to traditional SVM.

License
This project is provided for research and educational purposes. Please see the LICENSE file for details.

Author
JACKisREAL

# Breast Cancer Detection

A Machine Learning project to predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous) using the scikit-learn breast cancer dataset.

## Project Overview
This project uses Logistic Regression to classify breast tumors based on features from the scikit-learn dataset. It includes Jupyter notebooks for data analysis and model training, and Streamlit apps for interactive predictions.

## Folder Structure
- **notebooks/**: Jupyter notebooks for data preprocessing, feature selection, and model training.
  - `AI_model.ipynb`: Main notebook with feature selection and high-accuracy model (~98%).
  - `feature_selection.ipynb`: Feature analysis and selection.
  - `Logistic.ipynb` & `logistic-1.ipynb`: Logistic Regression model training.
  - `LR_usage.ipynb`: Demonstrates model loading and prediction.
  - `cancer_data_linear_regression.ipynb`: Experimental (uses linear regression, not ideal for classification).
- **app/**: Streamlit apps for user-friendly predictions.
  - `cancer.py`: Full app with 30 input features.
  - `cancer_updated.py`: Simplified app using 5 key features.
- **models/**: Pre-trained models.
  - `breast_cancer_model.joblib`: Model for `cancer.py`.
  - `Cancer_model_AI.joblib`: Model for `cancer_updated.py`.
- **requirements.txt**: Lists required Python libraries.

## How to Run
1. Clone the repository: `git clone https://github.com/YOUR_USERNAME/breast-cancer-detection.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run a notebook: Open `notebooks/AI_model.ipynb` in Jupyter Notebook to explore or retrain the model.
4. Run the app: Use `streamlit run app/cancer_updated.py` to launch the simplified prediction app.

## Results
- **Model Performance** (from `AI_model.ipynb`):
  - Accuracy: 96.8%
  - Precision: 97.5%
  - Recall: 97.5%
  - F1-Score: 97.5%
- **Key Features**: Mean concave points, worst radius, worst perimeter, worst area, worst concave points (selected using mutual information).

## Dataset
The project uses the breast cancer dataset from scikit-learn (`load_breast_cancer`), which includes 30 features and 569 samples.

## Notes
- The `cancer_data_linear_regression.ipynb` notebook uses linear regression, which is not suitable for this classification task (low R² score). It’s included for experimental purposes but should be updated to use Logistic Regression.
- The `cancer_updated.py` app is recommended for quick predictions due to its simplified input requirements.

## Future Improvements
- Add more model evaluation metrics (e.g., ROC curve).
- Deploy the Streamlit app to Streamlit Community Cloud.
- Include visualizations of feature importance.

# Enhanced Depression Prediction Project

This project focuses on building an **ideal, unbiased depression prediction model** using synthetic data.  
It includes both a **Jupyter Notebook** for model training and analysis, and a **Streamlit App** for interactive predictions.

---

## 🧠 Project Overview
This is the continuation of the previous depression model, which was trained on noisy data.  
The current version uses a **synthetic, ideal, and unbiased dataset** to achieve clean experimentation and high model performance.  
The app allows users to provide inputs for key psychological and behavioral parameters to predict depression likelihood.

---

## ⚙️ Features
- Trained using **XGBoost Classifier** for accurate binary classification  
- **7 key input features:**  
  - sadness_score  
  - anxiety_score  
  - fatigue_score  
  - sleep_issues  
  - social_withdrawal  
  - concentration_issues  
  - mood_swings  
- **TF-IDF vectorizer** for handling text input (used only during training)  
- Visualizations in notebook and app for better model understanding  
- Interactive and user-friendly **Streamlit interface**

---

## 📊 Dataset
The project uses a **synthetic, ideal, and unbiased dataset** with:  
- 10,000 samples  
- 7 numeric features (scores and boolean symptoms)  
- Optional text features used during training  
- Target variable: `depression` (0 = Not Depressed, 1 = Depressed)

This dataset ensures unbiased learning and allows testing of prediction logic in a clean environment.

---

## 💾 Saved Models
- **`depression_xgb_model.pkl`** → XGBoost model trained on synthetic data  
- **`tfidf_vectorizer.pkl`** → TF-IDF vectorizer for text preprocessing (used only in training)

Both models are loaded in the app for inference and experimentation.

---

## 📈 Visualizations
The notebook and app include visual insights such as:
- Correlation heatmaps  
- Feature importance graphs  
- Distribution plots for numeric features  
- Prediction summary and accuracy metrics  

---

## 🧩 Streamlit App
The **Enhanced Depression Prediction App** lets users input their mental health parameters through sliders and checkboxes.  
It then predicts the likelihood of depression using the trained model and displays relevant charts.  
User text interpretation has been **removed** to focus purely on numeric and model-based predictions.

---

## 🧰 Tech Stack
- Python  
- XGBoost  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Streamlit  
- Joblib  

---

## 🧪 Model Accuracy
The model achieves **100% accuracy** on the synthetic dataset, since it’s an idealized, bias-free dataset designed for controlled testing — not for real-world deployment.

---

## 📷 Screenshot
This screenshot shows the Enhanced Depression Prediction App interface, where users can provide numeric inputs and visualize predictions and graphs interactively.

---

## 🚀 Usage
1. Clone the repository  
2. Create a virtual environment  
3. Install dependencies using `pip install -r requirements.txt`  
4. Run the notebook for training (`.ipynb` file)  
5. Launch the Streamlit app using:  
   ```bash
   streamlit run app.py

# 🧠 Advanced Depression Predictor Model

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-red)
![License](https://img.shields.io/badge/License-Educational-yellow)

*A machine learning project for depression prediction using synthetic data and XGBoost classification*

[🚀 Features](#-features) • [📊 Dataset](#-dataset) • [⚙️ Installation](#️-installation) • [🎯 Usage](#-usage) • [📈 Model Performance](#-model-performance)

</div>

---

## 📖 Overview

This project represents an **advanced, unbiased depression prediction system** built as a continuation of a previous model trained on noisy data. The current implementation leverages a **synthetic, ideal, and unbiased dataset** to ensure clean experimentation and demonstrate high-performance machine learning capabilities.

### 🎯 Objectives
- Build a robust binary classification model for depression prediction
- Demonstrate clean ML experimentation using synthetic, bias-free data
- Provide an interactive interface for real-time mental health predictions
- Showcase feature importance and model interpretability

> ⚠️ **Disclaimer**: This project is for **experimental and educational purposes only**. It does not replace professional medical diagnosis or clinical assessment.

---

## ✨ Features

### 🔬 Machine Learning
- **XGBoost Classifier** for high-performance binary classification
- **TF-IDF Vectorizer** for text preprocessing during training
- **7 key psychological and behavioral features**:
  - `sadness_score` - Intensity of sadness feelings
  - `anxiety_score` - Level of anxiety experienced
  - `fatigue_score` - Physical and mental exhaustion
  - `sleep_issues` - Sleep quality and disturbances
  - `social_withdrawal` - Social isolation tendencies
  - `concentration_issues` - Difficulty focusing
  - `mood_swings` - Emotional volatility

### 📊 Visualization & Analysis
- Correlation heatmaps for feature relationships
- Feature importance graphs
- Distribution plots for numeric features
- Prediction confidence metrics
- Interactive data exploration

### 💻 Interactive Application
- **Streamlit-powered** web interface
- Real-time predictions with slider inputs
- Visual feedback and charts
- User-friendly design with immediate results

---

## 📊 Dataset

### Characteristics
- **Size**: 10,000 samples
- **Features**: 7 numeric features (continuous scores and boolean indicators)
- **Target**: Binary classification (`0` = Not Depressed, `1` = Depressed)
- **Quality**: Synthetic, ideal, and unbiased for controlled testing
- **Format**: CSV file (`depression_dataset_10000.csv`)

### Feature Distribution
All features are carefully balanced to ensure unbiased learning and testing of prediction logic in a controlled environment.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Core Language** | Python 3.8+ |
| **ML Framework** | XGBoost, Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Web Interface** | Streamlit |
| **Model Persistence** | Joblib |

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/willow788/Advanced-depression-predictor-model.git
   cd Advanced-depression-predictor-model
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` doesn't exist, install manually:
   ```bash
   pip install xgboost pandas numpy matplotlib seaborn scikit-learn streamlit joblib jupyter
   ```

---

## 🎯 Usage

### 🔧 Model Training

To retrain the model or explore the training process:

```bash
jupyter notebook prediction.ipynb
```

The notebook includes:
- Data loading and exploration
- Feature engineering
- Model training with XGBoost
- Performance evaluation
- Visualization of results

### 🚀 Running the Streamlit App

Launch the interactive web application:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### 🎮 Using the Application

1. **Input Parameters**:  Use sliders and checkboxes to set mental health indicators
2. **Generate Prediction**: Click to analyze depression likelihood
3. **View Results**: See prediction, confidence score, and visualizations
4. **Explore Charts**: Analyze feature importance and distributions

---

## 📈 Model Performance

### Accuracy Metrics
- **Training Accuracy**: 100%
- **Test Accuracy**: 100%

> 📌 **Note**: Perfect accuracy is achieved due to the synthetic, idealized nature of the dataset.  This represents a controlled testing environment, **not real-world deployment readiness**.

### Model Artifacts

| File | Description | Size |
|------|-------------|------|
| `depression_xgb_model.pkl` | Trained XGBoost classifier | ~218 KB |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer for text features | ~3.6 KB |

Both models are serialized using `joblib` and loaded during inference.

---

## 📁 Project Structure

```
Advanced-depression-predictor-model/
│
├── app.py                          # Streamlit application
├── prediction. ipynb                # Training notebook
├── depression_dataset_10000.csv    # Synthetic dataset
├── depression_xgb_model.pkl        # Trained model
├── tfidf_vectorizer.pkl            # TF-IDF vectorizer
├── Screenshot 2025-10-15 022315.png # App interface demo
└── README.md                       # Documentation
```

---

## 🖼️ Screenshots

### Application Interface
![App Interface](Screenshot%202025-10-15%20022315.png)

*The Enhanced Depression Prediction App allows users to input psychological parameters through an intuitive interface and receive instant predictions with visual analytics.*

---

## 🔬 Methodology

### Training Pipeline
1. **Data Generation**:  Synthetic dataset creation with controlled distributions
2. **Feature Engineering**:  Numerical and text feature processing
3. **Model Selection**: XGBoost chosen for gradient boosting capabilities
4. **Training**:  Supervised learning with binary classification
5. **Evaluation**:  Performance metrics and visualization
6. **Deployment**: Model serialization and Streamlit integration

### Key Design Decisions
- **XGBoost**: Selected for handling non-linear relationships and feature importance
- **Synthetic Data**: Ensures bias-free testing environment
- **Interactive UI**: Streamlit provides rapid prototyping and user engagement

---

## 🚧 Future Enhancements

- [ ] Add SHAP values for model explainability
- [ ] Implement ensemble methods (Random Forest, Neural Networks)
- [ ] Create API endpoint for programmatic access
- [ ] Add unit tests and CI/CD pipeline
- [ ] Deploy to cloud platform (Streamlit Cloud, Heroku)
- [ ] Include multi-class severity classification
- [ ] Add data validation and input sanitization

---

## 📝 Important Notes

### ⚠️ Limitations
- **Synthetic Data**: Results do not represent real-world clinical accuracy
- **Educational Purpose**: Not validated for medical or diagnostic use
- **Bias Considerations**: Real-world depression involves complex factors not captured here
- **No Clinical Validation**: Not approved by medical authorities

### 🔒 Ethical Considerations
- This tool should **never** replace professional mental health assessment
- Predictions are based on simplified synthetic patterns
- Always consult qualified healthcare professionals for mental health concerns

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

---

## 📄 License

This project is available for **educational and research purposes**. Please use responsibly and ethically.

---

## 👤 Author

**willow788**
- GitHub: [@willow788](https://github.com/willow788)

---

## 🙏 Acknowledgments

- XGBoost community for the powerful ML framework
- Streamlit team for the intuitive app framework
- Open-source contributors in the ML/mental health space

---

## 📞 Contact & Support

If you have questions or feedback: 
- Open an issue on GitHub
- Star ⭐ the repository if you found it helpful! 

---

<div align="center">

**Built with ❤️ for machine learning education**

</div>

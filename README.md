# Diabetes Prediction System

![Diabetes Prediction System](https://img.shields.io/badge/Version-1.0.0-blue)
![Flask](https://img.shields.io/badge/Backend-Flask-green)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **A Production-Ready Web Application for Diabetes Risk Assessment using Machine Learning**

## 📋 Project Overview

The Diabetes Prediction System is a comprehensive web application that uses Machine Learning (Random Forest algorithm) to predict diabetes risk based on medical parameters. This system provides a user-friendly interface for healthcare professionals and individuals to assess potential diabetes risk quickly and accurately.

## ✨ Features

- **🎯 Accurate Predictions**: 72% accuracy using Random Forest classifier
- **⚡ Real-Time Analysis**: Instant prediction results
- **📱 Fully Responsive**: Works seamlessly on desktop, tablet, and mobile
- **🔒 Secure & Private**: No permanent data storage
- **🎨 Modern UI**: Clean, professional interface with intuitive navigation
- **📊 Educational Content**: Detailed information about diabetes and risk factors
- **🔧 Easy Setup**: Simple installation and deployment process

## 🏗️ Technology Stack

### Backend
- **Python 3.8+** - Core programming language
- **Flask 3.1.2** - Web framework
- **Scikit-learn 1.8.0** - Machine learning library
- **Pandas 2.3.3** - Data manipulation
- **NumPy 2.4.1** - Numerical computations
- **Joblib 1.5.3** - Model serialization

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with CSS Grid & Flexbox
- **JavaScript** - Client-side interactivity
- **Font Awesome** - Icon library
- **Google Fonts (Inter & Poppins)** - Typography

### Machine Learning
- **Algorithm**: Random Forest Classifier
- **Dataset**: Pima Indians Diabetes Dataset
- **Accuracy**: 72% (on test set)
- **Features**: 8 medical parameters

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- Git (optional)

### Step 1: Clone or Download the Project
```bash
# Clone the repository
git clone <your-repository-url>
cd diabetes_prediction

# Or download and extract the ZIP file
```

### Step 2: Set Up Virtual Environment (Windows)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### Step 3: Install Dependencies
Create a `requirements.txt` file with:
```txt
flask==3.1.2
pandas==2.3.3
numpy==2.4.1
scikit-learn==1.8.0
joblib==1.5.3
```

Install using pip:
```bash
pip install -r requirements.txt
```

### Step 4: Train the Machine Learning Model
```bash
python model_train.py
```
**Expected Output:**
```
Training started...
Dataset loaded
Model Accuracy: 0.7207792207792207
Model saved successfully
Training completed.
```

### Step 5: Run the Application
```bash
python app.py
```
**Expected Output:**
```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Debugger is active!
 * Debugger PIN: 799-966-041
```

### Step 6: Access the Application
Open your web browser and navigate to:
```
http://localhost:5000
```

## 📁 Project Structure
```
diabetes_prediction/
│
├── app.py                    # Main Flask application
├── model_train.py           # Machine learning model training script
├── model.pkl                # Trained Random Forest model
├── diabetes.csv             # Diabetes dataset
├── requirements.txt         # Python dependencies
│
├── templates/               # HTML templates
│   ├── base.html           # Base template with header/footer
│   ├── home.html           # Home page
│   ├── index.html          # Prediction form
│   └── result.html         # Results page
│
└── static/                  # Static files
    ├── style.css           # Main stylesheet
    └── script.js           # JavaScript file
```

## 🎯 How to Use the Application

### 1. Home Page (`/`)
- Overview of the system
- Features and benefits
- Navigation to prediction tool

### 2. Prediction Form (`/predict-form`)
- Enter 8 medical parameters:
  1. Pregnancies
  2. Glucose Level
  3. Blood Pressure
  4. Skin Thickness
  5. Insulin Level
  6. Body Mass Index (BMI)
  7. Diabetes Pedigree Function
  8. Age

### 3. Results Page (`/predict` - POST request)
- Immediate prediction result
- Clear indication: "Diabetic" or "Not Diabetic"
- Option to make new prediction

## 🔧 Model Training Details

### Training Script (`model_train.py`)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save model
joblib.dump(model, "model.pkl")
```

### Model Performance
- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Test Size**: 20%
- **Random State**: 42 (for reproducibility)
- **Accuracy**: 72.08%

### Dataset Information
The model is trained on the **Pima Indians Diabetes Database**:
- **Samples**: 768
- **Features**: 8
- **Target**: Outcome (1 = Diabetic, 0 = Not Diabetic)

## 🌐 Application Routes

| Route | Method | Description | Template |
|-------|--------|-------------|----------|
| `/` | GET | Home page | `home.html` |
| `/predict-form` | GET | Prediction form | `index.html` |
| `/predict` | POST | Process prediction | `result.html` |
| `/predict` | GET | Redirect to form | `index.html` |

## ⚠️ Common Issues & Solutions

### Issue 1: `sklearn.utils.validation` Warning
```
UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
```
**Solution**: This is a harmless warning. The model works correctly despite the warning. You can suppress it if needed:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
```

### Issue 2: Port Already in Use
```
OSError: [Errno 98] Address already in use
```
**Solution**: Change the port or kill the existing process:
```bash
# Change port in app.py
app.run(debug=True, port=5001)

# Or kill process on port 5000 (Linux/Mac)
lsof -ti:5000 | xargs kill -9
```

### Issue 3: Missing Dependencies
```
ModuleNotFoundError: No module named 'flask'
```
**Solution**: Ensure virtual environment is activated and install dependencies:
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Deployment Options

### Option 1: Local Development (Current)
```bash
python app.py
```

### Option 2: Production with Gunicorn (Linux/Mac)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Option 3: Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t diabetes-prediction .
docker run -p 5000:5000 diabetes-prediction
```

## 📊 Medical Parameters Reference

| Parameter | Normal Range | Importance |
|-----------|--------------|------------|
| Glucose | 70-100 mg/dL | Primary indicator |
| Blood Pressure | <120/80 mmHg | Cardiovascular health |
| BMI | 18.5-24.9 | Weight status |
| Age | 21-81 years | Risk factor |
| Pregnancies | 0-20 | Gestational diabetes risk |
| Insulin | 16-166 μU/mL | Insulin resistance |
| Skin Thickness | 10-40 mm | Adipose tissue indicator |
| Diabetes Pedigree | 0.08-2.42 | Genetic predisposition |

## ⚠️ Important Disclaimer

**MEDICAL DISCLAIMER**: This application is for **EDUCATIONAL AND INFORMATIONAL PURPOSES ONLY**. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.

**Important Notes:**
1. The model has 72% accuracy - there's a 28% chance of incorrect prediction
2. Always consult with healthcare professionals for medical decisions
3. This tool should be used as a preliminary assessment only
4. Real medical diagnosis requires comprehensive testing and professional evaluation

## 🔒 Security & Privacy

- **No Data Storage**: User inputs are processed in real-time and not stored permanently
- **Session-based**: Each prediction is independent and doesn't retain user data
- **Secure by Design**: No personal identifiable information is collected
- **Transparent**: Open-source code allows for security review

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs**: Create an issue with detailed description
2. **Suggest Features**: Propose new features or improvements
3. **Code Contributions**: Fork the repository and submit a pull request
4. **Improve Documentation**: Help enhance this README or add tutorials

### Development Setup
```bash
# 1. Fork and clone the repository
git clone https://github.com/your-username/diabetes_prediction.git
cd diabetes_prediction

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train model
python model_train.py

# 5. Run development server
python app.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: Pima Indians Diabetes Database from UCI Machine Learning Repository
- **Libraries**: Flask, Scikit-learn, Pandas, NumPy communities
- **Icons**: Font Awesome
- **Fonts**: Google Fonts (Inter & Poppins)

## 📞 Support

For support or questions:
1. Check the [Common Issues](#common-issues--solutions) section
2. Review the application logs for error messages
3. Ensure all dependencies are installed correctly
4. Verify the model file (`model.pkl`) exists

## 📈 Future Enhancements

Planned features for future versions:
1. **Higher Accuracy Models**: Implement gradient boosting or neural networks
2. **User Accounts**: Save prediction history (with consent)
3. **Advanced Analytics**: Visualize risk factors and trends
4. **Multi-language Support**: Reach non-English speaking users
5. **Mobile App**: Native iOS and Android applications
6. **API Access**: RESTful API for integration with other systems
7. **Export Reports**: Generate PDF reports of predictions

## 🎓 Educational Value

This project serves as an excellent educational resource for:
- **Machine Learning Students**: Real-world ML implementation
- **Web Developers**: Full-stack Flask application
- **Healthcare Students**: Application of AI in medicine
- **Data Science Enthusiasts**: End-to-end data science project

---

<div align="center">

## 🏆 Successfully Deployed by Many Users

**Marjan** - *"Successfully trained model with 72% accuracy and deployed the application!"*

</div>

---

<div align="center">
  
**Made with ❤️ for Healthcare Education**

*Empowering early detection through technology*

📧 **Contact**: For questions or support, create an issue on GitHub

⭐ **If you find this project useful, please give it a star!**

</div>

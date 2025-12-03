# ML Web App v1 — House Prices Prediction

## Project Overview

This project is part of my AI & Robotics Engineer portfolio.  
It demonstrates the end-to-end workflow of building and deploying a **machine learning web application** using **Scikit-learn** and **Streamlit**, deployed on **Streamlit Cloud**.

The app predicts **house prices** based on user inputs, showcasing skills in:

- Data preprocessing
- Model training and evaluation
- Web app development with Streamlit
- Cloud deployment
- Iterative project scaling across phases

---

## Project Roadmap (Phases)

### **Phase 1 — MVP**

- Baseline regression model (Linear Regression).
- Minimal Streamlit app with input form + prediction output.
- Deployed on Streamlit Cloud.

### **Phase 2 — Enhanced Modeling**

- Add advanced models (Random Forest, Gradient Boosting).
- Compare performance metrics (MAE, RMSE, R²).
- Save and load best-performing model.

### **Phase 3 — UI & Visualization**

- Improve Streamlit interface with clean layout.
- Add dataset preview and charts (feature distributions).
- Feature importance visualization.

### **Phase 4 — Deployment & Scaling**

- Optimize app performance.
- Add error handling for user inputs.
- Ensure reproducibility with `requirements.txt`.
- Document deployment steps clearly.

### **Phase 5 — Portfolio Polish**

- Add screenshots of the app.
- Write reflection on challenges and lessons learned.
- Share project link and insights on LinkedIn.

---

## Tech Stack

- **Python**
- **Scikit-learn**
- **Streamlit**
- **Joblib** (model persistence)

---

## Project Structure

```
ml_web_app_v1/
│── data/              # dataset (raw/cleaned)
│── notebooks/         # exploration & preprocessing
│── models/            # saved joblib models
│── app.py             # Streamlit app
│── requirements.txt   # dependencies
│── README.md          # project documentation
```

---

## Deployment

- Hosted on **Streamlit Cloud** (free tier).
- Public link included in README once deployed.

---

## Future Directions

- Expand to multiple datasets (Titanic classification in v2).
- Integrate more interactive visualizations.
- Explore containerization (Docker) for flexible deployment.
- Add CI/CD for automated testing and deployment.

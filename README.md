# 📊 Advanced Banking Customer Analytics

This project performs an end-to-end analysis of banking customer data using clustering techniques, churn prediction, and cross-selling recommendation systems. It also includes a Gradio-powered web app for interactive predictions.

---

## 🚀 Features

* 🔍 Exploratory Data Analysis (EDA)
* 🔗 K-Modes, K-Means, K-Prototypes & Agglomerative Clustering
* 📉 Elbow Method for optimal cluster selection
* 🧪 PCA for dimensionality reduction and visualization
* 🌲 Random Forest for churn prediction
* 🤝 Rule-based product recommendation engine
* 🖼️ Data visualization with saved plots
* 🌐 Gradio web app for real-time predictions and recommendations

---

## 🧠 ML Techniques Used

* **Clustering:**

  * K-Modes (categorical)
  * K-Means (numerical)
  * K-Prototypes (mixed)
  * Agglomerative (hierarchical)
* **Dimensionality Reduction:**

  * PCA (Principal Component Analysis)
* **Modeling:**

  * Random Forest Regressor
  * Random Forest Classifier

---

## 📁 File Structure

```
├── aml_project_.py                   # Main pipeline script
├── app.py                            # Gradio web app
├── model.pkl                         # Trained model file
├── banking_data.xlsx                 # Input data (not included here)
├── stay_likelihood.csv               # Output stay probability
├── cross_sell_recommendations.csv    # Output product suggestions
├── *.png                             # Generated clustering and analysis plots
```

---

## 📊 Sample Visualizations

* gender\_occupation\_clustering.png
* kmeans\_elbow\_method.png
* pca\_clustering.png
* agglomerative\_clustering.png
* pca\_churn\_probability.png

---

## 🌐 Gradio Web App

**Title:** *Customer Retention & Cross-Sell Recommendation*

**Inputs:**

* Customer ID
* Satisfaction Score
* NPS Score
* Product Holdings (Savings, Credit Card, Loan, Investment)

**Outputs:**

* 📈 Stay Likelihood (%)
* 🎯 Product Recommendation

To run:

```bash
python app.py
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/banking-customer-analytics.git
cd banking-customer-analytics
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Place the dataset**
   Ensure `banking_data.xlsx` is in the root directory.

4. **Run the project**

```bash
python aml_project_.py
```

5. **Launch the Web App**

```bash
python app.py
```

---

## 📦 Dependencies

Create a `requirements.txt` using:

```bash
pip freeze > requirements.txt
```

Or use these manually:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
kmodes
openpyxl
gradio
```

---

## 🔮 Future Scope

* Add SHAP or LIME for model explainability
* Migrate to Streamlit or cloud deployment
* Integrate with customer CRM systems
* Use LLM-based recommendation reasoning

---

## 🡩‍💻 Author

**Chirag Patankar**
*B.E. in Artificial Intelligence & Data Science*
📧 [Email](mailto:officialchiragp1605@gmail.com) 

---

## 📃 License

This project is for educational and demonstration purposes only. Modify and adapt freely.

---

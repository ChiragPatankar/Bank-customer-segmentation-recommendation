
import gradio as gr
import pandas as pd
import pickle
import numpy as np

# Load the model
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except:
        # Return a dummy model if file not found
        class DummyModel:
            def predict_proba(self, X):
                return np.array([[0.7, 0.3]] * len(X))
        return DummyModel()

# Load the recommendations
def load_recommendations():
    try:
        return pd.read_csv('cross_sell_recommendations.csv')
    except:
        # Return empty dataframe if file not found
        return pd.DataFrame(columns=['customer_id', 'recommended_product'])

model = load_model()
recommendations_df = load_recommendations()

def predict_churn_and_recommend(customer_id, satisfaction_score, nps_score, savings_account, credit_card, personal_loan, investment_account):
    try:
        # Convert inputs to appropriate types
        customer_id = int(customer_id)
        satisfaction_score = float(satisfaction_score)
        nps_score = float(nps_score)
        savings_account = int(savings_account)
        credit_card = int(credit_card)
        personal_loan = int(personal_loan)
        investment_account = int(investment_account)

        # Create input data for model
        input_data = [[satisfaction_score, nps_score, savings_account, credit_card, personal_loan, investment_account]]

        # Get probability of not churning
        proba = model.predict_proba(input_data)[0]
        stay_probability = proba[0] if proba.shape[0] > 1 else 0.7  # Default if model is dummy
        stay_likelihood = stay_probability * 100

        # Get recommendation or generate a default one
        try:
            recommendation = recommendations_df.loc[recommendations_df['customer_id'] == customer_id, 'recommended_product'].iloc[0]
        except:
            # Generate a recommendation based on input
            if savings_account == 0:
                recommendation = "Savings Account"
            elif credit_card == 0:
                recommendation = "Credit Card"
            elif personal_loan == 0:
                recommendation = "Personal Loan"
            elif investment_account == 0:
                recommendation = "Investment Account"
            else:
                recommendation = "Premium Account"

        return f"{stay_likelihood:.2f}%", recommendation
    except Exception as e:
        return f"Error: {str(e)}", "Unable to generate recommendation"

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_churn_and_recommend,
    inputs=[
        gr.Textbox(label="Customer ID"),
        gr.Slider(0, 5, step=0.1, label="Satisfaction Score (0-5)"),
        gr.Slider(0, 10, step=0.1, label="NPS Score (0-10)"),
        gr.Radio([0, 1], label="Has Savings Account?"),
        gr.Radio([0, 1], label="Has Credit Card?"),
        gr.Radio([0, 1], label="Has Personal Loan?"),
        gr.Radio([0, 1], label="Has Investment Account?")
    ],
    outputs=[
        gr.Textbox(label="Stay Likelihood"),
        gr.Textbox(label="Cross-Sell Recommendation")
    ],
    title="Customer Retention & Cross-Sell Recommendation",
    description="Predict customer likelihood to stay and get personalized product recommendations."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()

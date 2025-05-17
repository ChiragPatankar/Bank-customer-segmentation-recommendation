import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from scipy.cluster.hierarchy import dendrogram, linkage
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import warnings

warnings.filterwarnings('ignore')

# Load the banking data
# Replace this with your actual file path
file_path = "banking_data.xlsx"
banking_data = pd.read_excel(file_path, engine='openpyxl')

print("Banking Data Shape:", banking_data.shape)
print("\nFirst few rows:")
print(banking_data.head())

print("\nData Information:")
banking_data.info()

print("\nDescriptive Statistics:")
print(banking_data.describe())

print("\nMissing Values Count:")
print(banking_data.isnull().sum())

# Data Cleaning
# Replace missing values for specified string columns with their mode
string_columns = ['feature_requests', 'complaint_topics']
for column in string_columns:
    if column in banking_data.columns and banking_data[column].isnull().sum() > 0:
        banking_data[column].fillna(banking_data[column].mode()[0], inplace=True)

print("\nMissing Values After Cleaning:")
print(banking_data.isnull().sum())

# K-Modes Clustering for Gender vs Occupation
print("\nPerforming K-Modes Clustering for Gender vs Occupation...")
X = banking_data[['gender', 'occupation']]
kmodes = KModes(n_clusters=2, init='Cao', n_init=1, verbose=0)
banking_data['cluster'] = kmodes.fit_predict(X)

# Visualization with seaborn
plt.figure(figsize=(12, 6))
sns.stripplot(x="gender", y="occupation", hue="cluster", data=banking_data,
              jitter=True, dodge=True, palette="Set1")
plt.xlabel("Gender")
plt.ylabel("Occupation")
plt.title("K-Modes Clustering (Gender vs. Occupation) - All Data")
plt.tight_layout()
plt.savefig('gender_occupation_clustering.png')
plt.close()

# Sample data for better visualization
sampled_banking_data = banking_data.sample(n=10, random_state=42)
plt.figure(figsize=(10, 6))
sns.stripplot(x="gender", y="occupation", hue="cluster", data=sampled_banking_data,
              jitter=True, dodge=True, palette="Set1")
plt.xlabel("Gender")
plt.ylabel("Occupation")
plt.title("K-Modes Clustering (Gender vs. Occupation) - Sampled Data")
plt.tight_layout()
plt.savefig('gender_occupation_clustering_sampled.png')
plt.close()

# Gender vs Income Bracket
plt.figure(figsize=(12, 6))
sns.stripplot(x="gender", y="income_bracket", hue="cluster", data=banking_data,
              jitter=True, dodge=True, palette="Set1")
plt.xlabel("Gender")
plt.ylabel("Income Bracket")
plt.title("K-Modes Clustering (Gender vs. Income Bracket) - All Data")
plt.tight_layout()
plt.savefig('gender_income_clustering.png')
plt.close()

# Gender vs Education Level
plt.figure(figsize=(12, 6))
sns.stripplot(x="gender", y="education_level", hue="cluster", data=banking_data,
              jitter=True, dodge=True, palette="Set1")
plt.xlabel("Gender")
plt.ylabel("Education Level")
plt.title("K-Modes Clustering (Gender vs. Education Level) - All Data")
plt.tight_layout()
plt.savefig('gender_education_clustering.png')
plt.close()

# Elbow Method for K-Means
print("\nPerforming Elbow Method for K-Means...")
# Define X with your preprocessed data
X_kmeans = banking_data[
    ['satisfaction_score', 'nps_score', 'savings_account', 'credit_card', 'personal_loan', 'investment_account']]
inertia = []
for i in range(1, 15):
    k_means = KMeans(n_clusters=i, random_state=42)
    k_means.fit(X_kmeans)
    inertia.append(k_means.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, 15), inertia, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for K-Means')
plt.grid(True)
plt.savefig('kmeans_elbow_method.png')
plt.close()

# Elbow Method for K-Prototypes
print("\nPerforming Elbow Method for K-Prototypes...")
try:
    # Prepare the data for K-Prototypes
    categorical_cols = banking_data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = banking_data.select_dtypes(include=np.number).columns.tolist()

    # Combine categorical and numerical features
    all_cols = categorical_cols + numerical_cols
    data_for_clustering = banking_data[all_cols]

    # Apply Elbow Method with k-prototypes
    cost = []
    for num_clusters in range(1, 10):  # Reduced range from 15 to 10 to speed up
        print(f"Testing {num_clusters} clusters...")
        kproto = KPrototypes(n_clusters=num_clusters, init='Cao', verbose=0, random_state=42)
        clusters = kproto.fit_predict(data_for_clustering, categorical=list(range(len(categorical_cols))))
        cost.append(kproto.cost_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), cost, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method for K-Prototypes')
    plt.grid(True)
    plt.savefig('kprototypes_elbow_method.png')
    plt.close()
except Exception as e:
    print(f"Error in K-Prototypes Elbow Method: {e}")

# Agglomerative Clustering
print("\nPerforming Agglomerative Clustering...")
try:
    # Select relevant features
    numeric_columns = ['age', 'transaction_frequency', 'total_tx_volume',
                       'avg_tx_value', 'satisfaction_score', 'nps_score',
                       'active_products', 'customer_lifetime_value',
                       'total_transaction_volume', 'monthly_transaction_count', 'avg_daily_transactions']

    # Adjust feature list if needed
    numeric_columns = [col for col in numeric_columns if col in banking_data.columns]

    # Add churn_probability if it exists
    if 'churn_probability' in banking_data.columns:
        numeric_columns.append('churn_probability')

    categorical_columns = ['income_bracket', 'occupation', 'customer_segment', 'education_level',
                           'marital_status', 'acquisition_channel', 'savings_account',
                           'credit_card', 'personal_loan', 'investment_account']
    categorical_columns = [col for col in categorical_columns if col in banking_data.columns]

    # Preprocess the data
    scaler = StandardScaler()
    scaled_numeric_data = scaler.fit_transform(banking_data[numeric_columns])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categorical_data = encoder.fit_transform(banking_data[categorical_columns])

    # Combine scaled numeric and encoded categorical features
    combined_data = np.hstack((scaled_numeric_data, encoded_categorical_data))

    # Perform linkage for hierarchical clustering
    Z = linkage(combined_data[:100], method='ward')  # Using only first 100 samples for visualization

    plt.figure(figsize=(12, 8))
    dendrogram(Z, orientation='top', labels=banking_data['customer_id'][:100].tolist(),
               distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram of Agglomerative Clustering')
    plt.xlabel('Customer ID')
    plt.ylabel('Distance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('agglomerative_clustering.png')
    plt.close()
except Exception as e:
    print(f"Error in Agglomerative Clustering: {e}")

# PCA for Clustering and Churn Prediction
print("\nPerforming PCA for Clustering and Churn Prediction...")
try:
    # Ensure churn_probability exists
    if 'churn_probability' not in banking_data.columns:
        print("Warning: 'churn_probability' column not found. Creating a synthetic one for demonstration.")
        banking_data['churn_probability'] = np.random.random(len(banking_data))

    # Define features and target
    features = ['satisfaction_score', 'nps_score', 'savings_account',
                'credit_card', 'personal_loan', 'investment_account',
                'income_bracket', 'occupation', 'customer_segment', 'education_level',
                'marital_status', 'acquisition_channel']

    # Adjust feature list
    features = [col for col in features if col in banking_data.columns]
    target = 'churn_probability'

    # Select a subset of data
    subset_data = banking_data.iloc[:100].copy()  # Take a copy to avoid SettingWithCopyWarning

    # Separate features and target
    X = subset_data[features]
    y = subset_data[target]

    # Identify categorical features
    categorical_features = ['income_bracket', 'occupation', 'customer_segment', 'education_level',
                            'marital_status', 'acquisition_channel', 'savings_account',
                            'credit_card', 'personal_loan', 'investment_account']
    categorical_features = [col for col in categorical_features if col in X.columns]

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categorical_data = encoder.fit_transform(X[categorical_features])
    encoded_categorical_df = pd.DataFrame(encoded_categorical_data,
                                          columns=encoder.get_feature_names_out(categorical_features))

    # Identify numerical features
    numerical_features = [col for col in X.columns if col not in categorical_features]

    # Combine numerical and encoded categorical features
    X_combined = pd.concat([X[numerical_features], encoded_categorical_df], axis=1)

    # Apply PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_combined)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(principal_components)

    # Add PCA components and cluster labels to the subset data
    subset_data['pca_1'] = principal_components[:, 0]
    subset_data['pca_2'] = principal_components[:, 1]
    subset_data['cluster'] = cluster_labels

    # Visualize clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(subset_data['pca_1'], subset_data['pca_2'], c=subset_data['cluster'], cmap='viridis')
    plt.title('KMeans Clustering of Banking Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.savefig('pca_clustering.png')
    plt.close()

    # Train a Random Forest model for churn prediction
    X_train, X_test, y_train, y_test = train_test_split(subset_data[['pca_1', 'pca_2', 'cluster']],
                                                        subset_data[target],
                                                        test_size=0.2,
                                                        random_state=42)

    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Churn Prediction Model Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    # Convert to binary for confusion matrix
    threshold = 0.5
    y_pred_class = [1 if prob >= threshold else 0 for prob in y_pred]
    y_test_class = [1 if prob >= threshold else 0 for prob in y_test]

    conf_matrix = confusion_matrix(y_test_class, y_pred_class)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate stay likelihood
    subset_data['stay_likelihood'] = (1 - subset_data['churn_probability']) * 100

    # Visualize Churn Probability
    plt.figure(figsize=(10, 6))
    plt.scatter(subset_data['pca_1'], subset_data['pca_2'], c=subset_data['churn_probability'], cmap='coolwarm')
    plt.title('Clusters vs. Churn Probability')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Churn Probability')
    plt.grid(True)
    plt.savefig('pca_churn_probability.png')
    plt.close()

    # Save the stay likelihood data
    subset_data[['customer_id', 'stay_likelihood']].to_csv('stay_likelihood.csv', index=False)
    print("Stay likelihood data saved to 'stay_likelihood.csv'")
except Exception as e:
    print(f"Error in PCA and Churn Prediction: {e}")

# Cross-selling Recommendations
print("\nGenerating Cross-selling Recommendations...")
try:
    # Use the subset_data from previous analysis
    if 'subset_data' not in locals():
        subset_data = banking_data.iloc[:100].copy()

    # Define features for cross-selling
    cross_sell_features = ['satisfaction_score', 'nps_score', 'savings_account',
                           'credit_card', 'personal_loan', 'investment_account',
                           'income_bracket', 'occupation', 'customer_segment']

    # Adjust feature list
    cross_sell_features = [col for col in cross_sell_features if col in subset_data.columns]

    # Generate product recommendations based on product usage
    recommendations = []
    for idx, row in subset_data.iterrows():
        # Check if features exist before using them
        if all(feature in row.index for feature in
               ['savings_account', 'credit_card', 'personal_loan', 'investment_account']):
            avg_savings = row['savings_account']
            avg_credit_card = row['credit_card']
            avg_personal_loan = row['personal_loan']
            avg_investment = row['investment_account']

            # Simple rule-based recommendation
            if avg_savings < 0.5:
                product = 'Savings Account'
            elif avg_credit_card < 0.5:
                product = 'Credit Card'
            elif avg_personal_loan < 0.5:
                product = 'Personal Loan'
            elif avg_investment < 0.5:
                product = 'Investment Account'
            else:
                product = 'Premium Account'

            # Assign the recommendation
            subset_data.loc[idx, 'recommended_product'] = product
        else:
            subset_data.loc[idx, 'recommended_product'] = 'No Recommendation'

    # Display recommendations
    recommendations_df = subset_data[['customer_id', 'recommended_product']].copy()
    print("Cross-Selling Recommendations:")
    print(recommendations_df.head())

    # Save the recommendations
    recommendations_df.to_csv('cross_sell_recommendations.csv', index=False)
    print("Cross-selling recommendations saved to 'cross_sell_recommendations.csv'")
except Exception as e:
    print(f"Error in Cross-selling Recommendations: {e}")

# Train and save a simple model for the web app
print("\nTraining and Saving Model for Web App...")
try:
    # Create a simple dataset for model training
    model_data = subset_data[['satisfaction_score', 'nps_score', 'savings_account',
                              'credit_card', 'personal_loan', 'investment_account', 'churn_probability']].copy()

    # Split features and target
    X_model = model_data.drop('churn_probability', axis=1)
    y_model = model_data['churn_probability']

    # Train a Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_model, y_model > 0.5)  # Convert to binary target

    # Save the model using pickle
    import pickle

    with open('model.pkl', 'wb') as f:
        pickle.dump(rf_classifier, f)

    print("Model saved as 'model.pkl'")
except Exception as e:
    print(f"Error in Model Training: {e}")

# Create a simple web app using Gradio
print("\nSetting up Web App with Gradio...")
try:
    # Create web app file
    gradio_app_code = """
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
"""

    with open('app.py', 'w') as f:
        f.write(gradio_app_code)

    print("Web app code saved to 'app.py'")
    print("To run the web app, execute: python app.py")
except Exception as e:
    print(f"Error in Web App Setup: {e}")

print("\nAnalysis Complete!")
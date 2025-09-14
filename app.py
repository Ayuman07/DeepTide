import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress the scikit-learn UserWarning about feature names
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Tsunami Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Use caching to avoid re-loading the data and re-training the model on every interaction.
# st.cache_data is for functions that return data, st.cache_resource is for functions that create models.

@st.cache_data
def load_and_preprocess_data():
    """
    Loads and preprocesses the tsunami dataset.
    This function will be cached and only run once.
    """
    try:
        # Use on_bad_lines to skip rows with parsing errors
        df = pd.read_csv('tsunami_dataset.csv', on_bad_lines='skip')
    except FileNotFoundError:
        st.error("Error: 'tsunami_dataset.csv' not found. Please place the file in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

    # Drop irrelevant columns
    columns_to_drop = ['ID', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'URL', 'COMMENTS',
                       'LOCATION_NAME', 'COUNTRY', 'REGION', 'CAUSE', 'DAMAGE_TOTAL_DESCRIPTION',
                       'HOUSES_TOTAL_DESCRIPTION', 'DEATHS_TOTAL_DESCRIPTION']
    df.drop(columns=columns_to_drop, inplace=True)

    # Handle missing values by filling with the median
    for col in ['LATITUDE', 'LONGITUDE', 'EQ_MAGNITUDE', 'EQ_DEPTH', 'TS_INTENSITY']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)

    # Convert the target variable to a binary format
    df['TSUNAMI_OCCURRED'] = df['EVENT_VALIDITY'].apply(lambda x: 1 if 'Tsunami' in str(x) else 0)

    # Drop the original columns and any remaining NaN values
    df.drop(columns=['EVENT_VALIDITY', 'TS_INTENSITY'], inplace=True)
    df.dropna(inplace=True)

    # Select features and target
    features = ['LATITUDE', 'LONGITUDE', 'EQ_MAGNITUDE', 'EQ_DEPTH']
    target = 'TSUNAMI_OCCURRED'

    X = df[features]
    y = df[target]

    return X, y

@st.cache_resource
def train_model(X, y):
    """
    Trains the Gradient Boosting Classifier model.
    This function will be cached and only run once.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model and print to the console for debugging/information
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.info(f"Model Accuracy: {accuracy * 100:.2f}%")
    return model

# Main application logic
st.title("🌊 Tsunami Predictor App")
st.write("This application uses a trained machine learning model to predict the likelihood of a tsunami based on earthquake parameters.")

# Load data and train model
X, y = load_and_preprocess_data()
model = train_model(X, y)

# Sidebar for information
with st.sidebar:
    st.header("About the Model")
    st.write("The model is a **Gradient Boosting Classifier** trained on historical earthquake and tsunami data.")
    st.write(f"**Number of training samples:** {len(X)}")

# Create interactive input widgets
st.header("Enter Earthquake Parameters")
col1, col2 = st.columns(2)

with col1:
    latitude = st.slider("Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.1, format="%.2f°")
    magnitude = st.slider("Magnitude ($M_w$)", min_value=3.0, max_value=10.0, value=7.0, step=0.1, format="%.2f")

with col2:
    longitude = st.slider("Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.1, format="%.2f°")
    depth = st.slider("Depth (km)", min_value=0, max_value=1000, value=50, step=1)

# Make and display the prediction
if st.button("Predict Tsunami"):
    # Prepare the input data for the model
    input_data = np.array([[latitude, longitude, magnitude, depth]])

    # Make the prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"🔴 Tsunami Predicted! (Confidence: {prediction_proba[1] * 100:.2f}%)")
        st.markdown("Based on the provided parameters, the model predicts that a tsunami is likely to occur.")
    else:
        st.success(f"🟢 No Tsunami Predicted. (Confidence: {prediction_proba[0] * 100:.2f}%)")
        st.markdown("Based on the provided parameters, the model predicts that a tsunami is not likely to occur.")

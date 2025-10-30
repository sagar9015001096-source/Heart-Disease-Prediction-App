import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# --- 1. Load Data and Train Model Function ---

@st.cache_resource
def load_and_train_model(csv_file):
    """Loads data, prepares the model pipeline, and trains the model."""
    
    # Load the dataset
    df = pd.read_csv(csv_file)
    
    # Data Cleaning & Target Binarization
    df_clean = df.copy()
    df_clean = df_clean.drop(['id', 'dataset'], axis=1)

    # Binarize the target variable 'num': 0 (No Disease) vs 1 (Disease)
    df_clean['target'] = df_clean['num'].apply(lambda x: 0 if x == 0 else 1)
    df_clean = df_clean.drop('num', axis=1)

    # Convert boolean/object columns to string 'True'/'False' for consistent OneHotEncoding
    df_clean['fbs'] = df_clean['fbs'].astype(str)
    df_clean['exang'] = df_clean['exang'].astype(str)
    
    # Separate features (X) and target (y)
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']

    # Define Feature Types
    numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    # Create Preprocessing Pipeline
    
    # Preprocessor for numerical features: impute with median, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessor for categorical features: impute with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a ColumnTransformer to apply transformations to the correct columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    # Split data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create the full pipeline (Preprocessor + Model)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline, X.columns.tolist()

# Load model and features (This runs only once thanks to @st.cache_resource)
try:
    model_pipeline, feature_cols = load_and_train_model("heart_disease_uci.csv")
except Exception as e:
    st.error(f"Error loading and training model: {e}")
    st.stop()


# --- 2. Streamlit App Layout and Inputs ---

st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

st.title("💖 Heart Disease Prediction App")
st.markdown("### Powered by Machine Learning (Random Forest)")
st.write("""
    This application predicts the presence of heart disease (1) or absence (0) 
    based on the input clinical parameters.
""")

# Define input fields based on feature columns
with st.form("prediction_form"):
    st.header("Patient Clinical Data")
    
    # --- Row 1: Age and Sex ---
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 50, help="Age of the patient in years.")
    with col2:
        sex = st.selectbox("Sex", ("Male", "Female"), help="Patient's biological sex.")

    # --- Row 2: Chest Pain and Resting BP ---
    col3, col4 = st.columns(2)
    with col3:
        cp = st.selectbox("Chest Pain Type (cp)", 
                          ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'),
                          help="Type of chest pain experienced.")
    with col4:
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120, 
                                   help="Resting blood pressure in mm Hg.")

    # --- Row 3: Cholesterol and Fasting Blood Sugar ---
    col5, col6 = st.columns(2)
    with col5:
        chol = st.number_input("Serum Cholesterol (chol)", 100, 600, 240, 
                               help="Serum cholesterol in mg/dl.")
    with col6:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", 
                           ('True', 'False'),
                           help="Fasting Blood Sugar > 120 mg/dl (True/False).")

    # --- Row 4: Resting ECG and Max Heart Rate ---
    col7, col8 = st.columns(2)
    with col7:
        restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", 
                               ('lv hypertrophy', 'normal', 'st-t abnormality'),
                               help="ECG results at rest.")
    with col8:
        thalch = st.number_input("Maximum Heart Rate Achieved (thalch)", 70, 220, 150, 
                                 help="Max heart rate during the stress test.")

    # --- Row 5: Exercise Induced Angina and Oldpeak ---
    col9, col10 = st.columns(2)
    with col9:
        exang = st.selectbox("Exercise Induced Angina (exang)", 
                             ('True', 'False'),
                             help="Presence of angina induced by exercise.")
    with col10:
        oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", 0.0, 6.2, 1.0, step=0.1, 
                                  help="ST depression relative to rest.")

    # --- Row 6: Slope, CA, and Thallium Scan ---
    col11, col12, col13 = st.columns(3)
    with col11:
        slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", 
                             ('upsloping', 'flat', 'downsloping'),
                             help="The slope of the peak exercise ST segment.")
    with col12:
        ca = st.slider("Number of Major Vessels Colored by Flouroscopy (ca)", 0, 3, 0, 
                       help="Number of major vessels (0-3) colored by flouroscopy.")
    with col13:
        thal = st.selectbox("Thallium Stress Test Result (thal)", 
                            ('normal', 'fixed defect', 'reversable defect'),
                            help="Result of the thallium stress test.")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Predict Heart Disease Risk")


# --- 3. Prediction Logic and Output ---

if submitted:
    # 1. Create a dictionary of all inputs
    input_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalch': thalch, 'exang': exang, 'oldpeak': oldpeak, 
        'slope': slope, 'ca': ca, 'thal': thal
    }
    
    # 2. Convert to DataFrame (preserving the feature order/columns used in training)
    input_df = pd.DataFrame([input_data], columns=feature_cols)
    
    # 3. Make Prediction
    try:
        prediction = model_pipeline.predict(input_df)[0]
        prediction_proba = model_pipeline.predict_proba(input_df)[0]
        
        st.subheader("Prediction Result:")

        if prediction == 1:
            st.error(f"**High Risk of Heart Disease**")
            st.markdown(f"**Probability of Disease:** `{prediction_proba[1]:.2%}`")
            st.warning("Please consult a healthcare professional for diagnosis.")
        else:
            st.success(f"**Low Risk of Heart Disease**")
            st.markdown(f"**Probability of No Disease:** `{prediction_proba[0]:.2%}`")
            st.info("The model suggests a low risk. This is not a substitute for medical advice.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
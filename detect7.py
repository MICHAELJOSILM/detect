import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import io
import base64
import pickle

# Set page configuration
st.set_page_config(page_title="Blockchain Fraud Detection", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-title {
        font-size: 26px;
        font-weight: bold;
        color: #333;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .highlight {
        color: #E53935;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">Blockchain Transaction Fraud Detection System</div>', unsafe_allow_html=True)

# Create tabs for the application
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🔍 Feature Engineering", "🛠️ Model Training", "📈 Evaluation", "🔮 Prediction"])

# Function to load data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

# Function to create a download link for the model
def get_download_link(model, filename="blockchain_fraud_model.pkl"):
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Trained Model</a>'
    return href

# Function to preprocess data
def preprocess_data(df):
    # Remove irrelevant columns
    if 'Index' in df.columns:
        df = df.drop('Index', axis=1)
    if 'Address' in df.columns:
        df = df.drop('Address', axis=1)

    # Handle missing values
    df = df.fillna(0)

    # Create new features
    df['sent_received_ratio'] = df['Sent tnx'] / (df['Received Tnx'] + 1)  # Adding 1 to avoid division by zero
    df['avg_value_per_tx'] = (df['total Ether sent'] + df['total ether received']) / (df['Sent tnx'] + df['Received Tnx'] + 1)
    df['contract_interaction_rate'] = df['Number of Created Contracts'] / (df['total transactions (including tnx to create contract'] + 1)
    df['unique_address_ratio'] = (df['Unique Sent To Addresses'] + df['Unique Received From Addresses']) / (df['Sent tnx'] + df['Received Tnx'] + 1)
    df['balance_to_transaction_ratio'] = df['total ether balance'] / (df['total Ether sent'] + df['total ether received'] + 1)

    # Convert token name columns to string first to handle any binary data, then to categorical
    token_cols = ['ERC20 uniq sent token name', 'ERC20 uniq rec token name', 'ERC20 most sent token type', 'ERC20_most_rec_token_type']
    for col in token_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].astype('category')

    return df

# Function to extract features and targets
def extract_features_targets(df):
    X = df.drop('FLAG', axis=1)
    y = df['FLAG']

    categorical_features = X.select_dtypes(include=['category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    return X, y, preprocessor

# Function to build and train models
def train_models(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    nn_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
    ])

    iso_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', IsolationForest(contamination=0.1, random_state=42))
    ])

    rf_model.fit(X_train, y_train)
    nn_model.fit(X_train, y_train)
    iso_model.fit(X_train)

    if len(X_train) > 50:  # Minimum threshold for LSTM
        X_train_array = preprocessor.fit_transform(X_train)
        if hasattr(X_train_array, "toarray"):
            X_train_array = X_train_array.toarray()

        X_train_lstm = X_train_array.reshape(X_train_array.shape[0], 1, X_train_array.shape[1])
        X_test_array = preprocessor.transform(X_test)
        if hasattr(X_test_array, "toarray"):
            X_test_array = X_test_array.toarray()

        X_test_lstm = X_test_array.reshape(X_test_array.shape[0], 1, X_test_array.shape[1])

        lstm_model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(1, X_train_array.shape[1])),
            Dropout(0.2),
            LSTM(units=30),
            Dropout(0.2),
            Dense(units=16, activation='relu'),
            Dense(units=1, activation='sigmoid')
        ])

        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

        return rf_model, nn_model, iso_model, lstm_model, X_test, y_test, X_test_lstm
    else:
        return rf_model, nn_model, iso_model, None, X_test, y_test, None

# Function to evaluate models
def evaluate_models(rf_model, nn_model, iso_model, lstm_model, X_test, y_test, X_test_lstm):
    evaluation_results = {}

    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    evaluation_results['Random Forest'] = (rf_accuracy, confusion_matrix(y_test, rf_pred), classification_report(y_test, rf_pred, output_dict=True))

    nn_pred = nn_model.predict(X_test)
    nn_accuracy = accuracy_score(y_test, nn_pred)
    evaluation_results['Neural Network'] = (nn_accuracy, confusion_matrix(y_test, nn_pred), classification_report(y_test, nn_pred, output_dict=True))

    iso_pred = iso_model.predict(X_test)
    iso_pred = np.where(iso_pred == -1, 1, 0)  # -1 indicates anomaly (fraud)
    iso_accuracy = accuracy_score(y_test, iso_pred)
    evaluation_results['Isolation Forest'] = (iso_accuracy, confusion_matrix(y_test, iso_pred), classification_report(y_test, iso_pred, output_dict=True))

    if lstm_model is not None and X_test_lstm is not None:
        lstm_pred_prob = lstm_model.predict(X_test_lstm)
        lstm_pred = (lstm_pred_prob > 0.5).astype(int).reshape(-1)
        lstm_accuracy = accuracy_score(y_test, lstm_pred)
        evaluation_results['LSTM'] = (lstm_accuracy, confusion_matrix(y_test, lstm_pred), classification_report(y_test, lstm_pred, output_dict=True))
    
    return evaluation_results

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]

        num_features = len(importances)
        top_n = min(15, num_features)

        top_indices = indices[:top_n]
        top_importances = importances[top_indices]

        feature_names_array = np.array(feature_names)

        if np.max(top_indices) < len(feature_names_array):
            top_features = feature_names_array[top_indices]
        else:
            st.error("Top indices exceed the number of available features.")
            return None

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=top_features,
                x=top_importances,
                orientation='h',
                marker_color='rgba(55, 83, 109, 0.7)',
                marker_line_color='rgba(55, 83, 109, 1.0)',
                marker_line_width=1
            )
        )

        fig.update_layout(
            title='Top Feature Importances',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=600,
            template='plotly_white'
        )

        return fig
    else:
        return None

# Main application
with tab1:
    st.markdown('<div class="section-title">Upload Your Blockchain Transaction Dataset</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    use_sample = st.checkbox("Use provided sample data (for demonstration)")

    if use_sample:
        sample_data = {
            'Index': [1] * 1000,
            'Address': ['0x00009277775ac7d0d59eaad8fee3d10ac6c805e8'] * 1000,
            'FLAG': np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),
            'Avg min between sent tnx': np.random.normal(844.26, 100, 1000),
            'Avg min between received tnx': np.random.normal(1093.71, 150, 1000),
            'Time Diff between first and last (Mins)': np.random.normal(704785.63, 10000, 1000),
            'Sent tnx': np.random.poisson(721, 1000),
            'Received Tnx': np.random.poisson(89, 1000),
            'Number of Created Contracts': np.random.poisson(1, 1000),
            'Unique Received From Addresses': np.random.poisson(40, 1000),
            'Unique Sent To Addresses': np.random.poisson(118, 1000),
            'min value received': np.zeros(1000),
            'max value received': np.random.normal(45.81, 5, 1000),
            'avg val received': np.random.normal(6.59, 1, 1000),
            'min val sent': np.zeros(1000),
            'max val sent': np.random.normal(31.22, 3, 1000),
            'avg val sent': np.random.normal(1.20, 0.2, 1000),
            'total transactions (including tnx to create contract': np.random.poisson(810, 1000),
            'total Ether sent': np.random.normal(865.69, 100, 1000),
            'total ether received': np.random.normal(586.47, 70, 1000),
            'total ether balance': np.random.normal(-279.22, 50, 1000),
            'Total ERC20 tnxs': np.random.poisson(265, 1000),
            'ERC20 total Ether received': np.random.normal(35588543.78, 1000000, 1000),
            'ERC20 total ether sent': np.random.normal(35603169.52, 1000000, 1000),
            'ERC20 uniq sent addr': np.random.poisson(30, 1000),
            'ERC20 uniq rec addr': np.random.poisson(54, 1000),
            'ERC20 uniq sent token name': np.random.choice(['Cofoundit', 'Ethereum', 'Bitcoin', 'Tether'], 1000),
            'ERC20 uniq rec token name': np.random.choice(['Numeraire', 'Ethereum', 'Bitcoin', 'Tether'], 1000),
            'ERC20 most sent token type': np.random.choice(['Cofoundit', 'Ethereum', 'Bitcoin', 'Tether'], 1000),
            'ERC20_most_rec_token_type': np.random.choice(['Numeraire', 'Ethereum', 'Bitcoin', 'Tether'], 1000)
        }
        df = pd.DataFrame(sample_data)
    elif uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        st.info("Please upload a CSV file or use the sample data to proceed.")
        df = None

    if df is not None:
        st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Transactions", len(df))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'FLAG' in df.columns:
                fraud_count = df['FLAG'].sum()
                fraud_percent = (fraud_count / len(df)) * 100
                st.metric("Fraudulent Transactions", f"{fraud_count} ({fraud_percent:.2f}%)")
            else:
                st.warning("FLAG column not found in dataset")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'total Ether sent' in df.columns:
                total_ether = df['total Ether sent'].sum()
                st.metric("Total Ether Sent", f"{total_ether:.2f} ETH")
            else:
                st.warning("total Ether sent column not found")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'total transactions (including tnx to create contract' in df.columns:
                avg_tx = df['total transactions (including tnx to create contract'].mean()
                st.metric("Avg Transactions per Address", f"{avg_tx:.2f}")
            else:
                st.warning("total transactions column not found")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Data preview
        st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head())

        # Basic statistics
        st.markdown('<div class="section-title">Basic Statistics</div>', unsafe_allow_html=True)
        st.write(df.describe())

        # Distribution of transactions
        st.markdown('<div class="section-title">Transaction Distributions</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            if 'Sent tnx' in df.columns and 'Received Tnx' in df.columns:
                fig = px.histogram(df, x='Sent tnx', color_discrete_sequence=['#1E88E5'])
                fig.update_layout(title='Distribution of Sent Transactions', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'Sent tnx' in df.columns and 'Received Tnx' in df.columns:
                fig = px.histogram(df, x='Received Tnx', color_discrete_sequence=['#43A047'])
                fig.update_layout(title='Distribution of Received Transactions', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

        # Correlation matrix
        st.markdown('<div class="section-title">Feature Correlation Matrix</div>', unsafe_allow_html=True)

        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        corr = numeric_df.corr()

        fig = px.imshow(corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r')
        fig.update_layout(title='Correlation Matrix of Numeric Features', height=800, width=800)
        st.plotly_chart(fig)

with tab2:
    if df is not None:
        st.markdown('<div class="section-title">Feature Engineering</div>', unsafe_allow_html=True)

        processed_df = preprocess_data(df)

        # Display the new features
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Engineered Features')
        st.write("The following features have been created to improve model performance:")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('- **sent_received_ratio**: Ratio of sent transactions to received transactions')
            st.markdown('- **avg_value_per_tx**: Average value per transaction')
            st.markdown('- **contract_interaction_rate**: Rate of contract creation relative to all transactions')

        with col2:
            st.markdown('- **unique_address_ratio**: Ratio of unique addresses to total transactions')
            st.markdown('- **balance_to_transaction_ratio**: Ratio of balance to total transaction value')

        st.markdown('</div>', unsafe_allow_html=True)

        # Display processed data
        st.markdown('### Processed Dataset')
        st.dataframe(processed_df.head())

        # Feature distributions by class
        st.markdown('<div class="section-title">Feature Distributions by Class</div>', unsafe_allow_html=True)

        if 'FLAG' in processed_df.columns:
            numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

            if 'FLAG' in numeric_cols:
                numeric_cols.remove('FLAG')

            selected_feature = st.selectbox('Select a feature to visualize distribution by class:', numeric_cols)

            fig = px.histogram(processed_df, x=selected_feature, color='FLAG',
                            barmode='overlay', color_discrete_sequence=['#4CAF50', '#F44336'])
            fig.update_layout(title=f'Distribution of {selected_feature} by Class',
                            xaxis_title=selected_feature,
                            yaxis_title='Count',
                            template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-title">Feature Pair Visualization</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                feature_x = st.selectbox('Select X-axis feature:', numeric_cols, index=0)

            with col2:
                default_index = 1 if len(numeric_cols) > 1 else 0
                feature_y = st.selectbox('Select Y-axis feature:', numeric_cols, index=default_index)

            fig = px.scatter(processed_df, x=feature_x, y=feature_y, color='FLAG',
                            opacity=0.7, color_discrete_sequence=['#4CAF50', '#F44336'])
            fig.update_layout(title=f'{feature_x} vs {feature_y} by Class',
                            xaxis_title=feature_x,
                            yaxis_title=feature_y,
                            template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    if df is not None:
        st.markdown('<div class="section-title">Model Training</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Select Models for Training')

        col1, col2 = st.columns(2)

        with col1:
            use_rf = st.checkbox('Random Forest Classifier', value=True)
            use_nn = st.checkbox('Neural Network (MLP)', value=True)

        with col2:
            use_iso = st.checkbox('Isolation Forest (Anomaly Detection)', value=True)
            use_lstm = st.checkbox('LSTM Neural Network', value=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Model Descriptions')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('**Random Forest Classifier**')
            st.markdown('- Ensemble learning method using multiple decision trees')
            st.markdown('- Effective for classification tasks with tabular data')
            st.markdown('- Provides feature importance for interpretability')

            st.markdown('**Neural Network (MLP)**')
            st.markdown('- Multi-layer perceptron with hidden layers')
            st.markdown('- Can capture complex non-linear relationships')
            st.markdown('- Works well with standardized numeric features')

        with col2:
            st.markdown('**Isolation Forest**')
            st.markdown('- Unsupervised anomaly detection algorithm')
            st.markdown('- Identifies outliers by isolation')
            st.markdown('- Works well for fraud detection where anomalies are rare')

            st.markdown('**LSTM Neural Network**')
            st.markdown('- Long Short-Term Memory network')
            st.markdown('- Captures temporal patterns in transaction sequences')
            st.markdown('- Effective for time-series transaction data')

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button('Start Model Training'):
            with st.spinner('Preprocessing data...'):
                processed_df = preprocess_data(df)

                if 'FLAG' in processed_df.columns:
                    X = processed_df.drop('FLAG', axis=1)
                    y = processed_df['FLAG']

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns),
                            ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['category']).columns)
                        ],
                        remainder='passthrough'
                    )

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    X_train_transformed = preprocessor.fit_transform(X_train)

                    feature_names = preprocessor.get_feature_names_out()

                    rf_model, nn_model, iso_model, lstm_model, X_test, y_test, X_test_lstm = train_models(X, y, preprocessor)

                    st.session_state['models'] = {
                        'Random Forest': rf_model,
                        'Neural Network': nn_model,
                        'Isolation Forest': iso_model,
                        'LSTM': lstm_model
                    }
                    st.session_state['feature_names'] = feature_names
                    st.session_state['test_data'] = (X_test, y_test)
                    st.session_state['X_test_lstm'] = X_test_lstm
                    st.session_state['preprocessor'] = preprocessor  # Store preprocessor in session state

                    st.success("All selected models have been trained successfully! Go to the Evaluation tab to see the results.")
                else:
                    st.error("FLAG column not found in the dataset. This column is required for training the models.")

with tab4:
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)

    if 'models' in st.session_state and 'feature_names' in st.session_state and 'test_data' in st.session_state and 'X_test_lstm' in st.session_state:
        models = st.session_state['models']
        feature_names = st.session_state['feature_names']
        X_test, y_test = st.session_state['test_data']
        X_test_lstm = st.session_state['X_test_lstm']  

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Model Comparison')

        evaluation_results = evaluate_models(
            models['Random Forest'],
            models['Neural Network'],
            models['Isolation Forest'],
            models.get('LSTM', None),
            X_test,
            y_test,
            X_test_lstm
        )

        model_names = list(evaluation_results.keys())
        accuracies = [results[0] for results in evaluation_results.values()]

        fig = px.bar(
            x=model_names, 
            y=accuracies,
            labels={'x': 'Model', 'y': 'Accuracy'},
            color=accuracies,
            color_continuous_scale='Viridis',
            title='Model Accuracy Comparison'
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        if accuracies:
            best_model_index = np.argmax(accuracies)
            best_model = model_names[best_model_index]
            st.markdown(f'#### Best Performing Model: <span class="highlight">{best_model}</span> with accuracy {accuracies[best_model_index]:.4f}', unsafe_allow_html=True)
        else:
            st.warning("No valid accuracy values found for comparison.")
        st.markdown('</div>', unsafe_allow_html=True)

        selected_model = st.selectbox('Select a model for detailed evaluation:', model_names)

        if selected_model in evaluation_results:
            results = evaluation_results[selected_model]

            st.markdown('### Confusion Matrix')
            conf_matrix = results[1]

            fig = px.imshow(
                conf_matrix,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Normal (0)', 'Fraud (1)'],
                y=['Normal (0)', 'Fraud (1)'],
                color_continuous_scale='Blues'
            )
            fig.update_layout(title=f'Confusion Matrix - {selected_model}', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('### Classification Report')
            report = results[2]

            report_df = pd.DataFrame(report).transpose()
            if 'support' in report_df.columns:
                report_df = report_df.drop('support', axis=1)

            if 'accuracy' in report_df.index:
                accuracy_row = report_df.loc[['accuracy']]
                report_df = report_df.drop('accuracy')

                keep_rows = [str(i) for i in range(2)] + ['macro avg']
                report_df = report_df.loc[report_df.index.intersection(keep_rows)]

                report_df = report_df.rename(index={'0': 'Normal Transactions', '1': 'Fraud Transactions'})

            st.dataframe(report_df.style.format("{:.4f}"))

            if selected_model == 'Random Forest' and 'classifier' in models[selected_model].named_steps:
                st.markdown('### Feature Importance')

                rf_model = models[selected_model]
                importances = rf_model.named_steps['classifier'].feature_importances_

                indices = np.argsort(importances)[::-1]
                top_n = min(15, len(feature_names))
                top_indices = indices[:top_n]
                top_importances = importances[top_indices]
                top_features = feature_names[top_indices]

                fig = px.bar(
                    x=top_importances,
                    y=top_features,
                    orientation='h',
                    labels={'x': 'Importance', 'y': 'Feature'},
                    title=f'Top {top_n} Feature Importances - {selected_model}',
                    color=top_importances,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

            if selected_model in ['Random Forest', 'Neural Network', 'LSTM']:
                st.markdown('### ROC Curve')

                if hasattr(models[selected_model], 'predict_proba'):
                    y_probs = models[selected_model].predict_proba(X_test)[:, 1]

                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                    roc_auc = auc(fpr, tpr)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.4f})'))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))

                    fig.update_layout(
                        title=f'ROC Curve - {selected_model}',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        template='plotly_white',
                        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if selected_model == 'LSTM':
                    if isinstance(results, dict) and 'history' in results:

                        st.markdown('### LSTM Training History')

                        history = results['history']

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                        fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))

                        fig.update_layout(
                            title='LSTM Training and Validation Loss',
                            xaxis_title='Epoch',
                            yaxis_title='Loss',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)

            st.markdown('### Export Model')
            st.markdown('You can download the trained model for later use:')
            st.markdown(get_download_link(models[selected_model], f'{selected_model.lower().replace(" ", "_")}_model.pkl'), unsafe_allow_html=True)
    else:
        st.info("No models have been trained yet. Please go to the Model Training tab to train models.")

with tab5:
    st.markdown('<div class="section-title">Fraud Prediction</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('### Make Predictions on New Data')

    # Check if models are available
    if 'models' in st.session_state and 'preprocessor' in st.session_state:
        models = st.session_state['models']
        preprocessor = st.session_state['preprocessor']  # Ensure preprocessor is available

        # Upload new data for prediction
        new_data_file = st.file_uploader("Upload new blockchain transaction data for prediction", type="csv")

        # Option to use sample data for prediction
        use_sample_for_pred = st.checkbox("Use sample data for prediction demonstration")

        pred_df = None

        if use_sample_for_pred:
            sample_data = {
                'Address': ['0x00009277775ac7d0d59eaad8fee3d10ac6c805e8'],
                'Avg min between sent tnx': [844.26],
                'Avg min between received tnx': [1093.71],
                'Time Diff between first and last (Mins)': [704785.63],
                'Sent tnx': [721],
                'Received Tnx': [89],
                'Number of Created Contracts': [1],
                'Unique Received From Addresses': [40],
                'Unique Sent To Addresses': [118],
                'min value received': [0],
                'max value received': [45.81],
                'avg val received': [6.59],
                'min val sent': [0],
                'max val sent': [31.22],
                'avg val sent': [1.20],
                'total transactions (including tnx to create contract': [810],
                'total Ether sent': [865.69],
                'total ether received': [586.47],
                'total ether balance': [-279.22],
                'Total ERC20 tnxs': [265],
                'ERC20 total Ether received': [35588543.78],
                'ERC20 total ether sent': [35603169.52],
                'ERC20 uniq sent addr': [30],
                'ERC20 uniq rec addr': [54],
                'ERC20 uniq sent token name': ['Cofoundit'],
                'ERC20 uniq rec token name': ['Numeraire'],
                'ERC20 most sent token type': ['Cofoundit'],
                'ERC20_most_rec_token_type': ['Numeraire']
            }
            pred_df = pd.DataFrame(sample_data)
        elif new_data_file is not None:
            pred_df = load_data(new_data_file)

        if pred_df is not None:
            # Display the data to be predicted
            st.dataframe(pred_df.head())

            # Preprocess the data
            processed_pred_df = preprocess_data(pred_df)

            # Select model for prediction
            selected_model_name = st.selectbox('Select a model for prediction:', list(models.keys()))

            if st.button('Run Prediction'):
                with st.spinner('Making predictions...'):
                    selected_model = models[selected_model_name]

                    if selected_model_name == 'LSTM':
                        if 'FLAG' in processed_pred_df.columns:
                            processed_pred_df = processed_pred_df.drop('FLAG', axis=1)

                        # Transform the data using the preprocessor
                        pred_data_array = preprocessor.transform(processed_pred_df)
                        
                        # Ensure it's a dense array if it's sparse
                        if hasattr(pred_data_array, "toarray"):
                            pred_data_array = pred_data_array.toarray()

                        # Reshape the array for LSTM input
                        pred_data_lstm = pred_data_array.reshape(pred_data_array.shape[0], 1, pred_data_array.shape[1])

                        pred_probs = selected_model.predict(pred_data_lstm)
                        predictions = (pred_probs > 0.5).astype(int).reshape(-1)
                        pred_probs = pred_probs.reshape(-1)
                    else:
                        if 'FLAG' in processed_pred_df.columns:
                            processed_pred_df = processed_pred_df.drop('FLAG', axis=1)

                        if hasattr(selected_model, 'predict_proba'):
                            pred_probs = selected_model.predict_proba(processed_pred_df)[:, 1]
                            predictions = (pred_probs > 0.5).astype(int)
                        else:
                            predictions = selected_model.predict(processed_pred_df)
                            if selected_model_name == 'Isolation Forest':
                                predictions = np.where(predictions == -1, 1, 0)
                            pred_probs = np.where(predictions == 1, 0.9, 0.1)

                    result_df = pred_df.copy()
                    result_df['Fraud_Prediction'] = predictions
                    result_df['Fraud_Probability'] = pred_probs

                    st.markdown('### Prediction Results')

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Total Transactions", len(result_df))
                        st.metric("Flagged as Fraud", int(result_df['Fraud_Prediction'].sum()))

                    with col2:
                        fraud_percent = (result_df['Fraud_Prediction'].sum() / len(result_df)) * 100
                        st.metric("Fraud Percentage", f"{fraud_percent:.2f}%")
                        avg_fraud_prob = result_df['Fraud_Probability'].mean() * 100
                        st.metric("Average Fraud Probability", f"{avg_fraud_prob:.2f}%")

                    st.dataframe(result_df)

                    if len(result_df) > 1:
                        st.markdown('### Prediction Visualization')

                        fig = px.histogram(result_df, x='Fraud_Probability', nbins=20,
                                           labels={'Fraud_Probability': 'Fraud Probability'},
                                           title='Distribution of Fraud Probabilities',
                                           color_discrete_sequence=['#E53935'])
                        fig.update_layout(template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)

                        if 'total Ether sent' in result_df.columns and 'Sent tnx' in result_df.columns:
                            fig = px.scatter(result_df,
                                             x='total Ether sent',
                                             y='Sent tnx',
                                             color='Fraud_Probability',
                                             size='total transactions (including tnx to create contract' if 'total transactions (including tnx to create contract' in result_df.columns else None,
                                             hover_data=['Address'] if 'Address' in result_df.columns else None,
                                             color_continuous_scale='Viridis',
                                             title='Transaction Value vs. Activity with Fraud Probability')
                            fig.update_layout(template='plotly_white')
                            st.plotly_chart(fig, use_container_width=True)

                    csv = result_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="fraud_predictions.csv">Download Prediction Results (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    if int(result_df['Fraud_Prediction'].sum()) > 0:
                        st.markdown('### Fraud Investigation Report')

                        fraud_df = result_df[result_df['Fraud_Prediction'] == 1]

                        st.markdown('<div style="background-color: #ffebee; padding: 15px; border-radius: 5px;">', unsafe_allow_html=True)
                        st.markdown(f"#### Alert: {len(fraud_df)} Potentially Fraudulent Transactions Detected")

                        if 'Fraud_Probability' in fraud_df.columns:
                            st.markdown("##### Most Suspicious Transactions:")
                            top_fraud = fraud_df.sort_values('Fraud_Probability', ascending=False).head(5)

                            for i, (_, row) in enumerate(top_fraud.iterrows()):
                                st.markdown(f"{i+1}. **Address:** {row['Address'] if 'Address' in row else 'Unknown'}")
                                st.markdown(f"   **Fraud Probability:** {row['Fraud_Probability'] * 100:.2f}%")
                                st.markdown(f"   **Ether Sent:** {row['total Ether sent'] if 'total Ether sent' in row else 'N/A'} ETH")

                        st.markdown("##### Recommended Actions:")
                        st.markdown("1. Freeze suspicious accounts pending investigation")
                        st.markdown("2. Perform detailed transaction analysis")
                        st.markdown("3. Verify with KYC data where available")
                        st.markdown("4. Implement additional verification steps for high-value transactions")

                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Please upload data or use the sample data for prediction.")
    else:
        st.info("No trained models available. Please go to the Model Training tab to train models first.")

# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f5f5f5; border-radius: 5px;">
    <p>Blockchain Transaction Fraud Detection System</p>
    <p>Powered by Machine Learning and Deep Learning Algorithms</p>
</div>
""", unsafe_allow_html=True)

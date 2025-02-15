import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Learn Regression Analysis", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .concept-box {
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 10px 0;
    }
    .example-box {
        padding: 15px;
        background-color: #e1e5eb;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main title and introduction
st.title("🎓 Arii's Regression Learning Tool")
st.markdown("""
Welcome to your interactive journey into regression analysis! This tool will help you understand:
- What regression is and how it works
- Different types of regression models
- How to evaluate model performance
- Common pitfalls and how to avoid them
""")

# Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["📚 Introduction to Regression", 
     "🔍 Interactive Visualization", 
     "📊 Model Comparison",
     "📏 Performance Metrics",
     "📈 Advanced Concepts"]
)

# Generate sample data
@st.cache_data
def generate_sample_data(n_samples=100, noise=0.1, scenario='linear'):
    np.random.seed(42)
    X = np.random.randn(n_samples, 1)
    if scenario == 'linear':
        y = 2 * X.squeeze() + np.random.normal(0, noise, n_samples)
    elif scenario == 'polynomial':
        y = 2 * X.squeeze()**2 + np.random.normal(0, noise, n_samples)
    elif scenario == 'complex':
        X = np.random.randn(n_samples, 3)
        y = 2 * X[:, 0] - X[:, 1]**2 + 0.5 * X[:, 2] + np.random.normal(0, noise, n_samples)
        return X, y
    return X, y

if page == "📚 Introduction to Regression":
    st.header("Understanding Regression Analysis")
    
    st.markdown("""
    <div class='concept-box'>
    <h3>What is Regression? 🤔</h3>
    Regression is like being a detective with data! It helps us understand relationships between variables 
    and make predictions. Imagine predicting house prices based on size, location, and number of rooms.
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive Example
    st.subheader("🎮 Try it yourself!")
    
    # Simple interactive visualization
    points = st.slider("Number of data points", 10, 200, 50)
    noise = st.slider("Add some randomness (noise)", 0.0, 2.0, 0.5)
    
    X, y = generate_sample_data(points, noise, 'linear')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.5)
    ax.set_xlabel("Input Variable (X)")
    ax.set_ylabel("Output Variable (y)")
    ax.set_title("Simple Linear Relationship")
    
    # Fit and plot regression line
    model = LinearRegression()
    model.fit(X, y)
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    ax.plot(X_line, y_line, color='red', label='Regression Line')
    ax.legend()
    
    st.pyplot(fig)
    
    st.markdown("""
    <div class='example-box'>
    <h4>What's happening here? 📝</h4>
    - The dots represent your data points
    - The red line is the 'best fit' line that regression finds
    - Try adjusting the sliders to see how noise affects the relationship!
    </div>
    """, unsafe_allow_html=True)

elif page == "🔍 Interactive Visualization":
    st.header("Exploring Different Regression Types")
    
    # Model selection
    model_type = st.selectbox(
        "Select Regression Model",
        ["Linear Regression", "Ridge Regression", "Lasso Regression"]
    )
    
    st.markdown(f"""
    <div class='concept-box'>
    <h3>About {model_type} 📖</h3>
    {
    {
        "Linear Regression": "The simplest form of regression. It finds the best straight line through the data.",
        "Ridge Regression": "Adds a penalty to prevent large coefficients. Good for when you have many correlated features.",
        "Lasso Regression": "Similar to Ridge, but can completely eliminate less important features."
    }[model_type]
    }
    </div>
    """, unsafe_allow_html=True)
    
    # Data generation controls
    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.slider("Number of samples", 50, 500, 100)
    with col2:
        noise_level = st.slider("Noise level", 0.0, 2.0, 0.5)
    with col3:
        data_type = st.selectbox("Data Pattern", ["Linear", "Polynomial", "Complex"])
    
    # Generate and prepare data
    X, y = generate_sample_data(n_samples, noise_level, data_type.lower())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Ridge Regression":
        alpha = st.slider("Ridge alpha", 0.0, 10.0, 1.0)
        model = Ridge(alpha=alpha)
    else:
        alpha = st.slider("Lasso alpha", 0.0, 10.0, 1.0)
        model = Lasso(alpha=alpha)
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Visualizations
    if X.shape[1] == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training data
        ax1.scatter(X_train, y_train, alpha=0.5, label='Actual')
        ax1.scatter(X_train, y_train_pred, alpha=0.5, label='Predicted')
        ax1.set_title("Training Data")
        ax1.legend()
        
        # Test data
        ax2.scatter(X_test, y_test, alpha=0.5, label='Actual')
        ax2.scatter(X_test, y_test_pred, alpha=0.5, label='Predicted')
        ax2.set_title("Test Data")
        ax2.legend()
        
        st.pyplot(fig)
    else:
        st.write("Feature Importance:")
        importance_df = pd.DataFrame({
            'Feature': [f'Feature {i+1}' for i in range(X.shape[1])],
            'Coefficient': model.coef_
        })
        st.bar_chart(importance_df.set_index('Feature'))
    
    # Model evaluation
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Metrics")
        st.write(f"R² Score: {r2_score(y_train, y_train_pred):.4f}")
        st.write(f"MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
    
    with col2:
        st.subheader("Test Metrics")
        st.write(f"R² Score: {r2_score(y_test, y_test_pred):.4f}")
        st.write(f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")

elif page == "📊 Model Comparison":
    st.header("Compare Different Regression Models")
    
    # Generate complex data for comparison
    X, y = generate_sample_data(200, 0.5, 'complex')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0)
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        results.append({
            'Model': name,
            'Training R²': train_score,
            'Test R²': test_score,
            'Coefficients': model.coef_
        })
    
    # Display results
    results_df = pd.DataFrame(results)
    st.dataframe(results_df[['Model', 'Training R²', 'Test R²']])
    
    # Coefficient comparison
    st.subheader("Feature Importance Comparison")
    coef_data = pd.DataFrame(
        [r['Coefficients'] for r in results],
        columns=[f'Feature {i+1}' for i in range(X.shape[1])],
        index=[r['Model'] for r in results]
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    coef_data.T.plot(kind='bar', ax=ax)
    plt.title("Coefficient Values Across Models")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.markdown("""
    <div class='concept-box'>
    <h3>Key Takeaways 🔑</h3>
    - Different models may perform similarly on training data but differently on test data
    - Ridge and Lasso tend to have smaller coefficients than Linear Regression
    - The best model depends on your specific data and needs
    </div>
    """, unsafe_allow_html=True)

elif page == "📏 Performance Metrics":
    st.header("Understanding Performance Metrics")
    
    st.markdown("""
    <div class='concept-box'>
    <h3>Why Do We Need Metrics? 🎯</h3>
    Performance metrics help us understand how well our model is doing. Think of them as different ways to 
    measure the distance between our predictions and the actual values. Each metric has its own strengths 
    and tells us something unique about our model's performance.
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive metric exploration
    st.subheader("🎮 Explore Different Metrics")
    
    # Generate sample data for metrics visualization
    n_samples = st.slider("Number of sample points", 20, 200, 50)
    noise_level = st.slider("Noise level", 0.1, 2.0, 0.5)
    X, y = generate_sample_data(n_samples, noise_level, 'linear')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Metrics visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### MAE vs MSE vs RMSE
        Let's visualize how these metrics measure prediction errors differently:
        """)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        errors = y_test - y_pred
        
        # Plot actual errors
        ax.scatter(range(len(errors)), errors, alpha=0.5, label='Actual Errors')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Plot different error measurements
        ax.plot(range(len(errors)), [mae] * len(errors), 'g--', label=f'MAE: {mae:.2f}')
        ax.plot(range(len(errors)), [rmse] * len(errors), 'r--', label=f'RMSE: {rmse:.2f}')
        
        ax.set_title('Error Metrics Comparison')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Error')
        ax.legend()
        st.pyplot(fig)
        
        st.markdown("""
        <div class='example-box'>
        <h4>Understanding Error Metrics:</h4>
        
        🔹 **MAE (Mean Absolute Error)**
        - Takes the absolute value of errors
        - Easier to interpret
        - Less sensitive to outliers
        - Value: {:.2f}
        
        🔹 **MSE (Mean Squared Error)**
        - Squares the errors
        - Penalizes larger errors more
        - Not in original units
        - Value: {:.2f}
        
        🔹 **RMSE (Root Mean Square Error)**
        - Square root of MSE
        - In original units
        - Popular choice
        - Value: {:.2f}
        </div>
        """.format(mae, mse, rmse), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### R² Score Explained
        R² tells us how much of the variance in the data our model explains:
        """)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot actual vs predicted values
        ax.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
        
        # Plot perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Predictions')
        
        ax.set_title(f'R² Score: {r2:.4f}')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.legend()
        st.pyplot(fig)
        
        st.markdown(f"""
        <div class='example-box'>
        <h4>Understanding R²:</h4>
        
        🔹 **What is R²?**
        - Measures proportion of variance explained
        - Ranges from 0 to 1 (or negative in bad fits)
        - Higher is better
        - Current value: {r2:.4f}
        
        🔹 **Interpretation:**
        - R² = 1: Perfect fit
        - R² = 0: Model just predicts mean
        - R² < 0: Worse than predicting mean
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive error analysis
    st.subheader("🔍 Error Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(errors, bins=20, ax=ax)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Distribution of Prediction Errors')
    ax.set_xlabel('Error')
    st.pyplot(fig)
    
    st.markdown("""
    <div class='concept-box'>
    <h3>Choosing the Right Metric 🤔</h3>
    
    The best metric depends on your specific needs:
    
    🔹 **Use MAE when:**
    - You need interpretability
    - Outliers are not very important
    
    🔹 **Use MSE/RMSE when:**
    - You want to penalize large errors
    - Your data scale is important
    
    🔹 **Use R² when:**
    - You want to compare different models
    - You need a scale-free metric
    </div>
    """, unsafe_allow_html=True)

elif page == "📈 Advanced Concepts":
    st.header("Advanced Regression Concepts")
    
    concept = st.selectbox(
        "Choose a concept to explore",
        ["Bias-Variance Tradeoff", "Overfitting vs Underfitting", "Feature Scaling", "Cross-Validation"]
    )
    
    if concept == "Bias-Variance Tradeoff":
        st.markdown("""
        <div class='concept-box'>
        <h3>Bias-Variance Tradeoff Explained</h3>
        Think of it like learning to play darts:
        - **High Bias**: Always throwing to the same spot (but missing the bullseye)
        - **High Variance**: Throws scattered all over the board
        - **Good Balance**: Consistent throws near the bullseye
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive demonstration
        complexity = st.slider("Model Complexity", 1, 10, 1)
        X, y = generate_sample_data(100, 0.5, 'polynomial')
        
        # Fit polynomial regression with different degrees
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X, y, alpha=0.5, label='Data')
        
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        for degree in [1, complexity]:
            X_poly = np.hstack([X_line ** i for i in range(1, degree + 1)])
            model = LinearRegression()
            model.fit(np.hstack([X ** i for i in range(1, degree + 1)]), y)
            y_poly = model.predict(X_poly)
            ax.plot(X_line, y_poly, label=f'Degree {degree}')
        
        ax.legend()
        ax.set_title("Polynomial Regression with Different Degrees")
        st.pyplot(fig)

    # Add similar detailed sections for other advanced concepts...

st.sidebar.markdown("""
---
### 📝 Learning Tips
- Take your time with each section
- Experiment with different parameters
- Pay attention to how changes affect the results
- Try to predict what will happen before making changes
""")
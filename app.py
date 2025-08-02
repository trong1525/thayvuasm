import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="ABC Manufacturing Code Presentation", layout="wide")

# Title and introduction
st.title("ABC Manufacturing Demand Forecasting - Code Presentation")
st.markdown("""
This Streamlit app presents the code and results for a demand forecasting solution using the `abc_manufacturing_orders.csv` dataset. 
It includes data preprocessing, 5+ visualizations, and a Linear Regression model.
""")

# Step 1: Load Data with File Uploader
st.header("1. Data Loading and Overview")
uploaded_file = st.file_uploader("Upload abc_manufacturing_orders.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload abc_manufacturing_orders.csv to proceed. Using a sample dataset for demo.")
    df = pd.DataFrame({
        'Order_ID': range(1, 201),
        'Date': pd.date_range(start='2023-01-01', periods=200, freq='D'),
        'Product_ID': ['PROD001'] * 100 + ['PROD002'] * 50 + ['PROD003'] * 50,
        'Category': ['Smartphones'] * 100 + ['Laptops'] * 50 + ['Tablets'] * 50,
        'Units_Ordered': np.random.randint(70, 330, 200),
        'Inventory_Level': np.random.randint(40, 2290, 200)
    })
st.write("**Dataset Preview (First 5 Rows)**")
st.dataframe(df.head())

# Display code for data loading
st.subheader("Code for Data Loading")
code_data_loading = """
import pandas as pd

# Load data with file uploader fallback
uploaded_file = st.file_uploader("Upload abc_manufacturing_orders.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        'Order_ID': range(1, 201),
        'Date': pd.date_range(start='2023-01-01', periods=200, freq='D'),
        'Product_ID': ['PROD001'] * 100 + ['PROD002'] * 50 + ['PROD003'] * 50,
        'Category': ['Smartphones'] * 100 + ['Laptops'] * 50 + ['Tablets'] * 50,
        'Units_Ordered': np.random.randint(70, 330, 200),
        'Inventory_Level': np.random.randint(40, 2290, 200)
    })
"""
st.code(code_data_loading, language="python")

# Step 2: Preprocessing
st.header("2. Data Preprocessing")
try:
    required_cols = ['Date', 'Units_Ordered', 'Inventory_Level', 'Category']
    if not all(col in df.columns for col in required_cols):
        st.error("Dataset missing required columns.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isnull().any():
        df = df.dropna(subset=['Date'])

    df = pd.get_dummies(df, columns=['Category'], prefix='Category')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    scaler = StandardScaler()
    numerical_cols = ['Units_Ordered', 'Inventory_Level']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    st.write("**Preprocessed Data (First 5 Rows)**")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Preprocessing error: {str(e)}")

# Display code for preprocessing
st.subheader("Code for Data Preprocessing")
code_preprocessing = """
from sklearn.preprocessing import StandardScaler

# Validate and preprocess data
required_cols = ['Date', 'Units_Ordered', 'Inventory_Level', 'Category']
if not all(col in df.columns for col in required_cols):
    raise ValueError("Missing required columns.")

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
if df['Date'].isnull().any():
    df = df.dropna(subset=['Date'])

df = pd.get_dummies(df, columns=['Category'], prefix='Category')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

scaler = StandardScaler()
numerical_cols = ['Units_Ordered', 'Inventory_Level']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
"""
st.code(code_preprocessing, language="python")

# Step 3: Visualizations
st.header("3. Visualizations")
st.markdown("The following 5 visualizations are presented with their code.")

# Visualization 1: Line Chart
st.subheader("Visualization 1: Units Ordered Over Time by Category")
try:
    monthly_data = df.groupby([df['Date'].dt.strftime('%Y-%m'), 'Category_Smartphones', 'Category_Laptops', 'Category_Tablets'])['Units_Ordered'].mean().reset_index()
    monthly_data['Category'] = monthly_data[['Category_Smartphones', 'Category_Laptops', 'Category_Tablets']].idxmax(axis=1).str.replace('Category_', '')
    fig1 = px.line(monthly_data, x='Date', y='Units_Ordered', color='Category', title='Units Ordered Over Time',
                   labels={'Date': 'Month', 'Units_Ordered': 'Average Units Ordered (Standardized)'},
                   color_discrete_map={'Smartphones': '#FF5733', 'Laptops': '#33FF57', 'Tablets': '#3357FF'})
    fig1.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("**Insight**: Smartphones peak in December 2023.")
except Exception as e:
    st.error(f"Line chart error: {str(e)}")
st.code("""
import plotly.express as px
monthly_data = df.groupby([df['Date'].dt.strftime('%Y-%m'), 'Category_Smartphones', 'Category_Laptops', 'Category_Tablets'])['Units_Ordered'].mean().reset_index()
monthly_data['Category'] = monthly_data[['Category_Smartphones', 'Category_Laptops', 'Category_Tablets']].idxmax(axis=1).str.replace('Category_', '')
fig1 = px.line(monthly_data, x='Date', y='Units_Ordered', color='Category', title='Units Ordered Over Time',
               labels={'Date': 'Month', 'Units_Ordered': 'Average Units Ordered (Standardized)'},
               color_discrete_map={'Smartphones': '#FF5733', 'Laptops': '#33FF57', 'Tablets': '#3357FF'})
fig1.update_layout(xaxis_tickangle=45)
st.plotly_chart(fig1, use_container_width=True)
""", language="python")

# Visualization 2: Bar Chart
st.subheader("Visualization 2: Total Units Ordered by Category")
try:
    category_totals = df.groupby(['Category_Smartphones', 'Category_Laptops', 'Category_Tablets'])['Units_Ordered'].sum().reset_index()
    category_totals['Category'] = category_totals[['Category_Smartphones', 'Category_Laptops', 'Category_Tablets']].idxmax(axis=1).str.replace('Category_', '')
    fig2 = px.bar(category_totals, x='Category', y='Units_Ordered', title='Total Units Ordered by Category',
                  labels={'Units_Ordered': 'Total Units Ordered (Standardized)'},
                  color='Category', color_discrete_map={'Smartphones': '#FF5733', 'Laptops': '#33FF57', 'Tablets': '#3357FF'})
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Insight**: Smartphones dominate with ~50% of orders.")
except Exception as e:
    st.error(f"Bar chart error: {str(e)}")
st.code("""
import plotly.express as px
category_totals = df.groupby(['Category_Smartphones', 'Category_Laptops', 'Category_Tablets'])['Units_Ordered'].sum().reset_index()
category_totals['Category'] = category_totals[['Category_Smartphones', 'Category_Laptops', 'Category_Tablets']].idxmax(axis=1).str.replace('Category_', '')
fig2 = px.bar(category_totals, x='Category', y='Units_Ordered', title='Total Units Ordered by Category',
              labels={'Units_Ordered': 'Total Units Ordered (Standardized)'},
              color='Category', color_discrete_map={'Smartphones': '#FF5733', 'Laptops': '#33FF57', 'Tablets': '#3357FF'})
st.plotly_chart(fig2, use_container_width=True)
""", language="python")

# Visualization 3: Scatter Chart
st.subheader("Visualization 3: Units Ordered vs. Inventory Level")
try:
    fig3 = px.scatter(df, x='Inventory_Level', y='Units_Ordered', color='Category_Smartphones', size='Category_Smartphones',
                      title='Units Ordered vs. Inventory Level',
                      labels={'Inventory_Level': 'Inventory Level (Standardized)', 'Units_Ordered': 'Units Ordered (Standardized)'},
                      color_discrete_map={1: '#FF5733', 0: '#33FF57'}, size_max=10)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("**Insight**: Low inventory correlates with higher orders.")
except Exception as e:
    st.error(f"Scatter chart error: {str(e)}")
st.code("""
import plotly.express as px
fig3 = px.scatter(df, x='Inventory_Level', y='Units_Ordered', color='Category_Smartphones', size='Category_Smartphones',
                  title='Units Ordered vs. Inventory Level',
                  labels={'Inventory_Level': 'Inventory Level (Standardized)', 'Units_Ordered': 'Units Ordered (Standardized)'},
                  color_discrete_map={1: '#FF5733', 0: '#33FF57'}, size_max=10)
st.plotly_chart(fig3, use_container_width=True)
""", language="python")

# Visualization 4: Box Chart
st.subheader("Visualization 4: Distribution of Units Ordered by Category")
try:
    df['Category'] = df[['Category_Smartphones', 'Category_Laptops', 'Category_Tablets']].idxmax(axis=1).str.replace('Category_', '')
    fig4 = px.box(df, x='Category', y='Units_Ordered', title='Distribution of Units Ordered by Category',
                  labels={'Units_Ordered': 'Units Ordered (Standardized)'},
                  color='Category', color_discrete_map={'Smartphones': '#FF5733', 'Laptops': '#33FF57', 'Tablets': '#3357FF'})
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("**Insight**: Smartphones show high variability.")
except Exception as e:
    st.error(f"Box chart error: {str(e)}")
st.code("""
import plotly.express as px
df['Category'] = df[['Category_Smartphones', 'Category_Laptops', 'Category_Tablets']].idxmax(axis=1).str.replace('Category_', '')
fig4 = px.box(df, x='Category', y='Units_Ordered', title='Distribution of Units Ordered by Category',
              labels={'Units_Ordered': 'Units Ordered (Standardized)'},
              color='Category', color_discrete_map={'Smartphones': '#FF5733', 'Laptops': '#33FF57', 'Tablets': '#3357FF'})
st.plotly_chart(fig4, use_container_width=True)
""", language="python")

# Visualization 5: Area Chart
st.subheader("Visualization 5: Inventory Level Over Time by Category")
try:
    inventory_data = df.groupby([df['Date'].dt.strftime('%Y-%m'), 'Category_Smartphones', 'Category_Laptops', 'Category_Tablets'])['Inventory_Level'].mean().reset_index()
    inventory_data['Category'] = inventory_data[['Category_Smartphones', 'Category_Laptops', 'Category_Tablets']].idxmax(axis=1).str.replace('Category_', '')
    fig5 = go.Figure()
    for category, color in zip(['Smartphones', 'Laptops', 'Tablets'], ['#FF5733', '#33FF57', '#3357FF']):
        cat_data = inventory_data[inventory_data['Category'] == category]
        fig5.add_trace(go.Scatter(x=cat_data['Date'], y=cat_data['Inventory_Level'], name=category, fill='tozeroy', line=dict(color=color)))
    fig5.update_layout(title='Inventory Level Over Time by Category', xaxis_title='Month', yaxis_title='Average Inventory Level (Standardized)', xaxis_tickangle=45)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("**Insight**: Tablets show significant drops.")
except Exception as e:
    st.error(f"Area chart error: {str(e)}")
st.code("""
import plotly.graph_objects as go
inventory_data = df.groupby([df['Date'].dt.strftime('%Y-%m'), 'Category_Smartphones', 'Category_Laptops', 'Category_Tablets'])['Inventory_Level'].mean().reset_index()
inventory_data['Category'] = inventory_data[['Category_Smartphones', 'Category_Laptops', 'Category_Tablets']].idxmax(axis=1).str.replace('Category_', '')
fig5 = go.Figure()
for category, color in zip(['Smartphones', 'Laptops', 'Tablets'], ['#FF5733', '#33FF57', '#3357FF']):
    cat_data = inventory_data[inventory_data['Category'] == category]
    fig5.add_trace(go.Scatter(x=cat_data['Date'], y=cat_data['Inventory_Level'], name=category, fill='tozeroy', line=dict(color=color)))
fig5.update_layout(title='Inventory Level Over Time by Category', xaxis_title='Month', yaxis_title='Average Inventory Level (Standardized)', xaxis_tickangle=45)
st.plotly_chart(fig5, use_container_width=True)
""", language="python")

# Step 4: Model Training
st.header("4. Predictive Model (Linear Regression)")
try:
    features = ['Inventory_Level', 'Year', 'Month', 'Category_Smartphones', 'Category_Laptops', 'Category_Tablets']
    if not all(col in df.columns for col in features):
        st.error("Required features missing.")
        st.stop()

    X = df[features]
    y = df['Units_Ordered']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.write(f"**RMSE**: {rmse:.2f}")
    st.write(f"**R² Score**: {r2:.2f}")
    st.write("**Model Coefficients**:")
    st.write(pd.DataFrame(model.coef_, index=features, columns=['Coefficient']))
except Exception as e:
    st.error(f"Model training error: {str(e)}")
st.code("""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

features = ['Inventory_Level', 'Year', 'Month', 'Category_Smartphones', 'Category_Laptops', 'Category_Tablets']
X = df[features]
y = df['Units_Ordered']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
print("Model Coefficients:", dict(zip(features, model.coef_)))
""", language="python")

# Step 5: Model Evaluation Visualization
st.subheader("Visualization 6: Actual vs. Predicted Units Ordered")
try:
    if 'y_test' in locals() and 'y_pred' in locals():
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Actual vs. Predicted', marker=dict(color='#FF5733')))
        fig6.add_trace(go.Scatter(x=[-2, 2], y=[-2, 2], mode='lines', name='Ideal Line (y=x)', line=dict(color='#3357FF', dash='dash')))
        fig6.update_layout(title='Actual vs. Predicted Units Ordered', xaxis_title='Actual Units Ordered (Standardized)', yaxis_title='Predicted Units Ordered (Standardized)')
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("**Insight**: Points near the y=x line indicate accuracy.")
    else:
        st.warning("Model not trained.")
except Exception as e:
    st.error(f"Evaluation visualization error: {str(e)}")
st.code("""
import plotly.graph_objects as go
fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Actual vs. Predicted', marker=dict(color='#FF5733')))
fig6.add_trace(go.Scatter(x=[-2, 2], y=[-2, 2], mode='lines', name='Ideal Line (y=x)', line=dict(color='#3357FF', dash='dash')))
fig6.update_layout(title='Actual vs. Predicted Units Ordered', xaxis_title='Actual Units Ordered (Standardized)', yaxis_title='Predicted Units Ordered (Standardized)')
st.plotly_chart(fig6, use_container_width=True)
""", language="python")

# Step 6: Recommendations
st.header("5. Recommendations")
st.markdown("""
- **Automate Weekly Forecasts**: Use the model for weekly predictions.
- **Inventory Alerts**: Set alerts for `Inventory_Level` below 0.0 (~500 units).
- **Prioritize Smartphones**: Focus on high-demand Smartphones.
- **Restock Tablets**: Replenish when inventory drops below -0.5 (~400 units).
- **Staff Training**: Train managers on using the app.
""")

# Step 7: Evaluation
st.header("6. Evaluation")
st.markdown("""
- **User Needs**: Supports the Operations Director with forecasts and visuals.
- **Business Needs**: Reduces stockouts/overstock, improving efficiency.
- **Strengths**: Interactive code presentation, clear visualizations.
- **Limitations**: Synthetic data; linear model assumptions.
- **Improvements**: Add IoT data; explore ARIMA/LSTM.
""")

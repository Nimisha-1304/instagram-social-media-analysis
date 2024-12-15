import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load Instagram data from local file
@st.cache_data
def load_data():
    file_path = 'Instagram data.csv'  # Replace with your actual file path
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    return data

# Calculate engagement rate
def calculate_engagement_rate(df):
    df['Engagement Rate'] = (df['Likes'] + df['Comments'] + df['Shares']) / df['Impressions'] * 100
    return df

# Advanced Visualizations - Interactive with Plotly
def plot_visualizations(df):
    st.subheader("Interactive Visualizations")

    # Impressions Breakdown Interactive Pie Chart
    st.subheader("Impressions Breakdown by Source")
    sources = ['From Home', 'From Hashtags', 'From Explore', 'From Other']
    source_sums = df[sources].sum()
    fig = px.pie(values=source_sums.values, names=source_sums.index, title="Impressions Breakdown by Source")
    st.plotly_chart(fig)

    # Bar chart for Likes vs Engagement Rate
    st.subheader("Likes vs Engagement Rate")
    fig = px.bar(df, x='Likes', y='Engagement Rate', title="Likes vs Engagement Rate", color='Engagement Rate')
    st.plotly_chart(fig)

    # Line chart for trend of Likes over different post types
    st.subheader("Trend of Likes")
    likes_over_time = df.groupby('Caption').agg({'Likes': 'sum'}).reset_index()
    fig = px.line(likes_over_time, x='Caption', y='Likes', title="Total Likes by Post Type")
    st.plotly_chart(fig)

    # Scatter plot for Comments vs Likes
    st.subheader("Comments vs Likes")
    fig = px.scatter(df, x='Comments', y='Likes', title="Comments vs Likes", trendline="ols")
    st.plotly_chart(fig)

    # Distribution of Engagement Rate
    st.subheader("Distribution of Engagement Rate")
    fig = px.histogram(df, x='Engagement Rate', nbins=50, title="Distribution of Engagement Rate", color_discrete_sequence=['indigo'])
    st.plotly_chart(fig)

    # Bar chart for Impressions by Post Type
    st.subheader("Impressions by Post Type")
    impressions_by_caption = df.groupby('Caption').agg({'Impressions': 'sum'}).reset_index()
    fig = px.bar(impressions_by_caption, x='Caption', y='Impressions', title="Total Impressions by Post Type", color='Impressions')
    st.plotly_chart(fig)

    # Stacked bar chart for Source Breakdown
    st.subheader("Impressions by Source Breakdown")
    fig = px.bar(df, x='Caption', y=sources, title="Impressions by Source Breakdown", 
                 labels={"value": "Impressions", "Caption": "Post Type"}, 
                 color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig)

    # Box plot for Likes Distribution
    st.subheader("Likes Distribution")
    fig = px.box(df, y='Likes', title="Distribution of Likes", color='Caption')
    st.plotly_chart(fig)

    # Heatmap for Correlation between Metrics
    st.subheader("Correlation Heatmap between Metrics")
    fig = px.imshow(df[['Likes', 'Comments', 'Shares', 'Profile Visits', 'Engagement Rate', 'Impressions']].corr(),
                    color_continuous_scale='RdBu_r', text_auto=True, title='Correlation Matrix')
    st.plotly_chart(fig)

# Predict Engagement Rate using multiple models and cross-validation
def predict_engagement(df, model_type):
    features = df[['Impressions', 'Saves', 'Comments', 'Shares', 'Profile Visits']]
    target = df['Engagement Rate']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'Decision Tree':
        model = DecisionTreeRegressor(random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write(f"### Model Performance ({model_type})")
    st.write(f"Root Mean Square Error: {rmse:.2f}")
    st.write(f"R-squared: {r2:.2f}")

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    st.write(f"Cross-Validation Score: {np.mean(cv_scores):.2f}")
    
    return model

# Simulate prediction for a new post
def simulate_new_post(model):
    st.sidebar.subheader("Simulate a New Post")
    impressions = st.sidebar.number_input("Impressions", min_value=0, value=5000, step=500)
    saves = st.sidebar.number_input("Saves", min_value=0, value=50, step=10)
    comments = st.sidebar.number_input("Comments", min_value=0, value=20, step=5)
    shares = st.sidebar.number_input("Shares", min_value=0, value=10, step=2)
    profile_visits = st.sidebar.number_input("Profile Visits", min_value=0, value=30, step=5)

    new_data = pd.DataFrame({
        'Impressions': [impressions],
        'Saves': [saves],
        'Comments': [comments],
        'Shares': [shares],
        'Profile Visits': [profile_visits]
    })

    predicted_engagement = model.predict(new_data)
    st.sidebar.write(f"### Predicted Engagement Rate: {predicted_engagement[0]:.2f}%")

# Main Streamlit app
def main():
    st.title("📊 Instagram Insights & Advanced Prediction Tool")
    st.markdown("""This tool provides interactive visual analytics and machine learning predictions for Instagram post engagement. You can explore key metrics, visualize data with interactive charts, and simulate future post performance.""")

    # Load data
    data = load_data()

    # Calculate engagement rate
    data = calculate_engagement_rate(data)

    # Sidebar options for selecting visualizations and models
    st.sidebar.title("🔍 Insights Options")
    show_visualizations = st.sidebar.checkbox("Show Interactive Visualizations", value=True)
    top_posts_metric = st.sidebar.selectbox("Top Posts By", ["Likes", "Comments", "Shares", "Engagement Rate"])
    top_n = st.sidebar.slider("Number of Top Posts to Display", 1, 10, 5)

    model_type = st.sidebar.selectbox("Choose Machine Learning Model", ["Linear Regression", "Random Forest", "Decision Tree"])

    # Show visualizations
    if show_visualizations:
        plot_visualizations(data)

    # Machine Learning: Engagement Rate Prediction
    model = predict_engagement(data, model_type)
    
    # Simulate new post performance
    simulate_new_post(model)

    # Show Top Posts by Selected Metric
    st.subheader(f"Top {top_n} Posts by {top_posts_metric}")
    show_top_posts(data, top_posts_metric, top_n)

# Show top posts by the selected metric
def show_top_posts(df, metric, n):
    top_posts = df.sort_values(by=metric, ascending=False).head(n)
    st.write(top_posts[['Caption', metric]])

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from PIL import Image
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# Page config
st.set_page_config(page_title="GreenScore", layout="wide")

# Custom CSS for green + white theme
st.markdown("""
    <style>
        body { background-color: #ffffff; color: #1a7f37; }
        .main { background-color: #ffffff; }
        h1, h2, h3, h4, h5, h6 { color: #1a7f37; }
        .stButton>button { background-color: #1a7f37; color: white; }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🌿 GreenScore")
st.write("Analyze and rank products by sustainability, cost, and carbon footprint.")

# Sidebar – Upload CSV
st.sidebar.header("📁 Upload Product CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load Data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("sample_products_massive.csv")

# Ensure correct data types
numeric_columns = ['Carbon Footprint (kg CO2)', 'Sustainability Score', 'Price ($)']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows with NaNs in numeric columns
df = df.dropna(subset=numeric_columns)

# Calculate GreenScore
df['GreenScore'] = df['Sustainability Score'] / df['Price ($)']

# Sidebar Filters
st.sidebar.header("🔍 Filter Products")

selected_materials = st.sidebar.multiselect(
    "Select Material(s):",
    options=df['Material'].unique(),
    default=df['Material'].unique()
)

selected_types = st.sidebar.multiselect(
    "Select Type(s):",
    options=df['Type'].unique(),
    default=df['Type'].unique()
)

max_carbon = st.sidebar.slider(
    "Max Carbon Footprint (kg CO2)",
    float(df['Carbon Footprint (kg CO2)'].min()),
    float(df['Carbon Footprint (kg CO2)'].max()),
    float(df['Carbon Footprint (kg CO2)'].max())
)

min_sustainability = st.sidebar.slider(
    "Min Sustainability Score",
    float(df['Sustainability Score'].min()),
    float(df['Sustainability Score'].max()),
    float(df['Sustainability Score'].min())
)

min_price, max_price = st.sidebar.slider(
    "Price Range ($)",
    float(df['Price ($)'].min()),
    float(df['Price ($)'].max()),
    (float(df['Price ($)'].min()), float(df['Price ($)'].max()))
)

# Filtered Data
filtered_df = df[
    (df['Material'].isin(selected_materials)) &
    (df['Type'].isin(selected_types)) &
    (df['Carbon Footprint (kg CO2)'] <= max_carbon) &
    (df['Sustainability Score'] >= min_sustainability) &
    (df['Price ($)'] >= min_price) &
    (df['Price ($)'] <= max_price)
]

sorted_df = filtered_df.sort_values(by='GreenScore', ascending=False)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Dashboard", "🔍 Product Comparison", "✨ Recommendations", "📷 Image Match"])

with tab1:
    st.subheader("📋 Ranked Products")
    st.dataframe(sorted_df[['Product', 'Material', 'Type', 'Sustainability Score', 'Price ($)', 'Carbon Footprint (kg CO2)', 'GreenScore']])

    st.subheader("📊 Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg GreenScore", f"{np.mean(sorted_df['GreenScore']):.2f}")
    with col2:
        st.metric("Lowest Carbon Footprint", f"{np.min(sorted_df['Carbon Footprint (kg CO2)']):.2f} kg")
    with col3:
        st.metric("Highest Sustainability", f"{np.max(sorted_df['Sustainability Score']):.2f}/10")

    st.subheader("📉 Visual Insights")

    bar = px.bar(
        sorted_df, x='Product', y='GreenScore', color='Material',
        title="GreenScore by Product",
        color_discrete_sequence=px.colors.sequential.Greens
    )
    st.plotly_chart(bar, use_container_width=True)

    scatter = px.scatter(
        sorted_df, x='Price ($)', y='Sustainability Score',
        color='Material', size='GreenScore', hover_data=['Product'],
        title="Sustainability vs Price",
        color_discrete_sequence=px.colors.sequential.Greens
    )
    st.plotly_chart(scatter, use_container_width=True)

    st.subheader("💰 Price Distribution")
    hist = px.histogram(
        sorted_df, x='Price ($)', nbins=12, color_discrete_sequence=['#1a7f37'],
        title="Distribution of Product Prices"
    )
    st.plotly_chart(hist, use_container_width=True)

    st.subheader("📌 Avg Price by Material")
    avg_price_df = (
        sorted_df.groupby('Material')['Price ($)']
        .mean()
        .reset_index()
        .sort_values(by='Price ($)', ascending=False)
    )
    avg_price_chart = px.bar(
        avg_price_df, x='Material', y='Price ($)', color='Material',
        title="Average Price per Material",
        color_discrete_sequence=px.colors.sequential.Greens
    )
    st.plotly_chart(avg_price_chart, use_container_width=True)

with tab2:
    st.subheader("🔍 Product Comparison & Review")
    comparison_options = sorted_df['Product'].tolist()

    product1 = st.selectbox("Select Product 1", comparison_options)
    product2 = st.selectbox("Select Product 2", comparison_options)

    product1_details = sorted_df[sorted_df['Product'] == product1].iloc[0]
    product2_details = sorted_df[sorted_df['Product'] == product2].iloc[0]

    def sustainability_review(sustainability_score, carbon_footprint):
        if sustainability_score > 8 and carbon_footprint < 2:
            return "🌱 Excellent sustainability! Low carbon footprint and eco-friendly materials."
        elif sustainability_score > 6 and carbon_footprint < 3:
            return "✅ Good sustainability, but could be improved with lower carbon footprint."
        elif sustainability_score > 5:
            return "⚠️ Average sustainability. Consider alternatives with better carbon footprint."
        else:
            return "❌ Needs improvement. High carbon footprint and questionable materials."

    def budget_friendly_review(price, green_score):
        if price < 10 and green_score > 1:
            return "💸 Budget-friendly and offers good green value!"
        elif price < 15:
            return "👌 Affordable, but there might be better eco-value alternatives."
        else:
            return "💰 A bit expensive. Consider if it's worth the price for its eco-impact."

    comparison_df = pd.DataFrame({
        'Attribute': ['Price ($)', 'Sustainability Score', 'Carbon Footprint (kg CO2)', 'GreenScore', 'Sustainability Review', 'Budget-Friendly Review'],
        product1: [
            product1_details['Price ($)'],
            product1_details['Sustainability Score'],
            product1_details['Carbon Footprint (kg CO2)'],
            product1_details['GreenScore'],
            sustainability_review(product1_details['Sustainability Score'], product1_details['Carbon Footprint (kg CO2)']),
            budget_friendly_review(product1_details['Price ($)'], product1_details['GreenScore'])
        ],
        product2: [
            product2_details['Price ($)'],
            product2_details['Sustainability Score'],
            product2_details['Carbon Footprint (kg CO2)'],
            product2_details['GreenScore'],
            sustainability_review(product2_details['Sustainability Score'], product2_details['Carbon Footprint (kg CO2)']),
            budget_friendly_review(product2_details['Price ($)'], product2_details['GreenScore'])
        ]
    })

    st.write(comparison_df)

with tab3:
    st.subheader("✨ Recommended Products")
    recommendations = {
        "🌿 Eco-Friendly Choices": sorted_df[(sorted_df['Sustainability Score'] >= 8) & (sorted_df['Carbon Footprint (kg CO2)'] < 2)],
        "💲 Budget Picks": sorted_df[sorted_df['Price ($)'] <= 10],
        "🏆 Top Rated": sorted_df.sort_values(by='GreenScore', ascending=False).head(5)
    }

    for category, subset in recommendations.items():
        with st.expander(category):
            st.table(subset[['Product', 'Type', 'Price ($)', 'Sustainability Score', 'Carbon Footprint (kg CO2)', 'GreenScore']])

with tab4:
    st.subheader("📷 Image-Based Product Match")
    uploaded_img = st.file_uploader("Upload an image of the product", type=["jpg", "png", "jpeg"])

    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        search_query = st.text_input("Enter a description of the product for search (e.g., bamboo toothbrush)")

        if search_query:
            st.info("🔎 Searching for matching product in dataset...")
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(df['Product'])
            query_vec = vectorizer.transform([search_query])
            similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_index = np.argmax(similarity_scores)

            matched_product = df.iloc[top_index]
            st.success(f"✅ Best Match: {matched_product['Product']}")
            st.write(matched_product[['Product', 'Material', 'Type', 'Price ($)', 'Sustainability Score', 'Carbon Footprint (kg CO2)', 'GreenScore']])

            st.markdown("---")
            st.write("### 🔍 GreenScore Details")
            st.metric("GreenScore", f"{matched_product['GreenScore']:.2f}")
            st.metric("Sustainability Score", f"{matched_product['Sustainability Score']:.1f}/10")
            st.metric("Carbon Footprint", f"{matched_product['Carbon Footprint (kg CO2)']:.2f} kg")
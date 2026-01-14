import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="MotoPredict Pro", page_icon="🏎️", layout="wide")

# --- ADVANCED CUSTOM CSS ---
st.markdown("""
    <style>
    /* Main Background with a subtle gradient */
    .stApp {
        background: radial-gradient(circle, #1a1a1a 0%, #000000 100%);
        color: #e0e0e0;
    }

    /* Glassmorphism Card Effect */
    .prediction-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* Neon Text Effect for Result */
    .mpg-display {
        font-size: 80px;
        font-weight: 900;
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    /* Styling buttons and sliders */
    .stSlider [data-baseweb="slider"] { margin-bottom: 20px; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(20, 20, 20, 0.8);
    }
    
    h1, h2, h3 { color: #4facfe !important; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_resource
def init_model():
    # Loading the MT-Cars dataset (Used in your project screenshots)
    url = "https://gist.githubusercontent.com/seankross/a16248f7ad8b0a8dabd6/raw/14350a2416998a4498334f08d151829f7974eb82/mtcars.csv"
    df = pd.read_csv(url)
    # Feature: Weight (wt), Target: MPG
    X = df[['wt']]
    y = df['mpg']
    model = LinearRegression()
    model.fit(X, y)
    return model, df

model, df = init_model()

# --- SIDEBAR DESIGN ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/741/741407.png", width=100)
st.sidebar.title("MotoPredict Pro")
st.sidebar.markdown("---")
st.sidebar.write("⚡ **Model:** Linear Regression")
st.sidebar.write("📊 **Dataset:** MT-Cars")
st.sidebar.markdown("---")
st.sidebar.info("Adjust the car weight to see how fuel efficiency drops as mass increases.")

# --- MAIN INTERFACE ---
st.title("🏎️ Car Intelligence Dashboard")
st.write("Real-time Mileage Analytics & Predictive Modeling")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### ⚙️ Vehicle Specs")
    st.write("Set the weight of the vehicle (1,000s of lbs)")
    
    # Custom interactive slider
    input_wt = st.slider("", 
                        min_value=float(df.wt.min()), 
                        max_value=float(df.wt.max()), 
                        value=3.2, 
                        step=0.1)
    
    # Run Prediction
    prediction = model.predict([[input_wt]])[0]
    
    # Display Result in Custom Card
    st.markdown(f"""
        <div class="prediction-container">
            <p style="letter-spacing: 3px; font-size: 14px; color: #888;">PREDICTED EFFICIENCY</p>
            <h1 class="mpg-display">{prediction:.1f}</h1>
            <p style="font-size: 18px; color: #4facfe;">MILES PER GALLON</p>
        </div>
    """, unsafe_allow_html=True)

    # Dynamic "Economy Rating"
    if prediction > 25:
        st.success("Rating: **Eco-Friendly** 🌱")
    elif prediction > 15:
        st.warning("Rating: **Moderate Consumer** ⛽")
    else:
        st.error("Rating: **Gas Guzzler** ⚠️")

with col2:
    st.markdown("### 📈 Visual Data Analysis")
    
    # Plotly Scatter Chart with Regression Line
    fig = px.scatter(df, x="wt", y="mpg", 
                     hover_name="model", 
                     template="plotly_dark",
                     color="mpg",
                     color_continuous_scale="Viridis",
                     labels={"wt": "Weight (1000 lbs)", "mpg": "MPG"})
    
    # Add the current prediction point as a glowing star
    fig.add_trace(go.Scatter(
        x=[input_wt], y=[prediction],
        mode='markers+text',
        marker=dict(color='#ff00ff', size=18, symbol='star', line=dict(width=2, color="white")),
        name="Current Setup",
        text=["YOU ARE HERE"],
        textposition="top center"
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- LOWER DASHBOARD ---
st.markdown("---")
tab1, tab2 = st.tabs(["📊 Correlation Matrix", "📋 Dataset View"])

with tab1:
    st.write("Understanding the relationship between engine variables.")
    # Show only numeric columns for correlation
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    fig_heat = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)

with tab2:
    st.dataframe(df.style.background_gradient(cmap='Blues'), use_container_width=True)

# --- FOOTER ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.caption("Developed by Ragul Saravanan | Machine Learning Engine v1.0")

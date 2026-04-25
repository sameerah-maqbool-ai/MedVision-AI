"""
CHEST X-RAY DIAGNOSIS SYSTEM - WEB APPLICATION
Run with: streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import os
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Chest X-Ray Diagnosis System",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .normal-result {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    }
    .pneumonia-result {
        background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
    }
    .tuberculosis-result {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.5rem 2rem;
        border-radius: 30px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained model"""
    model_paths = [
        'models/chest_xray_final_model.keras',
        'models/chest_xray_final_model.h5',
        'models/best_model.keras',
        'chest_xray_final_model.keras'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                return model, path
            except Exception as e:
                continue
    
    return None, None

# Class information
CLASSES = ['Normal', 'Pneumonia', 'Tuberculosis']
CLASS_INFO = {
    'Normal': {
        'description': '✅ No abnormalities detected. The chest X-ray appears normal.',
        'color': '#27ae60',
        'icon': '✅',
        'recommendation': 'Continue regular health check-ups and maintain a healthy lifestyle.'
    },
    'Pneumonia': {
        'description': '⚠️ Pneumonia detected. This is an infection that inflames air sacs in one or both lungs.',
        'color': '#e74c3c',
        'icon': '⚠️',
        'recommendation': 'Consult a pulmonologist immediately. Antibiotics may be prescribed. Rest and hydration are essential.'
    },
    'Tuberculosis': {
        'description': '⚠️ Tuberculosis detected. This is a serious bacterial infection that primarily affects the lungs.',
        'color': '#f39c12',
        'icon': '⚠️',
        'recommendation': 'Seek immediate medical attention. TB requires long-term antibiotic treatment under medical supervision.'
    }
}

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def create_prediction_chart(probabilities):
    """Create an interactive chart for prediction probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=CLASSES,
            y=probabilities,
            marker_color=['#27ae60', '#e74c3c', '#f39c12'],
            text=[f'{p:.1f}%' for p in probabilities],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Condition',
        yaxis_title='Confidence (%)',
        yaxis_range=[0, 100],
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_traces(marker_line_width=1, marker_line_color='white')
    
    return fig

def create_gauge_chart(confidence, class_name):
    """Create a gauge chart for confidence level"""
    color = CLASS_INFO[class_name]['color']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': f"Confidence for {class_name}"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 75], 'color': '#ffffcc'},
                {'range': [75, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        },
        number={'suffix': "%", 'font': {'size': 40}}
    ))
    
    fig.update_layout(height=300)
    
    return fig

def add_bg_image():
    """Add background image using base64"""
    # Create a subtle background pattern
    bg_style = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🩻 Chest X-Ray Diagnosis System</h1>
        <p style="color: white; margin-top: 10px; opacity: 0.9;">
            AI-Powered Medical Image Analysis | Detect Normal, Pneumonia, and Tuberculosis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/lungs.png", width=80)
        st.title("About")
        st.markdown("""
        ### 🩻 System Overview
        This AI system analyzes chest X-ray images to detect:
        
        - **Normal** - Healthy lungs
        - **Pneumonia** - Lung infection
        - **Tuberculosis** - Bacterial infection
        
        ### 📊 Model Information
        - Architecture: EfficientNetB0
        - Training: Transfer Learning + Fine-tuning
        - Input Size: 224 × 224 pixels
        
        ### ⚠️ Disclaimer
        This is an AI assistant tool. Always consult 
        with qualified medical professionals for 
        proper diagnosis and treatment.
        """)
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Tips")
        st.info("""
        - Use clear, frontal chest X-rays
        - Ensure good image quality
        - Supported formats: JPG, PNG, JPEG
        """)
    
    # Load model
    model, model_path = load_model()
    
    if model is None:
        st.error("""
        ❌ **Model not found!**
        
        Please train the model first by running:
        ```bash
        python train_model.py
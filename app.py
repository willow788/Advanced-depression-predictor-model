import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="DEPRESSION PREDICTOR",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling and Title ---
st.title("🧠 DEPRESSION PREDICTOR")
st.markdown(
    """
    Use the sliders and toggles in the sidebar to assess different dimensions of your current well-being.
    The interactive Radar Chart will update instantly to show your unique profile.
    """
)
st.divider()

# --- Feature Definitions ---
# Feature groups for organization
score_features = {
    "Sadness Score (Frequency/Intensity)": ("sadness_score", 1, 10),
    "Anxiety Score (Worry/Tension)": ("anxiety_score", 1, 10),
    "Fatigue Score (Energy Level)": ("fatigue_score", 1, 10),
}

binary_features = {
    "Do you experience significant sleep issues? 🌙": ("sleep_issues"),
    "Have you felt social withdrawal lately? 👤": ("social_withdrawal"),
    "Are you having trouble concentrating? 🤔": ("concentration_issues"),
    "Do you notice frequent, significant mood swings? 🎢": ("mood_swings"),
}

# --- Sidebar for Input (The Interactive Part) ---
st.sidebar.header("Input Your Scores (Scale: 1 = Low, 10 = High)")

user_inputs = {}

# 1. Score Inputs (using sliders)
st.sidebar.subheader("Emotional & Physical Metrics")
for label, (key, min_val, max_val) in score_features.items():
    user_inputs[key] = st.sidebar.slider(
        label,
        min_value=min_val,
        max_value=max_val,
        value=5, # Default value
        step=1,
        help=f"Rate this factor from {min_val} (very low) to {max_val} (very high)."
    )

# 2. Binary Inputs (using checkboxes)
st.sidebar.subheader("Behavioral & Cognitive Factors (Toggle if present)")
for label, key in binary_features.items():
    # Checkbox returns True/False. We convert this to 1 or 0 for the raw data.
    # The plot visualization handles the scaling to 10.
    is_present = st.sidebar.checkbox(label, value=False, key=f"check_{key}")
    user_inputs[key] = 1 if is_present else 0


# --- Data Processing for Radar Chart ---

# Map of raw feature keys and the display labels
all_labels_map = {
    "sadness_score": "Sadness",
    "anxiety_score": "Anxiety",
    "fatigue_score": "Fatigue",
    "sleep_issues": "Sleep Issues",
    "social_withdrawal": "Social Withdrawal",
    "concentration_issues": "Concentration",
    "mood_swings": "Mood Swings",
}

# The final list of labels for the chart axis
labels = list(all_labels_map.values())

# The final list of values, scaled to fit a 0-10 range for visual consistency
def scale_value(key, value):
    # Scale 0/1 binary scores to 0/10 for better visual impact on the 0-10 chart
    if key in [f[0] for f in binary_features.values()]:
        return value * 10
    # Score features are already 1-10
    return value

values = [scale_value(key, user_inputs[key]) for key in all_labels_map.keys()]

# Radar Chart Setup (based on the logic from prediction.ipynb)
# We need to close the loop for the radar plot
values += values[:1]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]
labels += labels[:1] # Include the first label again for plot clarity, but it won't be displayed twice.

# --- Radar Chart Visualization Function ---
def create_radar_chart(angles, values, labels):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot the data
    ax.plot(angles, values, color='#6B7280', linewidth=2, linestyle='solid', marker='o')
    ax.fill(angles, values, color='#10B981', alpha=0.3) # Tailwind emerald-500

    # Set the y-axis limit and ticks (0 to 10 for consistency)
    ax.set_ylim(0, 10)
    ax.set_yticks(np.arange(0, 11, 2.5)) # Gridlines at 0, 2.5, 5, 7.5, 10
    ax.tick_params(axis='y', colors='#4B5563') # Tailwind gray-600

    # Set the X-axis labels (feature names)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], color='#374151', size=12, weight='bold') # Tailwind gray-700

    # Remove radial lines and spines for a cleaner look
    ax.yaxis.grid(True, color='#D1D5DB') # Tailwind gray-300
    ax.spines['polar'].set_visible(False)
    ax.set_rlabel_position(0)

    # Title for the plot
    fig.suptitle("Radar Chart of your current mental health condition", size=16, color='#1F2937', weight='bold', y=1.0) # Tailwind gray-800
    
    # Set the overall background and face color
    fig.patch.set_facecolor('#F9FAFB')
    ax.set_facecolor('#F9FAFB')

    return fig

# --- Main Content Display ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("Depression-Predictor")
    # Calculate a simple aggregate score (Higher is less optimal well-being)
    total_score = sum(user_inputs.values())
    max_score = 3 * 10 + 4 * 1  # 3 score features (max 10) + 4 binary features (max 1)
    # Convert total score to a percentage of the maximum possible score
    severity_percentage = (total_score / max_score) * 100

    # Display a metric for overall profile
    st.metric(label="Total Profile Score (Normalized)",
              value=f"{total_score} / {max_score}",
              delta=f"{severity_percentage:.1f}% of Maximum Severity")

    # Interpretive message based on the severity percentage
    st.markdown("### Profile Summary")
    if severity_percentage < 30:
        st.success("Your profile indicates **High Well-being**! Keep up the great routines and self-care.")
        st.balloons()
    elif severity_percentage < 60:
        st.warning("Your profile indicates **Moderate Well-being** levels. Pay attention to the highest scored areas on your chart to maintain balance.")
    else:
        st.error("Your profile indicates a **Potential Area of Concern**. Please consult the chart to identify key areas (scores above 7 or checked factors) that may require professional support or immediate attention.")

    st.markdown("---")
    st.caption("Note: This visualization is for self-reflection only and is not a medical diagnostic tool.")


with col2:
    # Generate and display the chart
    radar_fig = create_radar_chart(angles, values, labels)
    st.pyplot(radar_fig)

# --- Explanation Footer ---
st.markdown("---")
st.markdown(
    """
    ### 📊 How to Read the Radar Chart:
    * **Axes:** Each spoke represents one well-being factor.
    * **Scale:** The center is 0 (Low Severity), and the outer edge is 10 (High Severity).
    * **Shape:** The shape of the shaded area illustrates your profile. A larger, more outward-reaching shape indicates higher scores across the factors.
    * **Spikes:** Look for noticeable 'spikes'—these are the areas where you rated yourself highest, suggesting where your focus should be.
    """
)

#my code gave so many error that i have to use gemini for this.
#the prediction.ipynb is all me, yay! that the main part honey!
#after 7hrs and 3hrs of battling the streamlit app errors, here it is
#ladies and gentlemen, bye bye!
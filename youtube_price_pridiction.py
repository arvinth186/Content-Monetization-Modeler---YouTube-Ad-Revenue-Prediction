import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
from googleapiclient.discovery import build
import isodate
import os
import plotly.express as px
import pandas as pd


# Load the saved Linear Regression model pipeline

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR,'LinearRegression.pkl')

try:
    model = jb.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ----------------------------------------Frontend UI----------------------------------------------

st.set_page_config(page_title="YouTube Ad Revenue Predictor 🎥💰",page_icon="🎬", layout="wide",initial_sidebar_state="expanded")
st.title("🎥 YouTube Ad Revenue Prediction App")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png", width=100)
    st.markdown("## 🎥 About the App")
    st.info("This app predicts YouTube Ad Revenue 💰 using machine learning. \n\n👉 Choose manual entry or paste a YouTube link!")

mode = st.sidebar.radio("Select Input Mode:", ("Manual Entry","YouTube Link","Insights"))


# ----------------------------------------Model Implementation----------------------------------------------

def make_prediction(input_dict):
    input_df = pd.DataFrame([input_dict])
    try:
        model_features = model.feature_names_in_
        input_df=input_df.reindex(columns=model_features, fill_value=0)
        prediction = model.predict(input_df)[0]
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None
    

# ----------------------------------------Manual Entry----------------------------------------------

if mode == "Manual Entry":
    st.header("Manual Input Mode")
    st.markdown("Enter the following details to predict YouTube Ad Revenue:")

    views = st.number_input("Total Views", min_value=0, value=0, step=1000)
    likes = st.number_input("Total Likes", min_value=0, value=0, step=10)
    comments = st.number_input("Total Comments", min_value=0, value=0, step=10)
    watch_time_minutes = st.number_input("Total Watch Time (in minutes)", min_value=0.0, value=0.0, step=10.0,format="%.2f")
    video_length_minutes = st.number_input("Video Length (in minutes)", min_value=0.0, value=0.0, step=0.1,format="%.2f")
    subscribers = st.number_input("Subscribers", min_value=0, step=100, value=10000)

    category = st.selectbox("Category", ["Entertainment", "Gaming", "Education", "Music", "Tech", "Lifestyle"])
    device = st.selectbox("Device", ["Mobile", "Desktop", "Tablet", "TV"])
    country = st.selectbox("Country", ["US", "IN", "UK", "CA", "DE", "AU"])

    if st.button("🔮 Predict Revenue"):
            engagement_rate = (likes + comments) / views if views > 0 else 0
            avg_watch_time_per_view = watch_time_minutes / views if views > 0 else 0

            input_dict = {

                "views": views,
                "likes": likes,
                "comments": comments,
                "watch_time_minutes": watch_time_minutes,
                "video_length_minutes": video_length_minutes,
                "subscribers": subscribers,
                "category": category,
                "device": device,
                "country": country,
                "engagement_rate": engagement_rate,
                "avg_watch_time_per_view": avg_watch_time_per_view
            }

            prediction = make_prediction(input_dict)
            if prediction is not None:
                st.success(f"💰 Estimated Ad Revenue: **${prediction:,.2f} USD**")
        

# ----------------------------------------YouTube Link Entry----------------------------------------------
 
elif mode == "YouTube Link":
    st.header("YouTube Link Input Mode")

    api_key = st.text_input("Enter your YouTube Data API Key:", type="password")
    youtube_url = st.text_input("Paste YouTube Video URL:")

    def extract_video_id(url:str)->str:
        if "watch?v=" in url:
            return url.split("watch?v=")[-1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[-1].split("?")[0]
        else:
            return None

    if st.button("Fetch & Predict"):
        if api_key and youtube_url:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please check and try again.")
            else:
                try:
                    youtube = build("youtube", "v3", developerKey=api_key)
                    request = youtube.videos().list(
                    part="snippet,statistics,contentDetails",
                        id=video_id
                    )
                    response = request.execute()

                    if "items" in response and len(response["items"]) > 0:
                        item = response["items"][0]
                        stats = item["statistics"]
                        snippet = item["snippet"]
                        content = item["contentDetails"]

                        st.success(f"✅ Video found: {snippet['title']}")

                        views = int(stats.get("viewCount", 0))
                        likes = int(stats.get("likeCount", 0))
                        comments = int(stats.get("commentCount", 0))
                        duration = content["duration"]
                        video_length_seconds = isodate.parse_duration(duration).total_seconds()
                        video_length_minutes = video_length_seconds / 60
                        mins, secs = divmod(int(video_length_seconds), 60)
                        formatted_duration = f"{mins}:{secs:02d}"

                        # Channel subscribers
                        channel_id = snippet["channelId"]
                        ch_request = youtube.channels().list(part="statistics", id=channel_id)
                        ch_response = ch_request.execute()
                        subscribers = int(ch_response["items"][0]["statistics"].get("subscriberCount", 0))

                        # Approximate watch time
                        watch_time_minutes = views * video_length_minutes

                        # Defaults (can be refined later)
                        category = "Entertainment"
                        device = "Mobile"
                        country = "US"

                        engagement_rate = (likes + comments) / views if views > 0 else 0
                        avg_watch_time_per_view = watch_time_minutes / views if views > 0 else 0

                        input_dict = {
                            "views": views,
                            "likes": likes,
                            "comments": comments,
                            "watch_time_minutes": watch_time_minutes,
                            "video_length_minutes": video_length_minutes,
                            "subscribers": subscribers,
                            "category": category,
                            "device": device,
                            "country": country,
                            "engagement_rate": engagement_rate,
                            "avg_watch_time_per_view": avg_watch_time_per_view
                        }

                        prediction = make_prediction(input_dict)
                        if prediction is not None:
                            st.write(f"📊 Views: {views:,}, 👍 Likes: {likes:,}, 💬 Comments: {comments:,}, 👥 Subsribers: {subscribers:,}, ⏱ Duration: {formatted_duration} (≈ {video_length_minutes:.2f} mins)")
                            st.metric("💰 Predicted Ad Revenue (USD)", value=f"${prediction:,.2f}")

                    else:
                        st.error("❌ Could not fetch video details.")
                except Exception as e:
                    st.error(f"API error: {e}")
        else:
            st.error("Please enter a valid API key and YouTube link.")
# ----------------------------------------Insights----------------------------------------------
elif mode == "Insights":
    st.header("📊 Model Performance Comparison")
    results_df = pd.DataFrame({
    "Model": ["LinearRegression", "GradientBoostingRegressor", "RandomForestRegressor", "XGBRegressor", "DecisionTreeRegressor"],
    "R2": [0.952574, 0.952279, 0.949413, 0.948469, 0.898078],
    "RSME": [13.480053, 13.521871, 13.922094, 14.051310, 19.761392],
    "MAE": [3.111614, 3.623578, 3.588200, 4.221793, 5.401433]
})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fig_r2 = px.bar(results_df, x="Model", y="R2", title="R² Score Comparison",
                        color="Model", text="R2", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='inside')
        fig_r2.update_layout(yaxis=dict(range=[0, 1]), xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        fig_rmse = px.bar(results_df, x="Model", y="RSME", title="RMSE Comparison",
                        color="Model", text="RSME", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_rmse.update_traces(texttemplate='%{text:.2f}', textposition='inside')
        fig_rmse.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col3:
        fig_mae = px.bar(results_df, x="Model", y="MAE", title="MAE Comparison",
                        color="Model", text="MAE", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_mae.update_traces(texttemplate='%{text:.2f}', textposition='inside')
        fig_mae.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown("### 🧠 Model Insights")

    best_r2_model = results_df.loc[results_df['R2'].idxmax(), 'Model']
    best_r2_value = results_df['R2'].max()

    lowest_rmse_model = results_df.loc[results_df['RSME'].idxmin(), 'Model']
    lowest_rmse_value = results_df['RSME'].min()

    lowest_mae_model = results_df.loc[results_df['MAE'].idxmin(), 'Model']
    lowest_mae_value = results_df['MAE'].min()

    worst_r2_model = results_df.loc[results_df['R2'].idxmin(), 'Model']
    highest_rmse_model = results_df.loc[results_df['RSME'].idxmax(), 'Model']
    highest_mae_model = results_df.loc[results_df['MAE'].idxmax(), 'Model']

    st.success(f"✅ **Best Overall Model (Highest R²):** `{best_r2_model}` with R² = {best_r2_value:.3f}")
    st.info(f"📉 **Lowest RMSE:** `{lowest_rmse_model}` (RMSE = {lowest_rmse_value:.2f})")
    st.info(f"📉 **Lowest MAE:** `{lowest_mae_model}` (MAE = {lowest_mae_value:.2f})")

    st.warning(f"⚠️ **Lowest Performing Model:** `{worst_r2_model}` with the smallest R² value.")
    st.warning(f"🚫 **Highest RMSE:** `{highest_rmse_model}` — more prediction error.")
    st.warning(f"🚫 **Highest MAE:** `{highest_mae_model}` — less accurate in absolute terms.")

    # ---------------- Optional Summary Paragraph ----------------
    st.markdown(f"""
    ### 📈 Summary:
    - The **{best_r2_model}** model achieved the **highest R² ({best_r2_value:.3f})**, indicating it explains most of the variance in the data.
    - **{lowest_rmse_model}** and **{lowest_mae_model}** showed the **lowest prediction errors**, making them more consistent and stable.
    - In contrast, the **{worst_r2_model}** model underperformed significantly, suggesting it may be overfitting or not capturing complex relationships well.
    """)
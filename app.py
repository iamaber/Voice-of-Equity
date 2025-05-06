import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from datetime import datetime

# Bangladeshi city coordinates (approximate)
CITY_COORDS = {
    "Sylhet": (24.8949, 91.8687),
    "Rajshahi": (24.3745, 88.6042),
    "Khulna": (22.8456, 89.5403),
    "Chittagong": (22.3569, 91.7832),
    "Barishal": (22.7010, 90.3535),
    "Mymensingh": (24.7471, 90.4203),
    "Dhaka": (23.8103, 90.4125),
}

# Initialize session state without the "Category" feature
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(
        columns=[
            "Date",
            "Location",
            "Profession",
            "Description",
            "Latitude",
            "Longitude",
        ]
    )


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def add_coordinates(df):
    """Add latitude and longitude based on Location"""
    df["Latitude"] = df["Location"].map(lambda x: CITY_COORDS.get(x, (None, None))[0])
    df["Longitude"] = df["Location"].map(lambda x: CITY_COORDS.get(x, (None, None))[1])
    return df.dropna(subset=["Latitude", "Longitude"])


def create_bubble_map(df):
    location_counts = df["Location"].value_counts().reset_index()
    location_counts.columns = ["Location", "Count"]
    location_counts = add_coordinates(location_counts)

    fig = px.scatter_mapbox(
        location_counts,
        lat="Latitude",
        lon="Longitude",
        size="Count",
        color="Count",
        hover_name="Location",
        size_max=50,
        zoom=5.5,
        center={"lat": 23.6850, "lon": 90.3563},
        title="Issue Frequency by Location in Bangladesh",
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig


st.title("Bangladesh Issue Reporting & Analysis Platform")

# Navigation
pages = [
    "Manual Entry",
    "File Upload",
    "View Data",
    "Analysis & Visualization",
    "Predict Protests",
]
page = st.sidebar.radio("Navigation", pages)


def plot_predictions(pred_df):
    """Visualize prediction results"""
    fig = px.scatter_mapbox(
        pred_df,
        lat="Latitude",
        lon="Longitude",
        color="Protest_Probability",
        size="Protest_Probability",
        hover_name="Location",
        hover_data=["Profession", "Description"],
        zoom=5.5,
        center={"lat": 23.6850, "lon": 90.3563},
        title="Protest Probability Heatmap",
        color_continuous_scale=px.colors.sequential.Redor,
    )
    fig.update_layout(mapbox_style="open-street-map")
    return fig


if page == "Predict Protests":
    st.header("Protest Prediction Analysis")

    # Load ML model
    model_file = st.file_uploader("Upload Prediction Model", type=["pkl"])

    if model_file:
        try:
            model = pickle.load(model_file)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()

        # Prediction options
        prediction_type = st.radio(
            "Choose prediction type:", ["Single Entry Prediction", "Batch Prediction"]
        )

        if prediction_type == "Single Entry Prediction":
            with st.form("single_prediction"):
                location = st.selectbox("Location", list(CITY_COORDS.keys()))
                profession = st.text_input("Profession")
                description = st.text_area("Problem Description")

                if st.form_submit_button("Predict"):
                    input_data = pd.DataFrame(
                        [
                            {
                                "Location": location,
                                "Profession": profession,
                                "Description": description,
                                "Latitude": CITY_COORDS[location][0],
                                "Longitude": CITY_COORDS[location][1],
                            }
                        ]
                    )

                    try:
                        prediction = model.predict_proba(input_data)[:, 1][0]
                        st.metric(
                            "Protest Probability",
                            f"{prediction * 100:.1f}%",
                            help="Likelihood of protest occurring based on this report",
                        )
                        st.map(
                            input_data[["Latitude", "Longitude"]].rename(
                                columns={"Latitude": "lat", "Longitude": "lon"}
                            )
                        )
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")

        else:  # Batch Prediction
            uploaded_data = st.file_uploader(
                "Upload Prediction Data", type=["csv", "xlsx"]
            )

            if uploaded_data:
                try:
                    if uploaded_data.name.endswith(".csv"):
                        pred_df = pd.read_csv(uploaded_data)
                    else:
                        pred_df = pd.read_excel(uploaded_data, engine="openpyxl")

                    # Preprocess data
                    pred_df = pred_df.rename(
                        columns={
                            "problem description": "Description",
                            "problem_description": "Description",
                        }
                    )
                    pred_df = add_coordinates(pred_df)

                    if not all(
                        col in pred_df.columns
                        for col in ["Location", "Profession", "Description"]
                    ):
                        st.error("Missing required columns in uploaded data")
                        st.stop()

                    with st.spinner("Making predictions..."):
                        pred_df["Protest_Probability"] = model.predict_proba(pred_df)[
                            :, 1
                        ]
                        pred_df["Protest_Risk"] = pd.cut(
                            pred_df["Protest_Probability"],
                            bins=[0, 0.3, 0.7, 1],
                            labels=["Low", "Medium", "High"],
                        )

                    st.subheader("Prediction Results")
                    st.dataframe(
                        pred_df.sort_values("Protest_Probability", ascending=False)
                    )

                    # Download results
                    csv = convert_df_to_csv(pred_df)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="protest_predictions.csv",
                        mime="text/csv",
                    )

                    # Visualizations
                    st.subheader("Risk Distribution")
                    risk_dist = pred_df["Protest_Risk"].value_counts()
                    st.plotly_chart(
                        px.pie(
                            risk_dist,
                            values=risk_dist.values,
                            names=risk_dist.index,
                            title="Protest Risk Distribution",
                        )
                    )

                    st.subheader("Geographic Risk Analysis")
                    st.plotly_chart(plot_predictions(pred_df))

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

if page == "Manual Entry":
    st.header("Manual Data Entry")

    with st.form("manual_entry"):
        date = st.date_input("Date", datetime.today())
        location = st.selectbox("Location", list(CITY_COORDS.keys()))
        profession = st.text_input("Profession")
        description = st.text_area("Problem Description")

        submitted = st.form_submit_button("Add Entry")
        if submitted:
            if all([location, profession, description]):
                new_entry = pd.DataFrame(
                    [
                        {
                            "Date": date.strftime("%b-%y"),
                            "Location": location,
                            "Profession": profession,
                            "Description": description,
                            "Latitude": CITY_COORDS.get(location, (None, None))[0],
                            "Longitude": CITY_COORDS.get(location, (None, None))[1],
                        }
                    ]
                )
                st.session_state.df = pd.concat(
                    [st.session_state.df, new_entry], ignore_index=True
                )
                st.success("Entry added successfully!")
            else:
                st.error("Please fill all required fields")

elif page == "File Upload":
    st.header("Upload CSV or Excel File")

    uploaded_file = st.file_uploader(
        "Choose a file", type=["csv", "xlsx"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")

            # Rename columns if necessary
            df = df.rename(
                columns={
                    "problem description": "Description",
                    "problem_description": "Description",
                }
            )

            # Check required columns (removed Category)
            required_columns = ["Date", "Location", "Profession", "Description"]
            if all(col in df.columns for col in required_columns):
                df = add_coordinates(df)
                st.session_state.df = pd.concat(
                    [st.session_state.df, df], ignore_index=True
                )
                st.success("File uploaded successfully!")
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
            else:
                missing = [col for col in required_columns if col not in df.columns]
                st.error(f"Missing required columns: {', '.join(missing)}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

elif page == "View Data":
    st.header("View All Data")

    if not st.session_state.df.empty:
        st.dataframe(st.session_state.df)

        csv = convert_df_to_csv(st.session_state.df)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="issue_reports.csv",
            mime="text/csv",
        )

        if st.button("Clear All Data"):
            st.session_state.df = pd.DataFrame(
                columns=[
                    "Date",
                    "Location",
                    "Profession",
                    "Description",
                    "Latitude",
                    "Longitude",
                ]
            )
            st.experimental_rerun()
    else:
        st.info("No data available yet")

elif page == "Analysis & Visualization":
    st.header("Data Analysis & Visualization")

    if st.session_state.df.empty:
        st.warning("No data available for analysis")
        st.stop()

    # Bubble Map
    st.subheader("Geographic Distribution of Issues")
    with st.spinner("Generating map..."):
        fig = create_bubble_map(st.session_state.df)
        st.plotly_chart(fig, use_container_width=True)

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Reports", len(st.session_state.df))
    with col2:
        st.metric("Unique Locations", st.session_state.df["Location"].nunique())

    # Time Series Analysis
    st.subheader("Reports Over Time")
    time_series = st.session_state.df.groupby("Date").size().reset_index(name="Count")
    st.line_chart(time_series.set_index("Date"))

    # Top Professions
    st.subheader("Top 5 Professions")
    top_profs = st.session_state.df["Profession"].value_counts().head(5)
    st.plotly_chart(
        px.bar(
            top_profs,
            labels={"value": "Count", "index": "Profession"},
            color=top_profs.values,
            color_continuous_scale="Greens",
        )
    )

    # Word Cloud
    try:
        from wordcloud import WordCloud

        st.subheader("Description Word Cloud")
        text = " ".join(st.session_state.df["Description"].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            text
        )
        st.image(wordcloud.to_array(), use_column_width=True)
    except ImportError:
        st.info("Install 'wordcloud' package for word cloud visualization")

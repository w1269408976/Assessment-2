import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path


# Web Application Configuration

st.set_page_config(
    page_title="EV Smart Campus Dashboard",
    layout="wide"
)

st.title(" EV Smart Campus Dashboard")

st.write(
    "Electric Vehicle Charging Data Analytics "
    "and Machine Learning Prediction System"
)  # Application description

# Load the cleaned dataset

BASE_DIR = Path(__file__).resolve().parent  # Get the current file directory

@st.cache_data  # Cache the dataset to improve performance
def load_data():
    file_path = BASE_DIR / "cleaned_ev_dataset.csv"

    with open(file_path, "r", encoding="utf-8-sig") as file:
        data = pd.read_csv(file)
    return data

try:
    data = load_data()  # Load cleaned EV data
except FileNotFoundError:
    st.error("Dataset not found. Please check the CSV file.")
    st.stop()  # Stop execution if the file is missing

# Sidebar Navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "Executive KPI Dashboard",
        "Visual Analytics Studio",
        "Live ML Inference"
    ]
)  # Allow users to choose a page


# Page 1: Executive KPI Dashboard
if page == "Executive KPI Dashboard":

    st.header("Executive KPI Dashboard")
    st.write("Overview of EV charging operations.")

    # Create two filter columns
    col1, col2 = st.columns(2)

    with col1:
        charger_options = ["All"] + sorted(
            data["Charger_Type"].dropna().unique().tolist()
        )  # Existing charger types only

        charger_filter = st.selectbox(
            "Select Charger Type",
            charger_options
        )  # Input valid options only

    with col2:
        peak_filter = st.selectbox(
            "Select Peak Hour",
            ["All", "Peak Hour", "Non-Peak Hour"]
        )  # Input valid peak_hour only

    # Apply filters
    filtered_data = data.copy()  # Keep original data unchanged

    if charger_filter != "All":
        filtered_data = filtered_data[
            filtered_data["Charger_Type"] == charger_filter
        ]  # Filter by charger type

    if peak_filter == "Peak Hour":
        filtered_data = filtered_data[
            filtered_data["Peak_Hour_Flag"] == 1
        ]  # Show peak-hour records only

    elif peak_filter == "Non-Peak Hour":
        filtered_data = filtered_data[
            filtered_data["Peak_Hour_Flag"] == 0
        ]  # Show non-peak records only


    # Calculate KPI values

    total_energy = filtered_data["Energy_kWh"].sum()  # Total energy

    avg_duration = filtered_data["Duration_Mins"].mean()  # Average duration

    if filtered_data.empty:
        avg_duration = 0  # Avoid NaN

    # Display
    st.subheader("Key Performance Indicators")

    kpi1, kpi2, kpi3 = st.columns(3)  # Create three KPI cards

    with kpi1:
        st.metric(
            "Total Energy Delivered",
            f"{total_energy:,.2f} kWh"
        )  # Display total energy

    with kpi2:
        st.metric(
            "Average Session Duration",
            f"{avg_duration:.2f} mins"
        )  # Display average charging duration

        # Calculate peak-hour revenue
    peak_data = filtered_data[
        (filtered_data["Start_Hour"] >= 12) &
        (filtered_data["Start_Hour"] < 17)
        ]  # Select peak-hour duration

    peak_revenue = len(peak_data) * 5  # RM 5 per peak-hour session

    with kpi3:
        st.metric(
            "Peak Revenue",
            f"RM {peak_revenue:,.2f}"
        )  # Display

    st.caption(
        "Peak revenue requires verified charging tariff data."
    )

    # Display filtered data table

    st.subheader("Charging Session Records")

    st.write(
        f"Total Records: {len(filtered_data)}"
    )  # Display number of filtered sessions

    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True
    )  # Display a data table




# Page 2: Visual Analytics Studio

elif page == "Visual Analytics Studio":
    st.header("Visual Analytics Studio")  # Page title

    chart = st.selectbox(
        "Select a Chart",
        [
            "Energy vs Battery Temperature",
            "Peak vs Non-Peak Energy",
            "Charger Type Distribution"
        ]
    )  # Select a chart

    fig, ax = plt.subplots(figsize=(8, 5))  # Create figure

    # Chart 1: Energy vs Battery Temperature
    if chart == "Energy vs Battery Temperature":

        sns.scatterplot(
            data=data,
            x="Battery_Temp_C",        # X-axis
            y="Energy_kWh",            # Y-axis
            hue="Charger_Type",        # Different colors for AC and DC
            ax=ax
        )

        ax.set_title("Energy vs Battery Temperature")
        ax.set_xlabel("Battery Temperature (°C)")
        ax.set_ylabel("Energy (kWh)")

    # Chart 2: Peak vs Non-Peak Energy
    elif chart == "Peak vs Non-Peak Energy":

        avg_energy = data.groupby(
            "Peak_Hour_Flag", as_index=False
        )["Energy_kWh"].mean()  # Calculate average energy by peak status

        avg_energy["Peak_Hour_Flag"] = avg_energy[
            "Peak_Hour_Flag"
        ].map({0: "Non-Peak", 1: "Peak"})  # Convert flags into labels

        sns.barplot(
            data=avg_energy,
            x="Peak_Hour_Flag",
            y="Energy_kWh",
            ax=ax
        )  # Create barplot

        ax.set_title("Average Energy: Peak vs Non-Peak")
        ax.set_xlabel("Charging Period")
        ax.set_ylabel("Average Energy (kWh)")



    # Chart 3: Charger Type Distribution
    elif chart == "Charger Type Distribution":

        sns.countplot(
            data=data,
            x="Charger_Type",
            ax=ax
        )  # Creat countplot

        ax.set_title("Charger Type Distribution")
        ax.set_xlabel("Charger Type")
        ax.set_ylabel("Number of Sessions")


    # Display the selected chart
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



# Page 3: Live ML Inference

elif page == "Live ML Inference":
    st.header("Live Machine Learning Inference")
    model = joblib.load(BASE_DIR / "ev_model.pkl")  # Load trained ML model
    st.write("Enter charging information to predict the grid status.")


    # User Input

    st.sidebar.subheader("ML Prediction Inputs")

    charger = st.sidebar.selectbox(
        "Charger Type", ["AC", "DC"]
    )  # Input valid type

    energy = st.sidebar.number_input(
        "Energy (kWh)", min_value=5.0, max_value=120.0, value=40.0
    )  # Limit energy input

    duration = st.sidebar.number_input(
        "Duration (Minutes)", min_value=15.0, max_value=300.0, value=60.0
    )  # Limit duration input

    battery_temp = st.sidebar.slider(
        "Battery Temperature (°C)",
        min_value=float(data["Battery_Temp_C"].min()),
        max_value=float(data["Battery_Temp_C"].max()),
        value=float(data["Battery_Temp_C"].median())
    )  # Use dataset temperature range

    peak = st.sidebar.selectbox(
        "Peak Hour", ["No", "Yes"]
    )  # Input valid hour

    peak_flag = 1 if peak == "Yes" else 0  # Convert to binary value

    power = energy / (duration / 60)  # Calculate charging power


    # Prepare input data for prediction
    input_data = pd.DataFrame([{
        "Charger_Type": charger,
        "Energy_kWh": energy,
        "Duration_Mins": duration,
        "Battery_Temp_C": battery_temp,
        "Peak_Hour_Flag": peak_flag,
        "Power_kW": power
    }])  # Match the training features

    st.subheader("Input Summary")  # Display input information
    st.dataframe(input_data, hide_index=True)  # Show selected values


    # Predict grid status
    prediction = model.predict(input_data)[0]  # Predict grid status

    st.subheader("Prediction Result")

    if prediction == "Normal":
        st.success("Grid Status: Normal")  # Green alert

    elif prediction == "Overheated":
        st.warning("Grid Status: Overheated")  # Yellow alert

    elif prediction == "Grid_Surge":
        st.error("Grid Status: Grid_Surge")  # Red alert

    else:
        st.error("Unknown prediction result")  # Handle unexpected output

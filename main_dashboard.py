
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# -------------------------------------------
# PAGE CONFIG
# -------------------------------------------
st.set_page_config(page_title="GradTrack Dashboard", layout="wide")

# -------------------------------------------
# LOAD DATA
# -------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("grad_outcome.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("`grad_outcome.csv` not found. Please place it in the same directory.")
    st.stop()

# Ensure columns are numeric where expected
for col in ["overall_gwa", "units_passed", "on_time_prob", "delayed_prob"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# -------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------
st.sidebar.header("Filters")

year_levels = sorted(df["year_level"].dropna().unique())
predictions = sorted(df["prediction"].dropna().unique())

selected_year = st.sidebar.multiselect("Select Year Level", year_levels, default=year_levels)
selected_pred = st.sidebar.multiselect("Select Prediction", predictions, default=predictions)

filtered_df = df[
    (df["year_level"].isin(selected_year)) &
    (df["prediction"].isin(selected_pred))
]

# -------------------------------------------
# SUMMARY METRICS
# -------------------------------------------
st.subheader("Summary Metrics")
col1, col2, col3, col4 = st.columns(4)

total_students = len(filtered_df)
on_time_rate = filtered_df["prediction"].value_counts(normalize=True).get("On Time", 0) * 100
avg_gwa = filtered_df["overall_gwa"].mean()
avg_units = filtered_df["units_passed"].mean()

col1.metric("Total Students", total_students)
col2.metric("On Time (%)", f"{on_time_rate:.2f}%")
col3.metric("Average GWA", f"{avg_gwa:.2f}")
col4.metric("Avg Units Passed", f"{avg_units:.1f}")

st.divider()

# -------------------------------------------
# CHARTS SECTION
# -------------------------------------------
st.subheader("Graduation Prediction Analytics")

# 1️⃣ Prediction Distribution
fig1 = px.pie(
    filtered_df,
    names="prediction",
    title="Prediction Distribution",
    color="prediction",
    color_discrete_map={"On Time": "green", "Delayed": "red"},
)
st.plotly_chart(fig1, use_container_width=True)

# 2️⃣ Prediction Confidence Scatter Plot
st.subheader("Prediction Confidence (On-Time vs Delayed Probability)")

# Filter valid points
scatter_df = filtered_df.dropna(subset=["on_time_prob", "delayed_prob", "units_passed"])
scatter_df["units_passed"] = np.where(scatter_df["units_passed"] <= 0, 1, scatter_df["units_passed"])

fig2 = px.scatter(
    scatter_df,
    x="on_time_prob",
    y="delayed_prob",
    color="prediction",
    size="units_passed",
    hover_data=["student_id", "overall_gwa", "year_level"],
    title="Prediction Confidence (On-Time vs Delayed Probability)",
)
st.plotly_chart(fig2, use_container_width=True)

# 3️⃣ Average GWA by Year Level
st.subheader("Average GWA by Year Level")
gwa_by_year = filtered_df.groupby("year_level")["overall_gwa"].mean().reset_index()
fig3 = px.bar(
    gwa_by_year,
    x="year_level",
    y="overall_gwa",
    title="Average GWA per Year Level",
    color="year_level",
    text_auto=True,
)
st.plotly_chart(fig3, use_container_width=True)

# 4️⃣ Units Passed Distribution
st.subheader("Units Passed Distribution")
fig4 = px.box(
    filtered_df,
    x="year_level",
    y="units_passed",
    color="year_level",
    title="Distribution of Units Passed by Year Level",
)
st.plotly_chart(fig4, use_container_width=True)

# -------------------------------------------
# INSIGHTS AND RECOMMENDATIONS
# -------------------------------------------
st.subheader("Insights & Recommendations")

if on_time_rate < 70:
    st.warning("The on-time graduation rate is below 70%. Recommend reviewing academic support for at-risk students.")
else:
    st.success("Graduation rates are strong! Continue current academic strategies.")

if avg_gwa > 2.5:
    st.warning("Average GWA suggests potential academic challenges. Consider targeted tutoring or advising.")
else:
    st.info("Overall academic performance is healthy.")

st.markdown("""
- **Observation:** Higher units passed correlate with higher on-time probabilities.  
- **Recommendation:** Encourage students to maintain consistent course loads and early remediation for failed subjects.  
- **Next Step:** Analyze historical trends per semester to detect when delays start increasing.
""")

# -------------------------------------------
# DATA TABLE
# -------------------------------------------
st.subheader("Student Details")
st.dataframe(filtered_df)



import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# -------------------------------------------
# PAGE CONFIG
# -------------------------------------------
st.set_page_config(page_title="GradTrack Strategic Dashboard", layout="wide")

# -------------------------------------------
# CUSTOM CSS
# -------------------------------------------
st.markdown("""
    <style>
        /* General page style */
        .main {
            background-color: #f5f7fa;
        }
        h2, h3 {
            color: #2b2d42;
            font-weight: 700;
        }

        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #6366F1, #8B5CF6);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            color: white;
            box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        }
        .metric-subtext {
            font-size: 14px;
            opacity: 0.9;
        }

        /* Section containers */
        .section {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-top: 20px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        }

        /* Insights cards */
        .insight-card {
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 15px;
        }
        .insight-warning {
            background-color: #fdecea;
            border-left: 5px solid #f87171;
        }
        .insight-success {
            background-color: #e0f7f0;
            border-left: 5px solid #34d399;
        }
        .insight-info {
            background-color: #e0f2fe;
            border-left: 5px solid #3b82f6;
        }
        .insight-title {
            font-weight: 600;
        }

        /* Section titles */
        .section-header {
            background: linear-gradient(90deg, #6366F1, #8B5CF6);
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
            /* Chart title tweaks */
.js-plotly-plot .plotly .main-svg {
    border-radius: 10px;
}

/* Section header icon alignment */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Optional subtle hover on chart container */
.section:hover {
    box-shadow: 0 5px 12px rgba(99, 102, 241, 0.2);
    transition: 0.3s ease;
}
    </style>
""", unsafe_allow_html=True)

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

# Ensure numeric columns
for col in ["overall_gwa", "units_passed", "on_time_prob", "delayed_prob"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# -------------------------------------------
# SIDEBAR
# -------------------------------------------
st.sidebar.title("GradTrack")
st.sidebar.caption("Strategic Student Analytics")
st.sidebar.success("Live Data Connected")

st.sidebar.divider()
st.sidebar.subheader("Display Filters")

year_levels = sorted(df["year_level"].dropna().unique())
predictions = sorted(df["prediction"].dropna().unique())

selected_year = st.sidebar.multiselect("Select Year Level", year_levels, default=year_levels)
selected_pred = st.sidebar.multiselect("Select Prediction", predictions, default=predictions)

filtered_df = df[
    (df["year_level"].isin(selected_year)) &
    (df["prediction"].isin(selected_pred))
]

# -------------------------------------------
# DASHBOARD TITLE
# -------------------------------------------
st.markdown("<h2 style='text-align:center;'>GradTrack Strategic Dashboard</h2>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------
# SUMMARY METRICS (Colored Cards)
# -------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_students = len(filtered_df)
on_time_rate = filtered_df["prediction"].value_counts(normalize=True).get("On Time", 0) * 100
avg_gwa = filtered_df["overall_gwa"].mean()
avg_units = filtered_df["units_passed"].mean()

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <h3>{total_students}</h3>
            <p>Total Students</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <h3>{on_time_rate:.1f}%</h3>
            <p>On-Time Graduation Rate</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_gwa:.2f}</h3>
            <p>Average GWA</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_units:.0f}</h3>
            <p>Average Units Passed</p>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------------------
# CHARTS SECTION
# -------------------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Graduation Analytics</div>", unsafe_allow_html=True)

colA, colB = st.columns(2)

# Prediction distribution
with colA:
    fig1 = px.pie(
        filtered_df,
        names="prediction",
        title="Prediction Distribution",
        color="prediction",
        color_discrete_map={"On Time": "#10b981", "Delayed": "#ef4444"},
    )
    st.plotly_chart(fig1, use_container_width=True)

# Average GWA
with colB:
    gwa_by_year = filtered_df.groupby("year_level")["overall_gwa"].mean().reset_index()
    fig2 = px.bar(
        gwa_by_year,
        x="year_level",
        y="overall_gwa",
        title="Average GWA per Year Level",
        text_auto=True,
        color="year_level"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------
# ADDITIONAL CHART: On-Time vs Delayed per Year Level
# -------------------------------------------

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>On-Time vs. Delayed Trends</div>", unsafe_allow_html=True)

# Create grouped data
trend = filtered_df.groupby(["year_level", "prediction"]).size().reset_index(name="count")

# Plot
fig3 = px.bar(
    trend,
    x="year_level",
    y="count",
    color="prediction",
    barmode="group",
    title="On-Time vs. Delayed Students per Year Level",
    text_auto=True,
    color_discrete_map={"On Time": "#10b981", "Delayed": "#ef4444"}
)

# Style chart
fig3.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    title_font=dict(size=18, color="#2b2d42"),
    legend=dict(title="", orientation="h", y=-0.2, x=0.3),
)

# Display chart
st.plotly_chart(fig3, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------
# INSIGHTS & RECOMMENDATIONS SECTION
# -------------------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Strategic Insights & Recommendations</div>", unsafe_allow_html=True)

colX, colY = st.columns(2)

with colX:
    if on_time_rate < 70:
        st.markdown("""
            <div class="insight-card insight-warning">
                <div class="insight-title">Attention Needed:</div>
                On-time graduation rate is below target (70%). Consider implementing academic advising or mentorship programs.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="insight-card insight-success">
                <div class="insight-title">Strong Performance:</div>
                Majority of students graduate on time — continue current academic support strategies.
            </div>
        """, unsafe_allow_html=True)

    if avg_gwa > 2.5:
        st.markdown("""
            <div class="insight-card insight-warning">
                <div class="insight-title">Academic Challenge:</div>
                Average GWA above 2.5 suggests students may need tutoring or study skill workshops.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="insight-card insight-info">
                <div class="insight-title">Healthy Academic Standing:</div>
                Overall GWA levels are within the strong performance range.
            </div>
        """, unsafe_allow_html=True)

with colY:
    st.markdown("""
        <div class="insight-card insight-info">
            <div class="insight-title">Consistent Course Load:</div>
            Higher unit completion correlates with on-time graduation. Encourage balanced load planning and early remediation.
        </div>

        <div class="insight-card insight-success">
            <div class="insight-title">Next Steps:</div>
            Track semester-based trends to identify at-risk cohorts and optimize early intervention programs.
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

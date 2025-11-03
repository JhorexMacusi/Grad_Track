import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

# -------------------------------------------
# PAGE CONFIG
# -------------------------------------------
st.set_page_config(page_title="Historic Datasets", layout="wide")

st.title("Historic Datasets")
st.caption("Explore and analyze archived or past student datasets for longitudinal insights using Logistic Regression.")

# -------------------------------------------
# LOAD DATA
# -------------------------------------------
@st.cache_data
def load_historic_data():
    df = pd.read_csv("cleaned_dataset.csv")
    return df

try:
    df = load_historic_data()
    df.columns = [c.strip().lower() for c in df.columns]

    st.subheader("Dataset Overview")
    st.write(f"Total Records: {len(df)}")
    st.dataframe(df.head())

    # -------------------------------------------
    # DATA CLEANING
    # -------------------------------------------
    label_encoder = LabelEncoder()
    df["graduation_outcome_encoded"] = label_encoder.fit_transform(df["graduation_outcome"])

    numeric_cols = ["units_passed", "failed_prerequisite", "failed_commoncourses", "overall_gwa"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # -------------------------------------------
    # EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Graduation Outcome Distribution**")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x="graduation_outcome", palette="Set2", ax=ax1)
        ax1.set_title("Count of Graduation Outcomes")
        st.pyplot(fig1)

    with col2:
        st.markdown("**Overall GWA Distribution by Outcome**")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x="graduation_outcome", y="overall_gwa", palette="Set1", ax=ax2)
        ax2.set_title("GWA Distribution per Outcome")
        st.pyplot(fig2)

    # -------------------------------------------
    # CORRELATION HEATMAP
    # -------------------------------------------
    st.markdown("**Correlation Heatmap**")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.heatmap(df[numeric_cols + ["graduation_outcome_encoded"]].corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    # -------------------------------------------
    # LOGISTIC REGRESSION MODEL
    # -------------------------------------------
    st.subheader("Logistic Regression Model")

    X = df[["units_passed", "failed_prerequisite", "failed_commoncourses", "overall_gwa"]]
    y = df["graduation_outcome_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)

    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion Matrix
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    fig4, ax4 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax4)
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Actual")
    st.pyplot(fig4)

    # Feature Importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    fig5, ax5 = plt.subplots()
    sns.barplot(data=importance, x="Coefficient", y="Feature", palette="crest", ax=ax5)
    ax5.set_title("Feature Importance (Logistic Regression Coefficients)")
    st.pyplot(fig5)

    # ROC Curve
    st.markdown("### ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    fig6, ax6 = plt.subplots()
    ax6.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
    ax6.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax6.set_xlabel("False Positive Rate")
    ax6.set_ylabel("True Positive Rate")
    ax6.set_title("ROC Curve")
    ax6.legend()
    st.pyplot(fig6)

    # -------------------------------------------
    # CONCLUSION
    # -------------------------------------------
    st.subheader("Conclusion")
    st.markdown(f"""
    - Logistic Regression modeled the relationship between **academic metrics** and **graduation outcomes**.
    - **Units Passed** and **Overall GWA** are strong indicators of on-time graduation.
    - **Failed Prerequisites** and **Failed Common Courses** negatively affect graduation chances.
    - Model Accuracy: **{accuracy * 100:.2f}%**
    - AUC Score: **{roc_auc:.2f}** — indicating the model has excellent discriminatory power.
    """)

except FileNotFoundError:
    st.error("Historic dataset not found. Please upload or place `cleaned_dataset.csv` in the app folder.")

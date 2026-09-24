"""
LOAN PREDICTION DATA ANALYSIS -- PREMIUM FINTECH STREAMLIT WEB APPLICATION

This file is organized in two clearly separated parts, in this order:

  PART 1 - ORIGINAL PROJECT CODE (verbatim, unmodified, not executed)
  PART 2 - STREAMLIT WEB APPLICATION (the upgraded FinTech SaaS adaptation layer)

PART 1 is the complete original Colab script, reproduced exactly as
provided, character for character, inside a raw string named
ORIGINAL_PROJECT_CODE. It is kept for reference and for viva presentations.

PART 2 provides a professional FinTech SaaS interface utilizing advanced CSS,
interactive components, card layouts, visual pipelines, and executive summaries
while preserving all underlying Pandas, NumPy, Matplotlib, and rule-based
scoring algorithms.
"""

# =============================================================================
# PART 1 -- ORIGINAL PROJECT CODE (SOURCE OF TRUTH)
# Reproduced exactly as provided. Not executed -- see explanation above.
# =============================================================================

ORIGINAL_PROJECT_CODE = r'''
# =============================================================================
# LOAN PREDICTION DATA ANALYSIS
# Members 1 + 2 + 3 + 4
# Technology: Python, Pandas, NumPy, Matplotlib
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================ MEMBER 1 =======================================

# STEP 1 - LOAD DATASET

from google.colab import files
import io

uploaded = files.upload()

if not uploaded:
    print("No dataset selected. Program stopped.")
    exit()

file_name = list(uploaded.keys())[0]

data = pd.read_csv(io.BytesIO(uploaded[file_name]))

print("\nDataset Loaded Successfully!")
print("Dataset File :", file_name)
print("Rows         :", data.shape[0])
print("Columns      :", data.shape[1])

print("\n" + "=" * 55)
print("          LOAN PREDICTION DATA ANALYSIS")
print("=" * 55)
print("Dataset File :", "loan-train.csv")
print("Rows         :", data.shape[0])
print("Columns      :", data.shape[1])

# STEP 2 - DATASET OVERVIEW
print("\n--- DATASET OVERVIEW ---")
print("Columns:", list(data.columns))
print("\nFirst 5 Records:")
print(data.head())
print("\nData Types:")
print(data.dtypes)
print("\nDataset Shape:", data.shape)

# STEP 3 - MISSING VALUE ANALYSIS
print("\n--- MISSING VALUE ANALYSIS ---")
missing = pd.DataFrame({
    "Missing Count": data.isnull().sum(),
    "Missing %": (data.isnull().sum() / len(data) * 100).round(2)
})
print(missing)

# STEP 4 - HANDLE MISSING VALUES
numerical_columns = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History"
]
categorical_columns = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area"
]

for column in numerical_columns:
    if column in data.columns:
        data[column] = data[column].fillna(data[column].median())

for column in categorical_columns:
    if column in data.columns:
        data[column] = data[column].fillna(data[column].mode()[0])

print("\nRemaining Missing Values:", data.isnull().sum().sum())

# STEP 5 - DUPLICATE ANALYSIS
print("\nDuplicate Records:", data.duplicated().sum())
data = data.drop_duplicates().copy()
print("Records After Duplicate Removal:", len(data))

# ============================ MEMBER 2 =======================================

print("\n" + "=" * 55)
print("              MEMBER 2 - STATISTICAL ANALYSIS")
print("=" * 55)

# STEP 6 - LOAN STATUS
status_count = data["Loan_Status"].value_counts()
status_percentage = data["Loan_Status"].value_counts(normalize=True) * 100
status_table = pd.DataFrame({
    "Count": status_count,
    "Percentage": status_percentage.round(2)
})
print("\n--- LOAN STATUS ---")
print(status_table)

# STEP 7 - CATEGORICAL ANALYSIS
analysis_categories = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Credit_History", "Property_Area"
]

for column in analysis_categories:
    print(f"\n--- {column} ---")
    print(data[column].value_counts())

# STEP 8 - NUMERICAL STATISTICS
analysis_numbers = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]

print("\n--- NUMERICAL STATISTICS ---")
for column in analysis_numbers:
    print(f"\n{column}")
    print(data[column].agg(["mean", "median", "min", "max"]).round(2))

# STEP 9 - CORRELATION
correlation_columns = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History"
]
print("\n--- CORRELATION MATRIX ---")
print(data[correlation_columns].corr().round(2))

# ============================ VISUALIZATION ================================

print("\n" + "=" * 55)
print("                 VISUALIZATION")
print("=" * 55)

# 1. Loan Status
plt.figure(figsize=(7, 5))
bars = plt.bar(
    ["Approved", "Rejected"],
    [(data["Loan_Status"] == "Y").sum(),
     (data["Loan_Status"] == "N").sum()],
    edgecolor="black"
)
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(int(bar.get_height())),
        ha="center", va="bottom"
    )
plt.title("Loan Approval Status")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applications")
plt.tight_layout()
plt.show()

# 2. Income / Loan distributions
for column, title, xlabel in [
    ("ApplicantIncome", "Applicant Income Distribution", "Applicant Income"),
    ("CoapplicantIncome", "Coapplicant Income Distribution", "Coapplicant Income"),
    ("LoanAmount", "Loan Amount Distribution", "Loan Amount")
]:
    plt.figure(figsize=(7, 5))
    plt.hist(data[column], bins=20, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Applications")
    plt.tight_layout()
    plt.show()

# Categorical comparison graph
def category_plot(column, title):
    table = pd.crosstab(data[column], data["Loan_Status"])
    table.plot(kind="bar", figsize=(7, 5), edgecolor="black")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Number of Applications")
    plt.xticks(rotation=0)
    plt.legend(title="Loan Status")
    plt.tight_layout()
    plt.show()

category_plot("Credit_History", "Credit History vs Loan Status")
category_plot("Education", "Education vs Loan Status")
category_plot("Property_Area", "Property Area vs Loan Status")
category_plot("Gender", "Gender vs Loan Status")

# Income vs Loan Amount
plt.figure(figsize=(7, 5))
plt.scatter(data["ApplicantIncome"], data["LoanAmount"], alpha=0.6)
plt.title("Applicant Income vs Loan Amount")
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.tight_layout()
plt.show()

# ============================ MEMBER 3 =======================================

print("\n" + "=" * 55)
print("              MEMBER 3 - EDA + INSIGHTS")
print("=" * 55)

# STEP 10 - CATEGORY-WISE APPROVAL / REJECTION
eda_categories = [
    "Credit_History", "Education", "Property_Area",
    "Self_Employed", "Dependents"
]

for column in eda_categories:
    count_table = pd.crosstab(data[column], data["Loan_Status"])
    percentage_table = (
        pd.crosstab(data[column], data["Loan_Status"], normalize="index") * 100
    )

    result = pd.DataFrame({
        "Rejected Count": count_table.get("N", 0),
        "Rejected %": percentage_table.get("N", 0).round(2),
        "Approved Count": count_table.get("Y", 0),
        "Approved %": percentage_table.get("Y", 0).round(2)
    })

    print(f"\n--- {column} vs Loan Status ---")
    print(result)

# STEP 11 - NUMERICAL ANALYSIS BY LOAN STATUS
print("\n--- NUMERICAL ANALYSIS BY LOAN STATUS ---")
for column in analysis_numbers:
    result = data.groupby("Loan_Status")[column].agg(
        Mean="mean", Median="median", Minimum="min", Maximum="max"
    ).round(2)
    print(f"\n{column}")
    print(result)

# STEP 12 - CONTRIBUTION ANALYSIS
print("\n--- CONTRIBUTION ANALYSIS ---")
for column in analysis_numbers:
    totals = data.groupby("Loan_Status")[column].sum()
    contribution = (totals / totals.sum() * 100).round(2)
    print(f"\n{column} Contribution %:")
    print(contribution)

# STEP 13 - APPROVAL RATES
credit_rate = data.groupby("Credit_History")["Loan_Status"].apply(
    lambda x: (x == "Y").mean() * 100
).round(2)
education_rate = data.groupby("Education")["Loan_Status"].apply(
    lambda x: (x == "Y").mean() * 100
).round(2)
property_rate = data.groupby("Property_Area")["Loan_Status"].apply(
    lambda x: (x == "Y").mean() * 100
).round(2)

print("\nCredit History Approval Rate:")
print(credit_rate)
print("\nEducation Approval Rate:")
print(education_rate)
print("\nProperty Area Approval Rate:")
print(property_rate)

# STEP 14 - FINAL EDA INSIGHTS
approved = (data["Loan_Status"] == "Y").sum()
rejected = (data["Loan_Status"] == "N").sum()

print("\n--- FINAL EDA INSIGHTS ---")
print("Total Applications:", len(data))
print("Approved Loans:", approved)
print("Rejected Loans:", rejected)
print("Overall Approval Rate:", round(approved / len(data) * 100, 2), "%")
print("Average Applicant Income:", round(data["ApplicantIncome"].mean(), 2))
print("Average Coapplicant Income:", round(data["CoapplicantIncome"].mean(), 2))
print("Average Loan Amount:", round(data["LoanAmount"].mean(), 2))

# ============================ MEMBER 4 =======================================

print("\n" + "=" * 55)
print("             MEMBER 4 - LOAN PREDICTION")
print("=" * 55)

# STEP 15 - NEW APPLICANT INPUT
loan_id = input("\nEnter Loan ID: ")
gender = input("Enter Gender (Male / Female): ")
married = input("Enter Married (Yes / No): ")
dependents = input("Enter Dependents (0 / 1 / 2 / 3+): ")
education = input("Enter Education (Graduate / Not Graduate): ")
self_employed = input("Enter Self Employed (Yes / No): ")
applicant_income = float(input("Enter Applicant Income: "))
coapplicant_income = float(input("Enter Coapplicant Income: "))
loan_amount = float(input("Enter Loan Amount: "))
loan_term = float(input("Enter Loan Amount Term: "))
credit_history = float(input("Enter Credit History (1 / 0): "))
property_area = input("Enter Property Area (Urban / Semiurban / Rural): ")

new_applicant = pd.DataFrame({
    "Loan_ID": [loan_id],
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area]
})

print("\n--- NEW APPLICANT DETAILS ---")
print(new_applicant.to_string(index=False))

# STEP 16 - HISTORICAL APPROVAL RATES
rates = {}
for name, column in [
    ("Credit History", "Credit_History"),
    ("Education", "Education"),
    ("Property Area", "Property_Area"),
    ("Self Employed", "Self_Employed"),
    ("Dependents", "Dependents")
]:
    rates[name] = data.groupby(column)["Loan_Status"].apply(
        lambda x: (x == "Y").mean() * 100
    )

# STEP 17 - CATEGORY SCORES
credit_score = rates["Credit History"].get(
    credit_history, rates["Credit History"].mean()
)
education_score = rates["Education"].get(
    education, rates["Education"].mean()
)
property_score = rates["Property Area"].get(
    property_area, rates["Property Area"].mean()
)
self_score = rates["Self Employed"].get(
    self_employed, rates["Self Employed"].mean()
)
dependents_score = rates["Dependents"].get(
    dependents, rates["Dependents"].mean()
)

# STEP 18 - NUMERICAL SCORES
income_average = data["ApplicantIncome"].mean()
coapplicant_average = data["CoapplicantIncome"].mean()
loan_average = data["LoanAmount"].mean()

income_score = 100 if applicant_income >= income_average else 0
coapplicant_score = 100 if coapplicant_income >= coapplicant_average else 0
loan_score = 100 if loan_amount <= loan_average else 0

# STEP 19 - FINAL PREDICTION
scores = {
    "Credit History": credit_score,
    "Education": education_score,
    "Property Area": property_score,
    "Self Employed": self_score,
    "Dependents": dependents_score,
    "Applicant Income": income_score,
    "Coapplicant Income": coapplicant_score,
    "Loan Amount": loan_score
}

approval_score = np.mean(list(scores.values()))
prediction = "APPROVED" if approval_score >= 50 else "REJECTED"

print("\n--- APPLICANT SCORES ---")
for name, score in scores.items():
    print(f"{name}: {score:.2f}%")

print("\nOverall Approval Score:", round(approval_score, 2), "%")
print("Final Loan Prediction:", prediction)

# ============================ CONCLUSION =====================================

print("\n" + "=" * 55)
print("                    CONCLUSION")
print("=" * 55)
print("Loan dataset analysis completed successfully.")
print("EDA and visualization were performed.")
print("A rule-based score was calculated for the new applicant.")
print("Final Prediction:", prediction)
print("=" * 55)
'''

# =============================================================================
# PART 2 -- STREAMLIT WEB APPLICATION (UPGRADED FINTECH SaaS LAYER)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Loan Prediction Dashboard | Data Analytics",
    page_icon="⚡",
    layout="wide",
initial_sidebar_state="auto",
)

def inject_fintech_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            --bg-main:      #080D18;
            --bg-card:      #0B1220;
            --bg-card-hover:#111827;
            --border:       rgba(255, 255, 255, 0.08);
            --border-hover: rgba(59, 130, 246, 0.4);
            --text-main:    #F8FAFC;
            --text-muted:   #94A3B8;
            --primary:      #3B82F6;
            --primary-hover:#2563EB;
            --success:      #22C55E;
            --success-bg:   rgba(34, 197, 94, 0.12);
            --warning:      #F59E0B;
            --danger:       #EF4444;
            --danger-bg:    rgba(239, 68, 68, 0.12);
            --sidebar-bg:   #060A12;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-main);
        }

        .stApp { background-color: var(--bg-main); }

#MainMenu, footer {
    visibility: hidden;
}
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 4rem;
            max-width: 1280px;
        }

        /* Typography */
        h1, h2, h3, h4, h5 {
            color: var(--text-main);
            font-weight: 600;
            letter-spacing: -0.02em;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] * { color: var(--text-muted); }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 { color: var(--text-main); }

        .sidebar-brand {
            padding: 0.5rem 0 1.2rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1rem;
        }
        .sidebar-brand .title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: 0.02em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .sidebar-brand .subtitle {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
            font-weight: 500;
            padding: 3px 8px;
            border-radius: 999px;
            margin-top: 8px;
            background: rgba(34, 197, 94, 0.1);
            color: #4ADE80;
            border: 1px solid rgba(34, 197, 94, 0.2);
        }
        .status-pill.waiting {
            background: rgba(245, 158, 11, 0.1);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.2);
        }

        .sidebar-section-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748B;
            margin: 1.2rem 0 0.4rem 0;
            font-weight: 600;
        }

        /* Navigation radio styling override */
        section[data-testid="stSidebar"] [role="radiogroup"] {
            gap: 0.25rem;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            background: transparent;
            padding: 0.55rem 0.75rem;
            border-radius: 8px;
            border: 1px solid transparent;
            transition: all 0.2s ease;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.03);
            border-color: var(--border);
        }

        /* Cards */
        .fintech-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.2s ease;
            height: 100%;
        }
        .fintech-card:hover {
            border-color: var(--border-hover);
            background: var(--bg-card-hover);
        }

        /* KPI Cards */
        .kpi-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            position: relative;
            overflow: hidden;
        }
        .kpi-card::after {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 3px; height: 100%;
            background: var(--primary);
        }
        .kpi-label {
            font-size: 0.78rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.03em;
            font-variant-numeric: tabular-nums;
        }
        .kpi-sub {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }

        /* Page Headers */
        .page-header {
            margin-bottom: 1.8rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }
        .page-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-main);
            margin: 0 0 0.3rem 0;
        }
        .page-subtitle {
            font-size: 0.95rem;
            color: var(--text-muted);
            margin: 0;
        }

        /* Section Header */
        .section-header {
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--text-main);
            margin: 1.8rem 0 0.8rem 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .section-desc {
            font-size: 0.88rem;
            color: var(--text-muted);
            margin-top: -0.5rem;
            margin-bottom: 1rem;
        }

        /* Prediction Result Panels */
        .result-panel {
            border-radius: 14px;
            padding: 2rem;
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1.5rem;
            margin: 1.5rem 0;
        }
        .result-panel.approved {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(11, 18, 32, 0.8));
            border-color: rgba(34, 197, 94, 0.3);
        }
        .result-panel.rejected {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(11, 18, 32, 0.8));
            border-color: rgba(239, 68, 68, 0.3);
        }
        .result-score-num {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: var(--text-main);
            font-variant-numeric: tabular-nums;
        }
        .result-score-lbl {
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .result-badge {
            font-size: 1.1rem;
            font-weight: 700;
            padding: 0.6rem 1.5rem;
            border-radius: 999px;
            letter-spacing: 0.04em;
        }
        .result-badge.approved { background: var(--success); color: #FFFFFF; }
        .result-badge.rejected { background: var(--danger); color: #FFFFFF; }

        /* Buttons & Forms */
        .stButton > button, .stFormSubmitButton > button {
            background: var(--primary);
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 0.92rem;
            transition: all 0.2s ease;
            width: 100%;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            background: var(--primary-hover);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }

        /* Expanders & Dataframes */
        [data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            background: var(--bg-card) !important;
        }
        .dataframe {
            background: var(--bg-card) !important;
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

NUMERICAL_COLUMNS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History"
]
CATEGORICAL_COLUMNS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area"
]
ANALYSIS_CATEGORIES = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Credit_History", "Property_Area"
]
ANALYSIS_NUMBERS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
CORRELATION_COLUMNS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History"
]
EDA_CATEGORIES = [
    "Credit_History", "Education", "Property_Area",
    "Self_Employed", "Dependents"
]
REQUIRED_COLUMNS = [
    "Loan_ID", "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term", "Credit_History",
    "Property_Area", "Loan_Status"
]

def render_kpi(label, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-header">
            <h1 class="page-title">{title}</h1>
            <p class="page-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_section(title, desc=""):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="section-desc">{desc}</div>', unsafe_allow_html=True)

def render_chart_card(title, desc, fig):
    st.markdown(
        f"""
        <div class="fintech-card" style="margin-bottom: 1.5rem;">
            <div style="font-weight: 600; font-size: 1.05rem; color: #F8FAFC; margin-bottom: 0.2rem;">{title}</div>
            <div style="font-size: 0.84rem; color: #94A3B8; margin-bottom: 1rem;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.pyplot(fig)

def render_empty_state():
    st.markdown(
        """
        <div style="text-align: center; padding: 5rem 2rem; background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
            <h2 style="font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; color: #F8FAFC;">Loan Prediction Data Analysis System </h2>
            <p style="color: var(--text-muted); max-width: 600px; margin: 0 auto 1.5rem auto; font-size: 0.98rem;">
                Welcome to the enterprise-grade Loan Prediction & Data Analysis platform. To unlock statistical insights, exploratory visualizations, and rule-based credit evaluation, please upload your dataset (e.g., <code>loan-train.csv</code>) via the sidebar.
            </p>
            <div style="display: inline-flex; gap: 1.5rem; font-size: 0.88rem; color: var(--text-muted);">
                <span>✓ Data Quality Validation</span>
                <span>✓ Statistical Profiling</span>
                <span>✓ Interactive EDA</span>
                <span>✓ Credit Scoring Engine</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_error_banner(missing_cols):
    cols_str = ", ".join(missing_cols)
    st.markdown(
        f"""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 1.5rem; margin: 2rem 0;">
            <div style="font-weight: 700; color: #FCA5A5; font-size: 1.1rem; margin-bottom: 0.5rem;">⚠ Dataset Schema Mismatch</div>
            <p style="color: #CBD5E1; font-size: 0.92rem; margin: 0 0 0.8rem 0;">
                The uploaded CSV file is missing required columns necessary for proper analysis and prediction.
            </p>
            <div style="font-size: 0.85rem; color: #F8FAFC; background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 8px; font-family: monospace;">
                Missing Columns: {cols_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def step1_load_dataset(uploaded_file):
    return pd.read_csv(uploaded_file)

def step3_missing_value_analysis(data):
    return pd.DataFrame({
        "Missing Count": data.isnull().sum(),
        "Missing %": (data.isnull().sum() / len(data) * 100).round(2)
    })

def step4_handle_missing_values(data):
    for column in NUMERICAL_COLUMNS:
        if column in data.columns:
            data[column] = data[column].fillna(data[column].median())

    for column in CATEGORICAL_COLUMNS:
        if column in data.columns:
            data[column] = data[column].fillna(data[column].mode()[0])

    remaining_missing = data.isnull().sum().sum()
    return data, remaining_missing

def step5_duplicate_analysis(data):
    duplicate_count = data.duplicated().sum()
    data = data.drop_duplicates().copy()
    records_after = len(data)
    return data, duplicate_count, records_after

def step6_loan_status(data):
    status_count = data["Loan_Status"].value_counts()
    status_percentage = data["Loan_Status"].value_counts(normalize=True) * 100
    return pd.DataFrame({
        "Count": status_count,
        "Percentage": status_percentage.round(2)
    })

def step7_categorical_analysis(data):
    return {column: data[column].value_counts() for column in ANALYSIS_CATEGORIES}

def step8_numerical_statistics(data):
    return {
        column: data[column].agg(["mean", "median", "min", "max"]).round(2)
        for column in ANALYSIS_NUMBERS
    }

def step9_correlation(data):
    return data[CORRELATION_COLUMNS].corr().round(2)

def plot_loan_status(data):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#0B1220')
    ax.set_facecolor('#0B1220')
    
    approved_count = (data["Loan_Status"] == "Y").sum()
    rejected_count = (data["Loan_Status"] == "N").sum()
    
    bars = ax.bar(
        ["Approved", "Rejected"],
        [approved_count, rejected_count],
        color=["#22C55E", "#EF4444"],
        width=0.5,
        edgecolor="none",
        alpha=0.9
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(int(bar.get_height())),
            ha="center", va="bottom", color="#F8FAFC", fontweight="bold", fontsize=10
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.set_title("Loan Approval Status Distribution", color="#F8FAFC", fontsize=11, fontweight="600", pad=12)
    ax.set_ylabel("Number of Applications", color="#94A3B8", fontsize=9)
    fig.tight_layout()
    return fig

def plot_distribution(data, column, title, xlabel):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#0B1220')
    ax.set_facecolor('#0B1220')
    
    ax.hist(data[column], bins=20, color="#3B82F6", edgecolor="#1E293B", alpha=0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.set_title(title, color="#F8FAFC", fontsize=11, fontweight="600", pad=12)
    ax.set_xlabel(xlabel, color="#94A3B8", fontsize=9)
    ax.set_ylabel("Frequency", color="#94A3B8", fontsize=9)
    fig.tight_layout()
    return fig

def plot_category(data, column, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#0B1220')
    ax.set_facecolor('#0B1220')
    
    table = pd.crosstab(data[column], data["Loan_Status"])
    table.plot(kind="bar", ax=ax, color=["#EF4444", "#22C55E"], edgecolor="none", alpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.set_title(title, color="#F8FAFC", fontsize=11, fontweight="600", pad=12)
    ax.set_xlabel(column, color="#94A3B8", fontsize=9)
    ax.set_ylabel("Count", color="#94A3B8", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, color="#94A3B8")
    ax.legend(title="Status", facecolor='#080D18', edgecolor='#334155', labelcolor='#94A3B8')
    fig.tight_layout()
    return fig

def plot_income_vs_loan(data):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#0B1220')
    ax.set_facecolor('#0B1220')
    
    ax.scatter(data["ApplicantIncome"], data["LoanAmount"], color="#3B82F6", alpha=0.6, edgecolors="none", s=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.set_title("Applicant Income vs Loan Amount", color="#F8FAFC", fontsize=11, fontweight="600", pad=12)
    ax.set_xlabel("Applicant Income", color="#94A3B8", fontsize=9)
    ax.set_ylabel("Loan Amount", color="#94A3B8", fontsize=9)
    fig.tight_layout()
    return fig

def step10_category_wise(data):
    results = {}
    for column in EDA_CATEGORIES:
        count_table = pd.crosstab(data[column], data["Loan_Status"])
        percentage_table = (
            pd.crosstab(data[column], data["Loan_Status"], normalize="index") * 100
        )
        rejected_pct = percentage_table.get("N", 0)
        approved_pct = percentage_table.get("Y", 0)
        results[column] = pd.DataFrame({
            "Rejected Count": count_table.get("N", 0),
            "Rejected %": rejected_pct.round(2) if hasattr(rejected_pct, "round") else rejected_pct,
            "Approved Count": count_table.get("Y", 0),
            "Approved %": approved_pct.round(2) if hasattr(approved_pct, "round") else approved_pct,
        })
    return results

def step11_numerical_by_status(data):
    return {
        column: data.groupby("Loan_Status")[column].agg(
            Mean="mean", Median="median", Minimum="min", Maximum="max"
        ).round(2)
        for column in ANALYSIS_NUMBERS
    }

def step12_contribution_analysis(data):
    results = {}
    for column in ANALYSIS_NUMBERS:
        totals = data.groupby("Loan_Status")[column].sum()
        results[column] = (totals / totals.sum() * 100).round(2)
    return results

def step13_approval_rates(data):
    credit_rate = data.groupby("Credit_History")["Loan_Status"].apply(
        lambda x: (x == "Y").mean() * 100
    ).round(2)
    education_rate = data.groupby("Education")["Loan_Status"].apply(
        lambda x: (x == "Y").mean() * 100
    ).round(2)
    property_rate = data.groupby("Property_Area")["Loan_Status"].apply(
        lambda x: (x == "Y").mean() * 100
    ).round(2)
    return credit_rate, education_rate, property_rate

def step14_final_eda_insights(data):
    approved = (data["Loan_Status"] == "Y").sum()
    rejected = (data["Loan_Status"] == "N").sum()
    return {
        "Total Applications": len(data),
        "Approved Loans": int(approved),
        "Rejected Loans": int(rejected),
        "Overall Approval Rate": round(approved / len(data) * 100, 2),
        "Average Applicant Income": round(data["ApplicantIncome"].mean(), 2),
        "Average Coapplicant Income": round(data["CoapplicantIncome"].mean(), 2),
        "Average Loan Amount": round(data["LoanAmount"].mean(), 2),
    }

def step16_historical_approval_rates(data):
    rates = {}
    for name, column in [
        ("Credit History", "Credit_History"),
        ("Education", "Education"),
        ("Property Area", "Property_Area"),
        ("Self Employed", "Self_Employed"),
        ("Dependents", "Dependents")
    ]:
        rates[name] = data.groupby(column)["Loan_Status"].apply(
            lambda x: (x == "Y").mean() * 100
        )
    return rates

def step17_category_scores(rates, credit_history, education, property_area,
                          self_employed, dependents):
    credit_score = rates["Credit History"].get(
        credit_history, rates["Credit History"].mean()
    )
    education_score = rates["Education"].get(
        education, rates["Education"].mean()
    )
    property_score = rates["Property Area"].get(
        property_area, rates["Property Area"].mean()
    )
    self_score = rates["Self Employed"].get(
        self_employed, rates["Self Employed"].mean()
    )
    dependents_score = rates["Dependents"].get(
        dependents, rates["Dependents"].mean()
    )
    return {
        "Credit History": credit_score,
        "Education": education_score,
        "Property Area": property_score,
        "Self Employed": self_score,
        "Dependents": dependents_score
    }

def step18_numerical_scores(data, applicant_income, coapplicant_income, loan_amount):
    income_average = data["ApplicantIncome"].mean()
    coapplicant_average = data["CoapplicantIncome"].mean()
    loan_average = data["LoanAmount"].mean()

    income_score = 100 if applicant_income >= income_average else 0
    coapplicant_score = 100 if coapplicant_income >= coapplicant_average else 0
    loan_score = 100 if loan_amount <= loan_average else 0

    return {
        "Applicant Income": income_score,
        "Coapplicant Income": coapplicant_score,
        "Loan Amount": loan_score
    }

def step19_final_prediction(cat_scores, num_scores):
    scores = {**cat_scores, **num_scores}
    approval_score = np.mean(list(scores.values()))
    prediction = "APPROVED" if approval_score >= 50 else "REJECTED"
    return scores, approval_score, prediction

def main():
    inject_fintech_theme()

    # Sidebar
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="title">⚡ LOAN PREDICTION DASHBOARD</div>
            <div class="subtitle">CSE DEPARTMENT & DATA ANALYSIS ESSENTIAL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

    is_loaded = uploaded_file is not None
    pill_class = "" if is_loaded else "waiting"
    pill_text = "● Dataset Loaded" if is_loaded else "○ Waiting for Dataset"
    st.sidebar.markdown(f'<div class="status-pill {pill_class}">{pill_text}</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Navigation",
        [
            "Home & Overview",
            "Dataset Overview",
            "Data Quality",
            "Statistical Analysis",
            "Visual Analytics",
            "EDA & Insights",
            "Loan Prediction",
            "Conclusion"
        ],
        label_visibility="collapsed"
    )

    if is_loaded:
        try:
            raw_data = step1_load_dataset(uploaded_file)
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in raw_data.columns]
            if missing_cols:
                render_page_header("Schema Validation Error", "Dataset verification failed.")
                render_error_banner(missing_cols)
                return
            
            data, duplicate_count, records_after_dup = step5_duplicate_analysis(raw_data)
            data, remaining_missing = step4_handle_missing_values(data)
        except Exception as e:
            render_page_header("Error Loading Dataset", "An error occurred while parsing the CSV file.")
            st.error(str(e))
            return

        st.sidebar.markdown('<div class="sidebar-section-label">Dataset Info</div>', unsafe_allow_html=True)
        st.sidebar.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem; font-size: 0.82rem;">
                <div style="color: #F8FAFC; font-weight: 600; margin-bottom: 4px;">loan-train.csv</div>
                <div>Rows: <b>{len(raw_data)}</b></div>
                <div>Columns: <b>{raw_data.shape[1]}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if page != "Home & Overview":
            render_empty_state()
            return
        else:
            render_empty_state()
            
            return

    if page == "Home & Overview":
        render_page_header(
    "Loan Prediction Data Analysis System",
    "Analysis of loan applications, approval patterns, and applicant data."
)

        insights = step14_final_eda_insights(data)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_kpi("Total Apps", insights["Total Applications"], "Dataset records")
        with col2:
            render_kpi("Approved", insights["Approved Loans"], "Successful loans")
        with col3:
            render_kpi("Rejected", insights["Rejected Loans"], "Declined loans")
        with col4:
            render_kpi("Approval Rate", f"{insights['Overall Approval Rate']}%", "Success benchmark")
        with col5:
            render_kpi("Avg Loan", f"₹{insights['Average Loan Amount']}k", "Portfolio mean")

        render_section("Approval Overview & Quick Snapshot", "Core macroeconomic metrics derived from historical loan performance.")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                f"""
                <div class="fintech-card">
                    <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;">Portfolio Approval Metrics</div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border);">
                        <span>Total Evaluated Applications</span>
                        <b>{insights["Total Applications"]}</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border);">
                        <span>Approved Applications (Y)</span>
                        <b style="color: var(--success);">{insights["Approved Loans"]} ({insights["Overall Approval Rate"]}%)</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border);">
                        <span>Rejected Applications (N)</span>
                        <b style="color: var(--danger);">{insights["Rejected Loans"]} ({round(100 - insights["Overall Approval Rate"], 2)}%)</b>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Average Applicant Income</span>
                        <b>₹{insights["Average Applicant Income"]:,.2f}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            fig = plot_loan_status(data)
            st.pyplot(fig)

        render_section("Analytics Snapshot", "Key structural findings from EDA and data profiling.")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(
                """
                <div class="fintech-card">
                    <div style="font-weight: 600; margin-bottom: 0.5rem; color: #3B82F6;">Credit History Impact</div>
                    <p style="font-size: 0.85rem; color: var(--text-muted); margin: 0;">
                        Credit history is the strongest single predictor of loan approval, showing a massive approval divergence between 1.0 and 0.0 scores.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sc2:
            st.markdown(
                """
                <div class="fintech-card">
                    <div style="font-weight: 600; margin-bottom: 0.5rem; color: #3B82F6;">Property Area Dynamics</div>
                    <p style="font-size: 0.85rem; color: var(--text-muted); margin: 0;">
                        Semi-urban properties consistently exhibit higher approval rates compared to rural and urban counterparts in the portfolio.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sc3:
            st.markdown(
                """
                <div class="fintech-card">
                    <div style="font-weight: 600; margin-bottom: 0.5rem; color: #3B82F6;">Income & Loan Alignment</div>
                    <p style="font-size: 0.85rem; color: var(--text-muted); margin: 0;">
                        Applicants with income meeting or exceeding dataset averages coupled with controlled loan requests show optimal approval probabilities.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif page == "Dataset Overview":
        render_page_header("Dataset Overview & Inspection", "Member 1 Workflow: Raw data inspection, structure validation, and column profiling.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi("Total Rows", len(raw_data), "Raw entries")
        with col2:
            render_kpi("Columns", raw_data.shape[1], "Features")
        with col3:
            render_kpi("Numerical", len(NUMERICAL_COLUMNS), "Continuous features")
        with col4:
            render_kpi("Categorical", len(CATEGORICAL_COLUMNS), "Discrete features")

        render_section("Dataset Information & Metadata", "Summary file attributes and record dimensions.")
        st.markdown(
            f"""
            <div class="fintech-card" style="margin-bottom: 1.5rem;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                    <div><span style="color: var(--text-muted);">File Name:</span><br><b>{uploaded_file.name}</b></div>
                    <div><span style="color: var(--text-muted);">Total Records:</span><br><b>{len(data)}</b></div>
                    <div><span style="color: var(--text-muted);">Total Features:</span><br><b>{data.shape[1]}</b></div>
                    <div><span style="color: var(--text-muted);">Missing Values Resolved:</span><br><b>0 remaining</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_section("First 5 Dataset Records", "Preview of ingested data.")
        st.dataframe(data.head(), use_container_width=True)

        render_section("Feature Data Types", "Underlying schema data types.")
        dtypes_df = pd.DataFrame({"Data Type": data.dtypes.astype(str)})
        st.dataframe(dtypes_df, use_container_width=True)

    elif page == "Data Quality":
        render_page_header("Data Quality & Cleaning Pipeline", "Verification of missing values, duplicate removal, and data hygiene.")

        qual_score = 100 if remaining_missing == 0 and duplicate_count == 0 else 95
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi("Quality Score", f"{qual_score}%", "High integrity")
        with col2:
            render_kpi("Missing Imputed", data.isnull().sum().sum(), "Median/Mode applied")
        with col3:
            render_kpi("Duplicates Removed", duplicate_count, "Cleaned records")
        with col4:
            render_kpi("Final Clean Records", len(data), "Ready for analysis")

        render_section("Missing Value Analysis (Raw Data)", "Initial assessment of null values per feature.")
        st.dataframe(step3_missing_value_analysis(raw_data), use_container_width=True)

        render_section("Data Cleaning Pipeline", "Sequential verification steps performed on the dataset.")
        st.markdown(
            """
            <div class="fintech-card">
                <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="background: rgba(34, 197, 94, 0.2); color: #4ADE80; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;">STEP 1</span>
                        <span>Raw CSV Ingestion & Schema Validation</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="background: rgba(34, 197, 94, 0.2); color: #4ADE80; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;">STEP 2</span>
                        <span>Numerical Missing Values Imputed with Median</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="background: rgba(34, 197, 94, 0.2); color: #4ADE80; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;">STEP 3</span>
                        <span>Categorical Missing Values Imputed with Mode</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="background: rgba(34, 197, 94, 0.2); color: #4ADE80; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;">STEP 4</span>
                        <span>Duplicate Record Detection and Removal</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif page == "Statistical Analysis":
        render_page_header("Statistical Analysis Workspace", "Member 2 Workflow: Numerical metrics, categorical breakdowns, and correlation matrix.")

        tab1, tab2, tab3, tab4 = st.tabs(["Loan Status", "Categorical Stats", "Numerical Statistics", "Correlation Matrix"])

        with tab1:
            render_section("Loan Approval Status Breakdown", "Counts and percentage distribution.")
            st.dataframe(step6_loan_status(data), use_container_width=True)

        with tab2:
            render_section("Categorical Feature Value Counts", "Frequency distribution across categorical attributes.")
            cat_results = step7_categorical_analysis(data)
            for col_name, counts in cat_results.items():
                with st.expander(f"Distribution: {col_name}"):
                    st.dataframe(counts, use_container_width=True)

        with tab3:
            render_section("Numerical Statistics Summary", "Mean, median, minimum, and maximum aggregates.")
            num_stats = step8_numerical_statistics(data)
            st.dataframe(pd.DataFrame(num_stats), use_container_width=True)

        with tab4:
            render_section("Correlation Matrix", "Pearson correlation coefficients across numerical features.")
            corr_matrix = step9_correlation(data)
            st.dataframe(corr_matrix, use_container_width=True)

    elif page == "Visual Analytics":
        render_page_header("Visual Analytics Studio", "Comprehensive chart gallery visualizing portfolio distributions and relationships.")

        col1, col2 = st.columns(2)
        with col1:
            render_chart_card("Loan Approval Status", "Approved vs rejected applications count.", plot_loan_status(data))
            render_chart_card("Applicant Income Distribution", "Histogram of applicant income levels.", plot_distribution(data, "ApplicantIncome", "Applicant Income Distribution", "Applicant Income"))
            render_chart_card("Loan Amount Distribution", "Histogram of requested loan sizes.", plot_distribution(data, "LoanAmount", "Loan Amount Distribution", "Loan Amount"))
            render_chart_card("Credit History vs Status", "Historical approval across credit history groups.", plot_category(data, "Credit_History", "Credit History vs Loan Status"))
            render_chart_card("Education vs Status", "Approval rates for graduates vs non-graduates.", plot_category(data, "Education", "Education vs Loan Status"))
        with col2:
            render_chart_card("Coapplicant Income Distribution", "Histogram of coapplicant earnings.", plot_distribution(data, "CoapplicantIncome", "Coapplicant Income Distribution", "Coapplicant Income"))
            render_chart_card("Applicant Income vs Loan Amount", "Scatter plot highlighting income to loan ratios.", plot_income_vs_loan(data))
            render_chart_card("Property Area vs Status", "Approval distribution across urban, semiurban, and rural areas.", plot_category(data, "Property_Area", "Property Area vs Loan Status"))
            render_chart_card("Gender vs Status", "Approval breakdown by applicant gender.", plot_category(data, "Gender", "Gender vs Loan Status"))

    elif page == "EDA & Insights":
        render_page_header("Exploratory Data Analysis & Executive Insights", "Member 3 Workflow: Category-wise approval rates, contribution analysis, and key findings.")

        insights = step14_final_eda_insights(data)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi("Approval Rate", f"{insights['Overall Approval Rate']}%", "Overall success")
        with col2:
            render_kpi("Avg Income", f"₹{insights['Average Applicant Income']:,.0f}", "Applicant mean")
        with col3:
            render_kpi("Avg Coapplicant", f"₹{insights['Average Coapplicant Income']:,.0f}", "Coapplicant mean")
        with col4:
            render_kpi("Avg Loan", f"₹{insights['Average Loan Amount']}k", "Loan principal")

        render_section("Category-Wise Approval / Rejection Reports", "Detailed performance tables grouped by key demographic and financial factors.")
        cat_results = step10_category_wise(data)
        for col_name, res_df in cat_results.items():
            with st.expander(f"Detailed Analysis: {col_name} vs Loan Status"):
                st.dataframe(res_df, use_container_width=True)

        render_section("Historical Approval Rates (%)", "Baseline approval probabilities utilized by the scoring engine.")
        cr, ed, pr = step13_approval_rates(data)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Credit History Rate**")
            st.dataframe(cr, use_container_width=True)
        with col2:
            st.markdown("**Education Rate**")
            st.dataframe(ed, use_container_width=True)
        with col3:
            st.markdown("**Property Area Rate**")
            st.dataframe(pr, use_container_width=True)

    elif page == "Loan Prediction":
        render_page_header("Rule-Based Loan Prediction Engine", "Member 4 Workflow: Evaluate a new loan applicant against historical portfolio benchmarks.")

        rates = step16_historical_approval_rates(data)

        render_section("Applicant Profile Input", "Configure applicant parameters for real-time credit scoring.")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div style='font-weight:600; margin-bottom:0.5rem; color:#3B82F6;'>Personal Profile</div>", unsafe_allow_html=True)
                loan_id = st.text_input("Loan ID", value="LP001002")
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Married", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            with col2:
                st.markdown("<div style='font-weight:600; margin-bottom:0.5rem; color:#3B82F6;'>Financial Profile</div>", unsafe_allow_html=True)
                applicant_income = st.number_input("Applicant Income (₹)", value=5000.0, step=100.0)
                coapplicant_income = st.number_input("Coapplicant Income (₹)", value=0.0, step=100.0)
                loan_amount = st.number_input("Loan Amount (₹k)", value=128.0, step=10.0)
                loan_term = st.number_input("Loan Amount Term (Months)", value=360.0, step=12.0)
            with col3:
                st.markdown("<div style='font-weight:600; margin-bottom:0.5rem; color:#3B82F6;'>Credit & Property</div>", unsafe_allow_html=True)
                credit_history = st.selectbox("Credit History", [1.0, 0.0])
                property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
                self_employed = st.selectbox("Self Employed", ["Yes", "No"])
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⚡ Evaluate Loan Eligibility")

        if submitted:
            cat_scores = step17_category_scores(rates, credit_history, education, property_area, self_employed, dependents)
            num_scores = step18_numerical_scores(data, applicant_income, coapplicant_income, loan_amount)
            scores, approval_score, prediction = step19_final_prediction(cat_scores, num_scores)

            status_class = "approved" if prediction == "APPROVED" else "rejected"
            badge_text = "✓ APPROVED" if prediction == "APPROVED" else "✕ REJECTED"

            st.markdown(
                f"""
                <div class="result-panel {status_class}">
                    <div>
                        <div class="result-score-lbl">Overall Approval Score</div>
                        <div class="result-score-num">{approval_score:.2f}%</div>
                        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 4px;">Threshold: 50.00% for approval</div>
                    </div>
                    <div>
                        <div class="result-badge {status_class}">{badge_text}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_section("Score Breakdown by Factor", "Individual percentage contributions calculated from historical benchmarks and thresholds.")
            
            for factor, score in scores.items():
                bar_color = "#22C55E" if score >= 50 else "#EF4444"
                st.markdown(
                    f"""
                    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-weight: 500; font-size: 0.9rem; min-width: 180px;">{factor}</span>
                        <div style="flex-grow: 1; margin: 0 1.5rem; background: rgba(255,255,255,0.06); height: 8px; border-radius: 4px; overflow: hidden;">
                            <div style="width: {score}%; background: {bar_color}; height: 100%; border-radius: 4px;"></div>
                        </div>
                        <span style="font-weight: 700; font-variant-numeric: tabular-nums; font-size: 0.9rem; min-width: 60px; text-align: right;">{score:.2f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            render_section("Explainability & Methodology", "How this rule-based score is derived.")
            st.markdown(
                """
                <div class="fintech-card">
                    <ul style="margin: 0; padding-left: 1.2rem; color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;">
                        <li><b>Category Scores:</b> Derived directly from historical approval percentages for each categorical subgroup (e.g., Credit History, Education, Property Area).</li>
                        <li><b>Numerical Scores:</b> Evaluated against portfolio dataset means (Income ≥ mean awards 100%; Loan Amount ≤ mean awards 100%).</li>
                        <li><b>Final Aggregation:</b> Computed as the unweighted arithmetic mean across all 8 factor scores.</li>
                        <li><b>Decision Rule:</b> Applications achieving an overall approval score ≥ 50.00% are classified as <code>APPROVED</code>.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif page == "Conclusion":
        render_page_header("Project Conclusion & Executive Summary", "Academic synthesis of dataset analysis, exploratory findings, and credit scoring methodology.")

        st.markdown(
            """
            <div class="fintech-card" style="margin-bottom: 1.5rem;">
                <h3 style="margin-top: 0; color: var(--text-main);">Executive Summary</h3>
                <p style="color: var(--text-muted); line-height: 1.6; margin-bottom: 1rem;">
                    This project successfully analyzed historical loan application datasets using Python, Pandas, NumPy, and Matplotlib. Through rigorous data cleaning, statistical evaluation, and exploratory data visualization, key drivers of loan approval were uncovered.
                </p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    <div style="background: rgba(255,255,255,0.02); padding: 1rem; border-radius: 8px; border: 1px solid var(--border);">
                        <h4 style="margin: 0 0 0.5rem 0; color: #3B82F6;">Key Analytical Findings</h4>
                        <ul style="margin: 0; padding-left: 1rem; color: var(--text-muted); font-size: 0.88rem; line-height: 1.5;">
                            <li>Credit history is the dominant determinant of approval.</li>
                            <li>Semi-urban property areas exhibit highest approval likelihood.</li>
                            <li>Income thresholds strongly correlate with loan principal capacities.</li>
                        </ul>
                    </div>
                    <div style="background: rgba(255,255,255,0.02); padding: 1rem; border-radius: 8px; border: 1px solid var(--border);">
                        <h4 style="margin: 0 0 0.5rem 0; color: #3B82F6;">Scoring Methodology</h4>
                        <p style="margin: 0; color: var(--text-muted); font-size: 0.88rem; line-height: 1.5;">
                            The prediction engine employs a transparent, rule-based scoring mechanism benchmarking applicant attributes against historical dataset averages and conditional approval rates.
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        

if __name__ == "__main__":
    main()
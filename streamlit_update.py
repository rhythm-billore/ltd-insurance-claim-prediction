import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

st.set_page_config(layout="wide")

# ---------- Load trained pipeline + SHAP explainer ----------
@st.cache_resource
def load_model_and_explainer():
    # 1) load trained pipeline
    model = joblib.load("model/ltd_best_model_pipeline.pkl")

    # 2) split pipeline into preprocess + xgboost
    preprocess = model.named_steps["preprocess"]
    xgb_model = model.named_steps["model"]

    # 3) background data (optional; useful if you later want interventional SHAP)
    # NOTE: For TreeExplainer, background isn't strictly required, but keeping it is fine.
    X_bg = pd.read_parquet("data/shap_training_sample.parquet")
    X_bg_trans = preprocess.transform(X_bg)

    # 4) build TreeExplainer on the XGB model
    explainer = shap.TreeExplainer(xgb_model)

    # feature names after preprocessing
    feature_names = preprocess.get_feature_names_out()

    return model, explainer, preprocess, feature_names, X_bg_trans


model, explainer, preprocess, feature_names, X_bg_trans = load_model_and_explainer()


# ---------- Friendly feature names for UI ----------
def shap_feature_friendly(name: str) -> str:
    # remove "transformer__" prefix pattern
    base = name.split("__", 1)[-1]

    # Heuristic for one-hot names: "feature_value"
    # (works for typical OneHotEncoder output)
    if "_" in base:
        var, val = base.split("_", 1)
        var = var.replace("_", " ").title()
        val = val.replace("_", " ").replace("-", " to ").title()
        return f"{var}: {val}"

    # numeric continuous
    return base.replace("_", " ").title()


friendly_feature_map = {f: shap_feature_friendly(f) for f in feature_names}


# ---------- UI Header ----------
st.markdown(
    """
    <h1 style='text-align: center;color:#2B78E4;'>
        🛡️ Long-Term Disability Claim Incident Prediction
    </h1>
    """,
    unsafe_allow_html=True
)
st.write(
    "Please enter the details to assess the likelihood of a "
    "long-term disability claim incident during the exposure period."
)

left_col, right_col = st.columns([1, 1])


# ---------- Inputs ----------
with left_col:
    with st.form("ltd_input_form"):
        st.subheader("Demographics & Employment")
        gender = st.selectbox("Gender", ["Female", "Male"])

        states = ["IL", "TX", "NH", "CA", "RI", "ME", "CT", "NY", "FL", "MA",
                  "VA", "OH", "PA", "WA", "MI", "GA", "NC", "NJ", "CO", "VT"]
        state = st.selectbox("State", states, index=states.index("RI"))

        industries = ["Technology", "Education", "Healthcare", "Finance",
                      "Retail", "Construction", "PublicSector", "Manufacturing"]
        industry = st.selectbox("Industry", industries, index=industries.index("Technology"))

        age = st.number_input("Age", min_value=18, max_value=80, value=51)

        years_with_employer = st.number_input(
            "Years with employer", min_value=0.0, max_value=30.0, value=3.0, step=0.1
        )

        hours_list = ["20", "25", "30", "35", "40"]
        hours_worked_per_week = st.selectbox(
            "Hours worked per week",
            hours_list,
            index=hours_list.index("30")
        )

        statuses = ["FullTime", "PartTime"]
        employment_status = st.selectbox("Employment status", statuses, index=statuses.index("FullTime"))

        st.subheader("Plan Design & Benefit Info")
        durations = ["2 Years", "5 Years", "To Age 65"]
        benefit_duration = st.selectbox("Benefit duration", durations, index=durations.index("To Age 65"))

        salary_options = ["<40k", "40-60k", "60-80k", "80-120k", ">120k"]
        salary_band = st.selectbox("Salary band", salary_options, index=salary_options.index("40-60k"))

        coverage_options = ["Employer-paid", "Voluntary"]
        coverage_type = st.selectbox("Coverage type", coverage_options, index=coverage_options.index("Employer-paid"))

        integration_options = ["Primary", "IntegratedSSDI", "None"]
        integration_type = st.selectbox(
            "Integration type",
            integration_options,
            index=integration_options.index("IntegratedSSDI")
        )

        ssdi_offset_indicator = st.selectbox("SSDI offset indicator", ["0", "1"], index=0)

        st.subheader("Plan Parameters")
        occ_list = ["1", "2", "3", "4"]
        occupation_class = st.selectbox("Occupation class", occ_list, index=occ_list.index("1"))

        elim_list = ["30", "60", "90", "180"]
        elimination_period_days = st.selectbox("Elimination period (days)", elim_list, index=elim_list.index("90"))

        benefit_pct_list = ["0.4", "0.5", "0.6", "0.7"]
        benefit_pct = st.selectbox(
            "Benefit percentage (e.g., 0.6 for 60%)",
            benefit_pct_list,
            index=benefit_pct_list.index("0.6")
        )

        max_monthly_benefit = st.number_input(
            "Max monthly benefit",
            min_value=0.0, max_value=20000.0, value=3000.0, step=500.0,
        )

        st.subheader("Exposure Period")
        is_covid_period = st.selectbox(
            "Is COVID period? (exposure year >= 2020)",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

        submitted = st.form_submit_button("Predict claim incident probability")


# ---------- Helpers for Explanation ----------
def influence_strength(shap_value: float) -> str:
    """
    Converts |SHAP| into a friendly bucket.
    These thresholds are a heuristic—tune based on your model.
    """
    a = abs(shap_value)
    if a >= 0.20:
        return "Strong"
    elif a >= 0.08:
        return "Moderate"
    else:
        return "Small"


# ---------- Prediction + Explanation ----------
if submitted:
    input_dict = {
        "gender": gender,
        "state": state,
        "industry": industry,
        "benefit_duration": benefit_duration,
        "salary_band": salary_band,
        "coverage_type": coverage_type,
        "integration_type": integration_type,
        "ssdi_offset_indicator": ssdi_offset_indicator,
        "employment_status": employment_status,
        "is_covid_period": is_covid_period,
        "age": age,
        "occupation_class": occupation_class,
        "elimination_period_days": elimination_period_days,
        "benefit_pct": benefit_pct,
        "years_with_employer": years_with_employer,
        "max_monthly_benefit": max_monthly_benefit,
        "hours_worked_per_week": hours_worked_per_week,
    }

    input_df = pd.DataFrame([input_dict])

    proba = float(model.predict_proba(input_df)[:, 1][0])
    pred_class = int(model.predict(input_df)[0])
    proba_pct = proba * 100

    with right_col:
        # ----- Prediction card -----
        st.markdown("## Prediction")
        st.markdown("---")

        outcome_text = "🟢 Likely" if pred_class == 1 else "🔴 Unlikely"
        st.markdown(f"#### 📈 Estimated Claim Incident Probability: **{proba_pct:.1f}%**")
        st.markdown(f"#### 🔍 Claim incident: **{outcome_text}**")

        # ----- SHAP explanation for this row -----
        input_trans = preprocess.transform(input_df)

        # TreeExplainer explanation (log-odds contributions for typical XGB binary classifiers)
        shap_values_row = explainer(input_trans, check_additivity=False)

        vals = shap_values_row.values[0]
        names = np.array(feature_names)

        # Top 6 by absolute impact
        top_idx = np.argsort(np.abs(vals))[::-1][:6]

        # Build explanation rows
        expl_rows = []
        for i in top_idx:
            raw_name = names[i]
            friendly = friendly_feature_map.get(raw_name, raw_name)
            impact = float(vals[i])

            if impact > 0:
                arrow = "▲"
                effect = "increases risk"
            else:
                arrow = "▼"
                effect = "reduces risk"

            expl_rows.append({
                "Feature": friendly,
                "Effect": f"{arrow} {effect}",
                "Influence": influence_strength(impact),
                "shap_value": impact,
                "abs_shap": abs(impact),
            })

        # ----- Business-friendly explanation (no raw SHAP) -----
        st.markdown("")
        st.markdown("### 🧠 Explanation")
        st.caption("Top factors influencing this prediction (direction + relative influence).")

        for r in expl_rows:
            st.markdown(f"- **{r['Feature']}** — {r['Effect']} *(Influence: {r['Influence']})*")

        st.caption("Note: These factors describe *model influence*, not causation.")

        # ----- Technical details (expandable) -----
        with st.expander("🔧 Technical details (SHAP)", expanded=False):
            st.markdown("**Explanation computed using SHAP values (log-odds contributions).**")
            st.markdown(
                "• Positive SHAP → pushes the prediction toward **claim incident (1)**.\n"
                "• Negative SHAP → pushes the prediction toward **no claim incident (0)**.\n\n"
                "Tree-based binary classifiers commonly operate in **log-odds** space, so SHAP values add up in log-odds."
            )

            tech_df = pd.DataFrame({
                "Feature": [r["Feature"] for r in expl_rows],
                "Effect": [r["Effect"] for r in expl_rows],
                "Influence": [r["Influence"] for r in expl_rows],
                "SHAP (log-odds)": [r["shap_value"] for r in expl_rows],
                "Abs(SHAP)": [r["abs_shap"] for r in expl_rows],
            }).sort_values("Abs(SHAP)", ascending=False)

            st.dataframe(
                tech_df.style.format({"SHAP (log-odds)": "{:+.3f}", "Abs(SHAP)": "{:.3f}"}),
                use_container_width=True
            )

else:
    with right_col:
        st.markdown("## Prediction")
        st.write("Fill the form on the left and click **Predict** to see results here.")

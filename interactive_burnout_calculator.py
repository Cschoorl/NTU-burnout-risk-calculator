"""Interactive burnout calculator (Jupyter/ipywidgets version).

Run this in Jupyter Notebook/Lab:
    from interactive_burnout_calculator import run_app
    run_app()
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display
import ipywidgets as widgets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False


NUMERIC_COLS = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
]

SLEEP_DISORDERS = [
    "None",
    "Insomnia",
    "Sleep Apnea",
    "Restless Leg Syndrome",
    "Narcolepsy",
    "Unknown",
]

GENDERS = ["Male", "Female", "Unknown"]
BMI_CATEGORIES = ["Underweight", "Normal", "Overweight", "Obese", "Unknown"]

_state: dict[str, object] = {}
DEFAULT_DATASET_PATH = Path("/Users/caesarschoorl/Downloads/expanded_sleep_health_dataset.csv")


def _encode_with_fallback(series: pd.Series, encoder: LabelEncoder, allowed_values: list[str]) -> pd.Series:
    cleaned = series.astype(str).replace("", "Unknown").fillna("Unknown")
    cleaned = cleaned.where(cleaned.isin(allowed_values), "Unknown")
    return pd.Series(encoder.transform(cleaned), index=series.index)


def _parse_bp_columns(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    bp = series.astype(str).replace("None", "0/0").replace("", "0/0").fillna("0/0")
    bp_split = bp.str.split("/", n=1, expand=True)
    systolic = pd.to_numeric(bp_split[0], errors="coerce").fillna(0)
    diastolic = pd.to_numeric(bp_split[1], errors="coerce").fillna(0)
    return systolic, diastolic


def create_burnout_target(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        df["Stress Level"] >= 7,
        df["Quality of Sleep"] <= 5,
        df["Sleep Duration"] < 6.5,
        df["Physical Activity Level"] < 30,
        df["Heart Rate"] > 90,
        df["Sleep Disorder"] != "None",
        df["BMI Category"].isin(["Overweight", "Obese"]),
        df["Daily Steps"] < 5000,
    ]
    df = df.copy()
    df["Burnout Risk"] = (sum(conditions) >= 3).astype(int)
    return df


def _normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Name" not in df.columns:
        df["Name"] = None

    if "Person ID" in df.columns and "Name" in df.columns:
        cols = df.columns.tolist()
        cols.remove("Name")
        cols.insert(cols.index("Person ID") + 1, "Name")
        df = df[cols]

    for col in ["Gender", "BMI Category", "Occupation"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("Unknown")
                .astype(str)
                .replace("None", "Unknown")
                .replace("", "Unknown")
            )

    if "Sleep Disorder" in df.columns:
        df["Sleep Disorder"] = (
            df["Sleep Disorder"].fillna("Unknown").astype(str).replace("", "Unknown")
        )

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def prepare_features(df: pd.DataFrame):
    df_processed = df.copy()

    systolic, diastolic = _parse_bp_columns(df_processed["Blood Pressure"])
    df_processed["Systolic BP"] = systolic
    df_processed["Diastolic BP"] = diastolic

    le_gender = LabelEncoder()
    le_gender.fit(GENDERS)
    df_processed["Gender"] = _encode_with_fallback(df_processed["Gender"], le_gender, GENDERS)

    le_bmi = LabelEncoder()
    le_bmi.fit(BMI_CATEGORIES)
    df_processed["BMI Category"] = _encode_with_fallback(df_processed["BMI Category"], le_bmi, BMI_CATEGORIES)

    le_sleep = LabelEncoder()
    le_sleep.fit(SLEEP_DISORDERS)
    df_processed["Sleep Disorder"] = _encode_with_fallback(df_processed["Sleep Disorder"], le_sleep, SLEEP_DISORDERS)

    df_processed["Occupation"] = (
        df_processed["Occupation"]
        .astype(str)
        .replace("None", "Unknown")
        .replace("", "Unknown")
        .fillna("Unknown")
    )
    df_processed = pd.get_dummies(df_processed, columns=["Occupation"], prefix="Occ", drop_first=True)

    target = df_processed["Burnout Risk"]
    features = df_processed.drop(["Person ID", "Blood Pressure", "Burnout Risk", "Name"], axis=1)

    scaler = StandardScaler()
    numeric_with_bp = NUMERIC_COLS + ["Systolic BP", "Diastolic BP"]
    features[numeric_with_bp] = scaler.fit_transform(features[numeric_with_bp])
    return features, target, scaler, le_gender, le_bmi, le_sleep


def calculate_bmi(height: float, weight: float, height_unit: str = "cm", weight_unit: str = "kg"):
    if height_unit == "in":
        height = height * 2.54
    height_m = height / 100
    if weight_unit == "lb":
        weight = weight * 0.453592

    bmi = weight / (height_m**2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return round(float(bmi), 1), category


def _get_indicator_dot(metric: str, value: float) -> str:
    color = "grey"
    if metric == "Sleep Duration":
        color = "#16a34a" if 7.0 <= value <= 9.0 else "#f97316" if 6.5 <= value < 7.0 or value > 9.0 else "#dc2626"
    elif metric == "Quality of Sleep":
        color = "#16a34a" if value >= 7 else "#f97316" if value >= 5 else "#dc2626"
    elif metric == "Stress Level":
        color = "#16a34a" if value <= 4 else "#f97316" if value <= 6 else "#dc2626"
    elif metric == "Heart Rate":
        color = "#16a34a" if 60 <= value <= 100 else "#f97316"
    elif metric == "Physical Activity Level":
        color = "#16a34a" if value >= 60 else "#f97316" if value >= 30 else "#dc2626"
    elif metric == "Daily Steps":
        color = "#16a34a" if value >= 7500 else "#f97316" if value >= 5000 else "#dc2626"
    return f"<span style='height:10px;width:10px;background-color:{color};border-radius:50%;display:inline-block;margin-left:8px;'></span>"


def _predict_burnout(person_data: dict | pd.Series):
    person_df = pd.DataFrame([person_data])
    person = person_df.copy()
    systolic, diastolic = _parse_bp_columns(person["Blood Pressure"])
    person["Systolic BP"] = systolic
    person["Diastolic BP"] = diastolic

    person["Gender"] = (
        person["Gender"].astype(str).replace("None", "Unknown").replace("", "Unknown").fillna("Unknown")
    )
    person["BMI Category"] = (
        person["BMI Category"].astype(str).replace("None", "Unknown").replace("", "Unknown").fillna("Unknown")
    )
    person["Sleep Disorder"] = person["Sleep Disorder"].astype(str).replace("", "Unknown").fillna("Unknown")
    person["Occupation"] = (
        person["Occupation"].astype(str).replace("None", "Unknown").replace("", "Unknown").fillna("Unknown")
    )

    person["Gender"] = _encode_with_fallback(person["Gender"], _state["le_gender"], GENDERS)
    person["BMI Category"] = _encode_with_fallback(person["BMI Category"], _state["le_bmi"], BMI_CATEGORIES)
    person["Sleep Disorder"] = _encode_with_fallback(person["Sleep Disorder"], _state["le_sleep"], SLEEP_DISORDERS)

    person = pd.get_dummies(person, columns=["Occupation"], prefix="Occ", drop_first=True)
    features = person.drop(["Person ID", "Blood Pressure"], axis=1)
    x_cols = _state["X_cols"]

    for col in x_cols:
        if col not in features.columns:
            features[col] = 0
    features = features[[c for c in x_cols]]

    numeric_with_bp = NUMERIC_COLS + ["Systolic BP", "Diastolic BP"]
    features[numeric_with_bp] = _state["scaler"].transform(features[numeric_with_bp])

    predictions = {}
    for name, model in _state["models"].items():
        pred = int(model.predict(features.values)[0])
        pred_proba = float(model.predict_proba(features.values)[0][1])
        predictions[name] = {"prediction": pred, "probability": pred_proba}

    avg_prob = float(np.mean([p["probability"] for p in predictions.values()]))
    return predictions, avg_prob


def _display_person_results(person: pd.Series):
    _, avg_prob = _predict_burnout(person)
    conditions = [
        person["Stress Level"] >= 7,
        person["Quality of Sleep"] <= 5,
        person["Sleep Duration"] < 6.5,
        person["Physical Activity Level"] < 30,
        person["Heart Rate"] > 90,
        person["Sleep Disorder"] != "None",
        person["BMI Category"] in ["Overweight", "Obese"],
        person["Daily Steps"] < 5000,
    ]
    risk_factors = sum(conditions)

    if avg_prob >= 0.7:
        risk_label, risk_color = "HIGH RISK", "#CC3300"
    elif avg_prob >= 0.3:
        risk_label, risk_color = "MODERATE RISK", "#FFCC00"
    else:
        risk_label, risk_color = "LOW RISK", "#006633"

    person_id = int(person["Person ID"])
    person_name = person.get("Name")
    person_display = f"Person ID: {person_id}" if pd.isna(person_name) or person_name is None else f"{person_name} (ID: {person_id})"

    html = f"""
    <div style="background:#FFFFFF;border-radius:12px;padding:25px;margin:20px 0;border:1px solid #E0E0E0;">
      <h3 style="color:#003366;border-bottom:2px solid #00508C;padding-bottom:10px;">{person_display}</h3>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:15px;margin:20px 0;">
        <div><strong>Gender:</strong> {person['Gender']}</div>
        <div><strong>Age:</strong> {int(person['Age'])} years</div>
        <div><strong>Occupation:</strong> {person['Occupation']}</div>
        <div><strong>BMI:</strong> {person['BMI Category']}</div>
      </div>
      <div style="background:#EBF5F8;border-radius:10px;padding:25px;text-align:center;border:1px solid #D0E0E8;">
        <h2 style="color:{risk_color};margin:0;">{risk_label}</h2>
        <p style="font-size:40px;font-weight:bold;color:{risk_color};margin:10px 0;">{round(avg_prob * 100)}%</p>
        <p style="margin:0;">Based on {risk_factors} risk factor(s)</p>
      </div>
      <h4 style="color:#003366;margin-top:30px;">Health Metrics:</h4>
      <table style="width:100%;border-collapse:collapse;">
    """
    metrics = [
        ("Sleep Duration", person["Sleep Duration"], f"{person['Sleep Duration']} hours"),
        ("Quality of Sleep", person["Quality of Sleep"], f"{int(person['Quality of Sleep'])}/10"),
        ("Stress Level", person["Stress Level"], f"{int(person['Stress Level'])}/10"),
        ("Physical Activity Level", person["Physical Activity Level"], f"{int(person['Physical Activity Level'])} min/day"),
        ("Daily Steps", person["Daily Steps"], f"{int(person['Daily Steps'])} steps"),
        ("Heart Rate", person["Heart Rate"], f"{int(person['Heart Rate'])} bpm"),
    ]
    for label, raw, text in metrics:
        html += f"<tr><td style='padding:10px;border-bottom:1px solid #F8F8F8;'><strong>{label}</strong></td><td style='padding:10px;border-bottom:1px solid #F8F8F8;'>{text}{_get_indicator_dot(label, raw)}</td></tr>"
    html += "</table></div>"
    display(HTML(html))


def _display_header():
    display(
        HTML(
            """
            <div style="text-align:center;margin-bottom:30px;padding:20px;background-color:#F0F4F8;border-bottom:3px solid #00508C;border-radius:8px;">
                <h1 style="color:#003366;margin-bottom:10px;">Burnout Risk Assessment</h1>
                <p style="color:#555555;font-size:15px;">Analyze your burnout risk based on lifestyle and health factors.</p>
            </div>
            """
        )
    )


def _display_new_registration_modal():
    dataset = _state["dataset"]
    name_input = widgets.Text(description="Name:", placeholder="Full Name", layout=widgets.Layout(width="100%"))
    gender = widgets.Dropdown(options=["", "Male", "Female"], description="Gender:")
    age = widgets.IntSlider(value=30, min=18, max=80, description="Age:")
    occupation = widgets.Dropdown(options=[""] + sorted(dataset["Occupation"].unique().tolist()), description="Occupation:")
    sleep_duration = widgets.FloatSlider(value=7.0, min=4.5, max=10, step=0.5, description="Sleep (h):")
    sleep_quality = widgets.IntSlider(value=7, min=1, max=10, description="Sleep Quality:")
    sleep_disorder = widgets.Dropdown(options=SLEEP_DISORDERS[:-1], description="Sleep Disorder:")
    activity = widgets.IntSlider(value=60, min=0, max=180, step=10, description="Activity (min):")
    steps = widgets.IntSlider(value=8000, min=1000, max=20000, step=1000, description="Daily Steps:")
    stress = widgets.IntSlider(value=5, min=1, max=10, description="Stress Level:")
    bmi_category = widgets.Dropdown(options=[""] + BMI_CATEGORIES[:-1], description="BMI Category:")
    height = widgets.FloatText(value=170, description="Height:")
    height_unit = widgets.Dropdown(options=["cm", "in"], description="")
    weight = widgets.FloatText(value=70, description="Weight:")
    weight_unit = widgets.Dropdown(options=["kg", "lb"], description="")
    heart_rate = widgets.IntSlider(value=75, min=40, max=120, description="Heart Rate:")
    blood_pressure = widgets.Text(value="120/80", description="Blood Pressure:", placeholder="120/80")

    bmi_out, result_out = widgets.Output(), widgets.Output()
    bmi_button = widgets.Button(description="Calculate BMI", button_style="info")
    submit_button = widgets.Button(description="Calculate Risk", button_style="success")
    cancel_button = widgets.Button(description="Cancel", button_style="danger")

    def on_bmi(_):
        with bmi_out:
            clear_output()
            bmi, category = calculate_bmi(height.value, weight.value, height_unit.value, weight_unit.value)
            bmi_category.value = category
            display(HTML(f"<p style='color:#16a34a;'>BMI: <strong>{bmi}</strong> ({category})</p>"))

    def on_submit(_):
        with result_out:
            clear_output(wait=True)
            missing = []
            if not name_input.value:
                missing.append("Name")
            if not gender.value:
                missing.append("Gender")
            if not occupation.value:
                missing.append("Occupation")
            if not bmi_category.value:
                missing.append("BMI Category")
            if not blood_pressure.value or "/" not in blood_pressure.value:
                missing.append("Blood Pressure (XXX/YYY)")
            if missing:
                display(HTML(f"<p style='color:#dc2626;'>Missing fields: {', '.join(missing)}</p>"))
                return

            person = {
                "Person ID": int(_state["dataset"]["Person ID"].max()) + 1,
                "Name": name_input.value.strip(),
                "Gender": gender.value,
                "Age": age.value,
                "Occupation": occupation.value,
                "Sleep Duration": sleep_duration.value,
                "Quality of Sleep": sleep_quality.value,
                "Physical Activity Level": activity.value,
                "Stress Level": stress.value,
                "BMI Category": bmi_category.value,
                "Blood Pressure": blood_pressure.value,
                "Heart Rate": heart_rate.value,
                "Daily Steps": steps.value,
                "Sleep Disorder": sleep_disorder.value,
            }

            _display_person_results(pd.Series(person))
            new_row = pd.DataFrame([person])
            missing_cols = set(_state["dataset"].columns) - set(new_row.columns)
            for col in missing_cols:
                new_row[col] = None
            new_row = new_row[_state["dataset"].columns]
            _state["dataset"] = pd.concat([_state["dataset"], new_row], ignore_index=True).drop_duplicates(subset=["Person ID"], keep="last")
            _state["dataset"].to_csv(_state["csv_path"], index=False)
            display(HTML(f"<p style='color:#16a34a;'>Saved to {_state['csv_path']}</p>"))

    def on_cancel(_):
        clear_output(wait=True)
        _display_search_interface()

    bmi_button.on_click(on_bmi)
    submit_button.on_click(on_submit)
    cancel_button.on_click(on_cancel)

    clear_output(wait=True)
    display(
        widgets.VBox(
            [
                widgets.HTML("<h2>New Registration</h2>"),
                name_input,
                widgets.HTML("<h4>Personal Information</h4>"),
                widgets.HBox([gender, age]),
                occupation,
                widgets.HTML("<h4>Sleep & Rest</h4>"),
                sleep_duration,
                sleep_quality,
                sleep_disorder,
                widgets.HTML("<h4>Activity & Stress</h4>"),
                activity,
                steps,
                stress,
                widgets.HTML("<h4>Health Metrics</h4>"),
                widgets.HBox([height, height_unit]),
                widgets.HBox([weight, weight_unit]),
                bmi_button,
                bmi_out,
                bmi_category,
                heart_rate,
                blood_pressure,
                widgets.HBox([submit_button, cancel_button]),
                result_out,
            ]
        )
    )


def _display_search_interface():
    dataset = _state["dataset"]
    display(HTML("<h2>Search by Person ID or Name</h2>"))
    search_input = widgets.Text(description="Person ID or Name:", placeholder="e.g. 5 or John Doe")
    search_button = widgets.Button(description="Search", button_style="info")
    output = widgets.Output()

    def on_search(_):
        with output:
            clear_output(wait=True)
            query = search_input.value.strip()
            person = None
            try:
                pid = int(query)
                m = dataset["Person ID"] == pid
                if m.any():
                    person = dataset[m].iloc[0]
            except ValueError:
                if "Name" in dataset.columns:
                    m = dataset["Name"].astype(str).str.lower() == query.lower()
                    if m.any():
                        person = dataset[m].iloc[0]

            if person is not None:
                _display_person_results(person)
            else:
                display(HTML(f"<p style='color:#CC3300;'>Person '{query}' not found</p>"))
                reg_btn = widgets.Button(description="Register New", button_style="warning")
                reg_btn.on_click(lambda _: _display_new_registration_modal())
                display(reg_btn)

    search_button.on_click(on_search)
    display(widgets.HBox([search_input, search_button]))
    display(output)


def run_app(csv_path: str | Path = DEFAULT_DATASET_PATH):
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(
            "Dataset CSV not found.\n"
            f"Expected path: {csv_path}\n"
            "Check if the file exists at that location, or pass a custom path:\n"
            "run_app('/absolute/path/to/expanded_sleep_health_dataset.csv')"
        )

    dataset = pd.read_csv(csv_path)
    dataset = _normalize_dataset(dataset)
    dataset = create_burnout_target(dataset)

    X, y, scaler, le_gender, le_bmi, le_sleep = prepare_features(dataset)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
    for model in models.values():
        model.fit(X, y)

    _state.clear()
    _state.update(
        {
            "dataset": dataset,
            "csv_path": str(csv_path),
            "X_cols": X.columns.tolist(),
            "scaler": scaler,
            "le_gender": le_gender,
            "le_bmi": le_bmi,
            "le_sleep": le_sleep,
            "models": models,
        }
    )

    display(HTML("<p style='color:#16a34a;'>Interactive interface ready.</p>"))
    _display_header()
    _display_search_interface()


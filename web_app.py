from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from interactive_burnout_calculator import (
    BMI_CATEGORIES,
    DEFAULT_DATASET_PATH,
    GENDERS,
    NUMERIC_COLS,
    SLEEP_DISORDERS,
    _encode_with_fallback,
    _normalize_dataset,
    _parse_bp_columns,
    create_burnout_target,
    prepare_features,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False


app = Flask(__name__)

STATE: dict[str, object] = {}


def _train(csv_path: Path) -> None:
    dataset = pd.read_csv(csv_path)
    dataset = _normalize_dataset(dataset)
    dataset = create_burnout_target(dataset)

    x_train, y_train, scaler, le_gender, le_bmi, le_sleep = prepare_features(dataset)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
    for model in models.values():
        model.fit(x_train, y_train)

    STATE.clear()
    STATE.update(
        {
            "dataset": dataset,
            "csv_path": str(csv_path),
            "x_cols": x_train.columns.tolist(),
            "scaler": scaler,
            "le_gender": le_gender,
            "le_bmi": le_bmi,
            "le_sleep": le_sleep,
            "models": models,
        }
    )


def _predict(person: pd.Series) -> tuple[float, str, int]:
    person_df = pd.DataFrame([person.to_dict()])
    person_df["Systolic BP"], person_df["Diastolic BP"] = _parse_bp_columns(person_df["Blood Pressure"])

    person_df["Gender"] = _encode_with_fallback(person_df["Gender"], STATE["le_gender"], GENDERS)
    person_df["BMI Category"] = _encode_with_fallback(person_df["BMI Category"], STATE["le_bmi"], BMI_CATEGORIES)
    person_df["Sleep Disorder"] = _encode_with_fallback(person_df["Sleep Disorder"], STATE["le_sleep"], SLEEP_DISORDERS)
    person_df["Occupation"] = (
        person_df["Occupation"].astype(str).replace("None", "Unknown").replace("", "Unknown").fillna("Unknown")
    )

    person_df = pd.get_dummies(person_df, columns=["Occupation"], prefix="Occ", drop_first=True)
    features = person_df.drop(["Person ID", "Blood Pressure"], axis=1)

    for col in STATE["x_cols"]:
        if col not in features.columns:
            features[col] = 0
    features = features[[c for c in STATE["x_cols"]]]

    numeric_with_bp = NUMERIC_COLS + ["Systolic BP", "Diastolic BP"]
    features[numeric_with_bp] = STATE["scaler"].transform(features[numeric_with_bp])

    probs = []
    for model in STATE["models"].values():
        probs.append(float(model.predict_proba(features.values)[0][1]))

    avg_prob = float(np.mean(probs))
    risk_factors = sum(
        [
            person["Stress Level"] >= 7,
            person["Quality of Sleep"] <= 5,
            person["Sleep Duration"] < 6.5,
            person["Physical Activity Level"] < 30,
            person["Heart Rate"] > 90,
            person["Sleep Disorder"] != "None",
            person["BMI Category"] in ["Overweight", "Obese"],
            person["Daily Steps"] < 5000,
        ]
    )
    if avg_prob >= 0.7:
        label = "HIGH RISK"
    elif avg_prob >= 0.3:
        label = "MODERATE RISK"
    else:
        label = "LOW RISK"
    return avg_prob, label, int(risk_factors)


def _clamp_score(x: float) -> int:
    return int(max(1, min(100, round(x))))


def compute_dimension_scores(person: pd.Series) -> list[dict]:
    """Per dimension score 1-100; higher means healthier."""
    out: list[dict] = []

    stress = float(person["Stress Level"])
    # 1 = laag stress (beste) … 10 = hoog (slechtst)
    stress_h = 100.0 - (stress - 1.0) / 9.0 * 99.0
    out.append(
        {
            "key": "stress",
            "label": "Stress Level",
            "score": _clamp_score(stress_h),
            "hint": f"{int(stress)}/10",
        }
    )

    qsleep = float(person["Quality of Sleep"])
    out.append(
        {
            "key": "sleep_quality",
            "label": "Sleep Quality",
            "score": _clamp_score(qsleep * 10.0),
            "hint": f"{int(qsleep)}/10",
        }
    )

    dur = float(person["Sleep Duration"])
    if 7.0 <= dur <= 9.0:
        dur_h = 100.0
    elif 6.5 <= dur < 7.0 or 9.0 < dur <= 10.0:
        dur_h = 62.0
    elif dur < 6.5:
        dur_h = max(8.0, 100.0 - (6.5 - dur) * 22.0)
    else:
        dur_h = 35.0
    out.append(
        {
            "key": "sleep_duration",
            "label": "Sleep Duration (hours)",
            "score": _clamp_score(dur_h),
            "hint": f"{dur:.1f} h",
        }
    )

    act = float(person["Physical Activity Level"])
    act_h = min(100.0, act / 90.0 * 100.0)
    out.append(
        {
            "key": "activity",
            "label": "Physical Activity (min/day)",
            "score": _clamp_score(act_h),
            "hint": f"{int(act)} min/day",
        }
    )

    hr = float(person["Heart Rate"])
    if 60.0 <= hr <= 100.0:
        hr_h = 100.0 - min(35.0, abs(hr - 72.0) / 40.0 * 35.0)
    elif hr > 100.0:
        hr_h = max(12.0, 78.0 - (hr - 100.0) * 2.2)
    else:
        hr_h = max(18.0, 72.0 - (60.0 - hr) * 1.8)
    out.append(
        {
            "key": "heart_rate",
            "label": "Heart Rate",
            "score": _clamp_score(hr_h),
            "hint": f"{int(hr)} bpm",
        }
    )

    sd = str(person["Sleep Disorder"])
    sd_h = 100.0 if sd == "None" else 22.0
    out.append(
        {
            "key": "sleep_disorder",
            "label": "Sleep Disorder",
            "score": _clamp_score(sd_h),
            "hint": sd if sd != "None" else "None",
        }
    )

    bmi = str(person["BMI Category"])
    bmi_map = {
        "Underweight": 58.0,
        "Normal": 100.0,
        "Overweight": 52.0,
        "Obese": 28.0,
        "Unknown": 50.0,
    }
    bmi_h = bmi_map.get(bmi, 50.0)
    out.append(
        {
            "key": "bmi",
            "label": "BMI Category",
            "score": _clamp_score(bmi_h),
            "hint": bmi,
        }
    )

    steps = float(person["Daily Steps"])
    if steps >= 10000:
        steps_h = 100.0
    elif steps >= 5000:
        steps_h = 45.0 + (steps - 5000) / 5000.0 * 55.0
    else:
        steps_h = steps / 5000.0 * 42.0
    out.append(
        {
            "key": "steps",
            "label": "Daily Steps",
            "score": _clamp_score(steps_h),
            "hint": f"{int(steps)}",
        }
    )

    for row in out:
        s = int(row["score"])
        if s >= 67:
            row["tier"] = "good"
            row["color"] = "#22c55e"
        elif s >= 34:
            row["tier"] = "mid"
            row["color"] = "#f97316"
        else:
            row["tier"] = "bad"
            row["color"] = "#ef4444"

    return out


def age_bucket(age: int) -> tuple[int, int, str]:
    if age <= 24:
        return 18, 24, "18-24"
    if age <= 34:
        return 25, 34, "25-34"
    if age <= 44:
        return 35, 44, "35-44"
    if age <= 54:
        return 45, 54, "45-54"
    if age <= 64:
        return 55, 64, "55-64"
    return 65, 120, "65+"


def compute_cohort_dimension_means(dataset: pd.DataFrame, person_age: int, person_id: int, max_sample: int = 600) -> dict[str, float]:
    """Mean wellbeing score (1-100) per dimension for same age group."""
    lo, hi, _ = age_bucket(person_age)
    cohort = dataset[(dataset["Age"] >= lo) & (dataset["Age"] <= hi) & (dataset["Person ID"] != person_id)].copy()
    if cohort.empty:
        cohort = dataset[dataset["Person ID"] != person_id].copy()
    if len(cohort) > max_sample:
        cohort = cohort.sample(max_sample, random_state=42)

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for _, row in cohort.iterrows():
        for d in compute_dimension_scores(row):
            k = d["key"]
            sums[k] = sums.get(k, 0.0) + float(d["score"])
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def build_health_metrics_for_ui(person: pd.Series, dimensions: list[dict]) -> list[dict]:
    """Six metrics for UI chart with English labels."""
    key_to_en = {
        "sleep_duration": "Sleep Duration",
        "sleep_quality": "Quality of Sleep",
        "stress": "Stress Level",
        "activity": "Physical Activity Level",
        "steps": "Daily Steps",
        "heart_rate": "Heart Rate",
    }
    order = ["sleep_duration", "sleep_quality", "stress", "activity", "steps", "heart_rate"]
    by_key = {d["key"]: d for d in dimensions}
    out = []
    for k in order:
        d = dict(by_key[k])
        d["label_en"] = key_to_en[k]
        # human-readable values for UI
        if k == "sleep_duration":
            d["value_display"] = f"{float(person['Sleep Duration']):.1f} hours"
        elif k == "sleep_quality":
            d["value_display"] = f"{int(person['Quality of Sleep'])}/10"
        elif k == "stress":
            d["value_display"] = f"{int(person['Stress Level'])}/10"
        elif k == "activity":
            d["value_display"] = f"{int(person['Physical Activity Level'])} min/day"
        elif k == "steps":
            d["value_display"] = f"{int(person['Daily Steps'])} steps"
        else:
            d["value_display"] = f"{int(person['Heart Rate'])} bpm"
        out.append(d)
    return out


def build_age_comparison(metrics: list[dict], cohort_means: dict[str, float]) -> list[dict]:
    rows = []
    for d in metrics:
        k = d["key"]
        user_s = float(d["score"])
        peer = float(cohort_means.get(k, user_s))
        diff = round(user_s - peer, 1)
        rows.append(
            {
                "key": k,
                "label": d.get("label_en", d["label"]),
                "user_score": int(d["score"]),
                "peer_avg": round(peer, 1),
                "diff": diff,
                "better_than_peer": diff >= 0,
            }
        )
    return rows


def find_person(dataset: pd.DataFrame, query: str) -> pd.Series | None:
    q = query.strip()
    if not q:
        return None

    if q.isdigit():
        match = dataset[dataset["Person ID"] == int(q)]
        return None if match.empty else match.iloc[0]

    if "Name" not in dataset.columns:
        return None

    name_series = dataset["Name"].fillna("").astype(str)
    exact = dataset[name_series.str.lower() == q.lower()]
    if not exact.empty:
        return exact.iloc[0]

    partial = dataset[name_series.str.lower().str.contains(q.lower(), regex=False, na=False)]
    if not partial.empty:
        return partial.iloc[0]
    return None


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    success = None
    query = ""
    show_create_button = False

    created_id = request.args.get("created")
    if created_id:
        success = f"New person created successfully with ID {created_id}."

    if request.method == "POST":
        query = (request.form.get("person_lookup") or "").strip()
        if not query:
            error = "Please enter a name or Personal ID."
        else:
            dataset = STATE["dataset"]
            person = find_person(dataset, query)
            if person is None:
                error = f"Person '{query}' was not found."
                show_create_button = True
            else:
                avg_prob, label, risk_factors = _predict(person)
                dimensions = compute_dimension_scores(person)
                avg_wellbeing = int(round(sum(d["score"] for d in dimensions) / len(dimensions)))
                lo, hi, age_group_label = age_bucket(int(person["Age"]))
                cohort_means = compute_cohort_dimension_means(dataset, int(person["Age"]), int(person["Person ID"]))
                health_metrics = build_health_metrics_for_ui(person, dimensions)
                age_comparison = build_age_comparison(health_metrics, cohort_means)
                result = {
                    "person_id": int(person["Person ID"]),
                    "name": None if pd.isna(person.get("Name")) else person.get("Name"),
                    "probability": round(avg_prob * 100),
                    "label": label,
                    "risk_factors": risk_factors,
                    "age": int(person["Age"]),
                    "gender": person["Gender"],
                    "occupation": person["Occupation"],
                    "dimensions": dimensions,
                    "avg_wellbeing": avg_wellbeing,
                    "health_metrics": health_metrics,
                    "age_group_label": age_group_label,
                    "age_range": (f"{lo}-{hi}" if hi < 120 else f"{lo}+"),
                    "age_comparison": age_comparison,
                }

    return render_template(
        "index.html",
        result=result,
        error=error,
        success=success,
        query=query,
        show_create_button=show_create_button,
    )


@app.route("/new-person", methods=["GET", "POST"])
def new_person():
    dataset = STATE["dataset"]
    occupations = sorted(dataset["Occupation"].dropna().astype(str).unique().tolist())
    prefill = (request.args.get("prefill") or "").strip()
    form_error = None

    if request.method == "POST":
        try:
            name = (request.form.get("name") or "").strip()
            gender = (request.form.get("gender") or "").strip()
            age = int(request.form.get("age") or "0")
            occupation = (request.form.get("occupation") or "").strip()
            sleep_duration = float(request.form.get("sleep_duration") or "0")
            sleep_quality = int(request.form.get("sleep_quality") or "0")
            activity = int(request.form.get("activity") or "0")
            stress = int(request.form.get("stress") or "0")
            bmi = (request.form.get("bmi_category") or "").strip()
            blood_pressure = (request.form.get("blood_pressure") or "").strip()
            heart_rate = int(request.form.get("heart_rate") or "0")
            steps = int(request.form.get("daily_steps") or "0")
            sleep_disorder = (request.form.get("sleep_disorder") or "").strip()
        except ValueError:
            form_error = "Please check numeric fields (age, scores, steps, etc.)."
            return render_template("new_person.html", occupations=occupations, form_error=form_error, prefill=prefill)

        required = [name, gender, occupation, bmi, blood_pressure, sleep_disorder]
        if any(not v for v in required) or "/" not in blood_pressure:
            form_error = "Please fill in all required fields. Blood Pressure must be in format 120/80."
            return render_template("new_person.html", occupations=occupations, form_error=form_error, prefill=prefill)

        source_csv = Path(STATE["csv_path"])
        raw_df = pd.read_csv(source_csv)
        if "Name" not in raw_df.columns:
            raw_df["Name"] = None

        new_id = int(raw_df["Person ID"].max()) + 1
        row = pd.DataFrame(
            [
                {
                    "Person ID": new_id,
                    "Name": name,
                    "Gender": gender,
                    "Age": age,
                    "Occupation": occupation,
                    "Sleep Duration": sleep_duration,
                    "Quality of Sleep": sleep_quality,
                    "Physical Activity Level": activity,
                    "Stress Level": stress,
                    "BMI Category": bmi,
                    "Blood Pressure": blood_pressure,
                    "Heart Rate": heart_rate,
                    "Daily Steps": steps,
                    "Sleep Disorder": sleep_disorder,
                }
            ]
        )

        for col in raw_df.columns:
            if col not in row.columns:
                row[col] = None
        row = row[raw_df.columns]
        raw_df = pd.concat([raw_df, row], ignore_index=True)
        raw_df.to_csv(source_csv, index=False)

        _train(source_csv)
        return redirect(url_for("index", created=new_id))

    return render_template("new_person.html", occupations=occupations, form_error=form_error, prefill=prefill)


if __name__ == "__main__":
    csv_path = DEFAULT_DATASET_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            "Update the path in interactive_burnout_calculator.py or provide a custom path."
        )
    _train(csv_path)
    app.run(host="127.0.0.1", port=5001, debug=False)

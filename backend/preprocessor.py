"""
preprocessor.py - Converts LLM-extracted raw JSON into the feature vectors
expected by the pre-trained RF (341 cols), DT (341 cols), and SVM (315 cols) models.
Strategy:
  RF/DT: Map conversational answers to raw column names, build single-row
         DataFrame, apply get_dummies(), reindex to training_columns.
  SVM:   Match against actual bilingual dummy columns from the training set
         using a reverse-lookup index built from svm_training_columns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
NUMERIC_COLS = [
    "Age",
    "BMI",
    "Age at first period",
    "Duration of menstruation",
    "Age when the first symptoms appeared",
    "How often have you experienced the following in the last 6 months?",      # Feeling tired without reason
    "How often have you experienced the following in the last 6 months?_1",    # Feeling nervous or anxious
    "How often have you experienced the following in the last 6 months?_2",    # Trouble sleeping
    "How often have you experienced the following in the last 6 months?_3",    # Mood swings or irritability
    "How often have you experienced the following in the last 6 months?_4",    # Feeling depressed or hopeless
    "How often have you experienced the following in the last 6 months?_5",    # Loss of interest in activities
    "Perceived stress level",
    "Duration of lifestyle change",
    "Motivation to maintain lifestyle changes",
    "How Helpful Social Media advices are?",
    "How helpful AI advices are?",
]

NUMERIC_DEFAULTS = {
    "Age": 24.0,
    "BMI": 23.5,
    "Age at first period": 12.0,
    "Duration of menstruation": 5.0,
    "Age when the first symptoms appeared": 18.0,
    "How often have you experienced the following in the last 6 months?": 2.0,
    "How often have you experienced the following in the last 6 months?_1": 2.0,
    "How often have you experienced the following in the last 6 months?_2": 2.0,
    "How often have you experienced the following in the last 6 months?_3": 2.0,
    "How often have you experienced the following in the last 6 months?_4": 2.0,
    "How often have you experienced the following in the last 6 months?_5": 2.0,
    "Perceived stress level": 5.0,
    "Duration of lifestyle change": 0.0,
    "Motivation to maintain lifestyle changes": 5.0,
    "How Helpful Social Media advices are?": 3.0,
    "How helpful AI advices are?": 3.0,
}

CATEGORICAL_DEFAULTS = {
    "In which region or district, you have spent most of your life": "Dhaka_0",
    "Have you had these tests?": "None",
    "Did any methods helped": "None",
    "Are you using alternative treatments": "None",
    "Which Social Media platforms you usually follow for PCOS related advices?": "None",
    "Age Category": "Young (18-25yrs)",
    "BMI Category": "Normal weight",
    "Menstrual cycle pattern": "Regular",        # RF naming
    "Cycle_Pattern": "Regular",                   # DT naming
    "Delay between two periods": "1 month",       # RF naming
    "Period_Delay": "1 month",                    # DT naming
    "Acne severity": "None",
    "Hair Loss": "None",
    "Family history": "None",
    "Vitamin D level": "Have not tested",
    "Diagnosed with anxiety or depression?": "No",
    "Are you undergoing mental health treatment?": "None",
    "Do you practice stress relief activities?": "None",
    "Number of meals per day": "3",
    "Type of food consumed most often": "Home-cooked",
    "Sleep duration per night": "7–8 hours",
    "Sleep quality": "Moderate",
    "Physical activity level": "Light",
    "Exercise type regularly performed": "None",
    "Tried to lose weight in the last 1 year?": "No",
    "If yes, which methods?": "no",
    "Do you smoke?": "No",
    "Alcohol consumption": "Never",
    "Have you made any lifestyle modifications in the past year?": "None",
    "Noticed improvement in symptoms?": "No",
    "Are you on any fertility medications?": "No",
    "Do you have children?": "No",
    "If yes, did you face infertility before?": "No",
    "Have you ever asked any AI tool": "No",
    "Do you follow any Social Media nutritionist for PCOS related advise?": "No",
}

RF_SYMPTOM_OPTIONS = [
    "Acne or oily skin", "Amon kichu hoy ni", "Excess facial",
    "Fatigue", "Hair thinning", "Infertility",
    "Irregular periods", "Was not diagnosed till now", "Weight gain",
]

DT_SYMPTOM_OPTIONS = [
    "Acne or oily skin", "Amon kichu hoy ni", "Excess facial",
    "Fatigue", "Hair thinning", "Infertility",
    "Irregular periods", "Was not diagnosed till now", "Weight gain",
]

TEST_TIMING_OPTIONS = ["1 month ago", "2-3 months ago", "6 months ago", "More than 6 months ago", "Never"]


def _derive_age_category(age: float) -> str:
    if age < 18:
        return "Adolescence (12-18yrs)"
    elif age <= 25:
        return "Young (18-25yrs)"
    elif age < 40:
        return "Adult (25-40yrs)"
    else:
        return "Middle-Aged (40-64yrs)"


def _derive_bmi_category(bmi: float) -> str:
    """Derive BMI category from numeric BMI."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    elif bmi < 40:
        return "Obesity"
    else:
        return "Severe Obesity"


def _get_val(d: Dict[str, Any], key: str, default: Any) -> Any:
    """Helper to safely get a value, falling back to default if key is missing or value is None."""
    val = d.get(key)
    return val if val is not None else default

def _map_llm_json_to_raw(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map the LLM-extracted JSON (with simplified keys) to the raw column
    names expected by our one-hot pipeline.
    """
    raw = {}

    # ── Numeric fields ──
    raw["Age"] = float(_get_val(extracted, "age", NUMERIC_DEFAULTS["Age"]))
    raw["BMI"] = float(_get_val(extracted, "bmi", NUMERIC_DEFAULTS["BMI"]))
    raw["Age at first period"] = float(_get_val(extracted, "age_at_first_period", NUMERIC_DEFAULTS["Age at first period"]))
    raw["Duration of menstruation"] = float(_get_val(extracted, "menstrual_cycle_length_days", NUMERIC_DEFAULTS["Duration of menstruation"]))
    raw["Age when the first symptoms appeared"] = float(_get_val(extracted, "age_first_symptoms", NUMERIC_DEFAULTS["Age when the first symptoms appeared"]))

    # Mental health frequency (0-4 scale)
    raw["How often have you experienced the following in the last 6 months?"] = float(_get_val(extracted, "feeling_tired", NUMERIC_DEFAULTS["How often have you experienced the following in the last 6 months?"]))
    raw["How often have you experienced the following in the last 6 months?_1"] = float(_get_val(extracted, "feeling_anxious", NUMERIC_DEFAULTS["How often have you experienced the following in the last 6 months?_1"]))
    raw["How often have you experienced the following in the last 6 months?_2"] = float(_get_val(extracted, "trouble_sleeping", NUMERIC_DEFAULTS["How often have you experienced the following in the last 6 months?_2"]))
    raw["How often have you experienced the following in the last 6 months?_3"] = float(_get_val(extracted, "mood_swings", NUMERIC_DEFAULTS["How often have you experienced the following in the last 6 months?_3"]))
    raw["How often have you experienced the following in the last 6 months?_4"] = float(_get_val(extracted, "feeling_depressed", NUMERIC_DEFAULTS["How often have you experienced the following in the last 6 months?_4"]))
    raw["How often have you experienced the following in the last 6 months?_5"] = float(_get_val(extracted, "loss_of_interest", NUMERIC_DEFAULTS["How often have you experienced the following in the last 6 months?_5"]))

    raw["Perceived stress level"] = float(_get_val(extracted, "stress_level", NUMERIC_DEFAULTS["Perceived stress level"]))
    raw["Duration of lifestyle change"] = float(_get_val(extracted, "lifestyle_change_months", NUMERIC_DEFAULTS["Duration of lifestyle change"]))
    raw["Motivation to maintain lifestyle changes"] = float(_get_val(extracted, "motivation", NUMERIC_DEFAULTS["Motivation to maintain lifestyle changes"]))
    
    # Social media / AI helpfulness — default to 3 (neutral)
    raw["How Helpful Social Media advices are?"] = float(_get_val(extracted, "social_media_helpfulness", NUMERIC_DEFAULTS["How Helpful Social Media advices are?"]))
    raw["How helpful AI advices are?"] = float(_get_val(extracted, "ai_helpfulness", NUMERIC_DEFAULTS["How helpful AI advices are?"]))

    # ── Derived categories ──
    raw["Age Category"] = _derive_age_category(raw["Age"])
    raw["BMI Category"] = _derive_bmi_category(raw["BMI"])

    # ── Single-select categorical ──
    raw["Menstrual cycle pattern"] = _get_val(extracted, "cycle_pattern", "Regular")
    raw["Cycle_Pattern"] = _get_val(extracted, "cycle_pattern", "Regular")  # DT naming

    raw["Delay between two periods"] = _get_val(extracted, "period_delay", "1 month")
    raw["Period_Delay"] = _get_val(extracted, "period_delay", "1 month")  # DT naming

    raw["Acne severity"] = _get_val(extracted, "acne_severity", "None")
    raw["Hair Loss"] = _get_val(extracted, "hair_loss", "None")
    raw["Vitamin D level"] = _get_val(extracted, "vitamin_d", "Have not tested")

    raw["Diagnosed with anxiety or depression?"] = _get_val(extracted, "anxiety_depression_diagnosed", "No")
    raw["Are you undergoing mental health treatment?"] = _get_val(extracted, "mental_health_treatment", "None")
    raw["Do you practice stress relief activities?"] = _get_val(extracted, "stress_relief", "None")

    raw["Number of meals per day"] = str(_get_val(extracted, "meals_per_day", "3"))
    raw["Type of food consumed most often"] = _get_val(extracted, "food_type", "Home-cooked")

    # Normalize sleep hours format
    sleep = _get_val(extracted, "sleep_hours", "7–8 hours")
    sleep = sleep.replace("-", "–")  # normalize dash
    raw["Sleep duration per night"] = sleep

    raw["Sleep quality"] = _get_val(extracted, "sleep_quality", "Moderate")
    raw["Physical activity level"] = _get_val(extracted, "activity_level", "Light")
    raw["Exercise type regularly performed"] = _get_val(extracted, "exercise_type", "None")
    raw["Tried to lose weight in the last 1 year?"] = _get_val(extracted, "tried_weight_loss", "No")
    raw["If yes, which methods?"] = _get_val(extracted, "weight_loss_method", "no")
    raw["Do you smoke?"] = _get_val(extracted, "smoking", "No")
    raw["Alcohol consumption"] = _get_val(extracted, "alcohol", "Never")
    raw["Have you made any lifestyle modifications in the past year?"] = _get_val(extracted, "lifestyle_modifications", "None")
    raw["Noticed improvement in symptoms?"] = _get_val(extracted, "symptom_improvement", "No")
    raw["Are you using alternative treatments"] = _get_val(extracted, "alternative_treatments", "None")
    raw["Are you on any fertility medications?"] = _get_val(extracted, "fertility_meds", "No")
    raw["Do you have children?"] = _get_val(extracted, "has_children", "No")
    raw["If yes, did you face infertility before?"] = _get_val(extracted, "faced_infertility", "No")

    raw["In which region or district, you have spent most of your life"] = _get_val(extracted, "region", "Dhaka_0")
    raw["Have you had these tests?"] = _get_val(extracted, "tests_done", "None")
    raw["Did any methods helped"] = _get_val(extracted, "helpful_methods", "None")

    raw["Have you ever asked any AI tool"] = _get_val(extracted, "asked_ai_tool", "No")
    raw["Do you follow any Social Media nutritionist for PCOS related advise?"] = _get_val(extracted, "follow_social_media_nutritionist", "No")
    raw["Which Social Media platforms you usually follow for PCOS related advices?"] = _get_val(extracted, "social_media_platforms", "None")

    return raw


def _build_symptom_columns(symptoms: List[str], prefix_rf: str, prefix_dt: str) -> Dict[str, int]:
    """Build binary columns for multi-select symptom fields."""
    result = {}
    symptom_list = symptoms if isinstance(symptoms, list) else []
    
    for opt in RF_SYMPTOM_OPTIONS:
        col_rf = f"{prefix_rf}_{opt}"
        col_dt = f"{prefix_dt}_{opt}"
        val = 1 if opt in symptom_list else 0
        result[col_rf] = val
        result[col_dt] = val
    
    return result


def _build_family_history_columns(history: List[str]) -> Dict[str, int]:
    """Build binary columns for family history multi-select."""
    options = [
        "Endometriosis later endometric cancer", "Gestational Diabetes",
        "Hypertension", "My mother was never really diagnosed but she used to have similar symptoms",
        "None", "Obesity", "PCOS", "Thelassemia", "Thyroid issues",
        "Type-2 Diabetes", "not sure",
    ]
    result = {}
    hist_list = history if isinstance(history, list) else []
    for opt in options:
        result[f"Family history_{opt}"] = 1 if opt in hist_list else 0
    return result


def _build_test_timing_columns(test_timings: Dict[str, str]) -> Dict[str, int]:
    """
    Build binary columns for 'When were these tests done last time?'
    There are 7 sub-tests (indices '', _1, _2, _3, _4, _5, _6).
    """
    result = {}
    suffixes = ["", "_1", "_2", "_3", "_4", "_5", "_6"]
    
    for suffix in suffixes:
        key = f"test_timing{suffix}" if suffix else "test_timing_0"
        user_val = test_timings.get(key, "Never")
        for opt in TEST_TIMING_OPTIONS:
            col_name = f"When were these tests done last time?{suffix}_{opt}"
            result[col_name] = 1 if opt == user_val else 0
    
    return result


def preprocess_for_model(
    extracted_json: Dict[str, Any],
    training_columns: List[str],
) -> pd.DataFrame:
    """
    Convert LLM-extracted JSON → a single-row DataFrame with the exact
    columns the model expects, in the correct order.
    
    Args:
        extracted_json: Raw dict from LLM extraction.
        training_columns: List of column names the model was trained on.
    
    Returns:
        A (1, N) DataFrame aligned to training_columns.
    """
    # Step 1: Map to raw column names
    raw = _map_llm_json_to_raw(extracted_json)
    
    # Step 2: Build a single-row DataFrame from raw values
    row = {}
    
    # Numeric columns
    for col in NUMERIC_COLS:
        row[col] = raw.get(col, NUMERIC_DEFAULTS.get(col, 0.0))
    
    # Step 3: One-hot encode categorical columns
    # For each categorical column, set the matching dummy column to 1
    categorical_raw_keys = [
        "In which region or district, you have spent most of your life",
        "Have you had these tests?",
        "Did any methods helped",
        "Are you using alternative treatments",
        "Which Social Media platforms you usually follow for PCOS related advices?",
        "Age Category",
        "BMI Category",
        "Menstrual cycle pattern",
        "Cycle_Pattern",
        "Delay between two periods",
        "Period_Delay",
        "Acne severity",
        "Hair Loss",
        "Vitamin D level",
        "Diagnosed with anxiety or depression?",
        "Are you undergoing mental health treatment?",
        "Do you practice stress relief activities?",
        "Number of meals per day",
        "Type of food consumed most often",
        "Sleep duration per night",
        "Sleep quality",
        "Physical activity level",
        "Exercise type regularly performed",
        "Tried to lose weight in the last 1 year?",
        "If yes, which methods?",
        "Do you smoke?",
        "Alcohol consumption",
        "Have you made any lifestyle modifications in the past year?",
        "Noticed improvement in symptoms?",
        "Are you on any fertility medications?",
        "Do you have children?",
        "If yes, did you face infertility before?",
        "Have you ever asked any AI tool",
        "Do you follow any Social Media nutritionist for PCOS related advise?",
    ]
    
    for cat_key in categorical_raw_keys:
        value = raw.get(cat_key, CATEGORICAL_DEFAULTS.get(cat_key, ""))
        # Set the one-hot column: "Category_Value" = 1
        dummy_col = f"{cat_key}_{value}"
        row[dummy_col] = 1
    
    # Step 4: Multi-select fields (symptoms)
    symptoms = extracted_json.get("symptoms", [])
    if isinstance(symptoms, str):
        symptoms = [s.strip() for s in symptoms.split(",")]
    
    symptom_cols = _build_symptom_columns(
        symptoms,
        "Symptoms in the last 6 months",  # RF prefix
        "Symptom",                         # DT prefix
    )
    row.update(symptom_cols)
    
    # Family history
    family = extracted_json.get("family_history", [])
    if isinstance(family, str):
        family = [f.strip() for f in family.split(",")]
    family_cols = _build_family_history_columns(family)
    row.update(family_cols)
    
    # Test timing columns
    test_timings = extracted_json.get("test_timings", {})
    if not isinstance(test_timings, dict):
        test_timings = {}
    timing_cols = _build_test_timing_columns(test_timings)
    row.update(timing_cols)
    
    # Step 5: Social media platform columns (Uses_AI, Uses_Social, Platform_*)
    # These are survey-only — default to 0
    for prefix in ["Uses_AI", "Uses_Social", "Platform_Instagram", 
                    "Platform_Reddit", "Platform_Facebook", "Platform_TikTok"]:
        row[prefix] = 0
    
    # Step 6: Create DataFrame and reindex to training columns
    df = pd.DataFrame([row])
    df_aligned = df.reindex(columns=training_columns, fill_value=0)
    
    # Ensure all values are numeric
    df_aligned = df_aligned.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return df_aligned


def build_collected_data_row(
    extracted_json: Dict[str, Any],
    rf_prediction: str,
    dt_prediction: str,
    svm_prediction: str = "N/A",
    final_verdict: str = "N/A",
) -> Dict[str, Any]:
    """
    Build a flat row dict suitable for appending to collected_pcos_data.csv.
    Includes the raw extracted values plus model predictions.
    """
    row = dict(extracted_json)
    row["rf_prediction"] = rf_prediction
    row["dt_prediction"] = dt_prediction
    row["svm_prediction"] = svm_prediction
    row["final_verdict"] = final_verdict
    return row


# ── SVM-specific preprocessing ──────────────────────────────────────────────

def build_svm_col_index(svm_cols: List[str], cat_prefixes: List[str]) -> Dict[str, List[str]]:
    """
    Build a reverse-lookup index: {column_prefix: [suffix_values]}
    from the actual svm_cols list.
    """
    col_index: Dict[str, List[str]] = {}
    for sc in svm_cols:
        if sc.startswith("Age Category_"):
            col_index.setdefault("Age Category", []).append(sc[len("Age Category_"):])
            continue
        if sc.startswith("BMI Category_"):
            col_index.setdefault("BMI Category", []).append(sc[len("BMI Category_"):])
            continue
        matched = False
        for prefix in cat_prefixes:
            if sc.startswith(prefix):
                rest = sc[len(prefix):]
                if rest.startswith("_"):
                    col_index.setdefault(prefix, []).append(rest[1:])
                    matched = True
                    break
                elif rest.startswith(":_"):
                    col_index.setdefault(prefix, []).append(rest[2:])
                    matched = True
                    break
        # Also capture Family history dummy columns
        if not matched and 'Family history' in sc and 'biological' in sc.lower():
            idx = sc.find(':_')
            if idx > 0:
                suffix = sc[idx + 2:]
                col_index.setdefault("__family_dummy__", []).append(suffix)
    return col_index


def _find_exact_match(value: str, candidates: List[str]):
    """Find exact or case-insensitive match for value in candidates."""
    val_str = value.strip()
    for c in candidates:
        if c.strip() == val_str:
            return c
    for c in candidates:
        if c.strip().lower() == val_str.lower():
            return c
    return None


def preprocess_for_svm(
    extracted_json: Dict[str, Any],
    svm_cols: List[str],
    svm_col_index: Dict[str, List[str]],
    numeric_col_map: Dict[str, tuple],
    categorical_col_map: Dict[str, str],
    symptom_col: str,
    family_dummy_prefix: str,
    timing_cols: List[str],
) -> pd.DataFrame:
    """
    Convert LLM-extracted JSON into a (1, N) DataFrame aligned to svm_cols.
    Uses the dynamically-built column index to match dummy column names.
    """
    row = {}

    # 1. Numeric features
    for raw_col, (json_key, default) in numeric_col_map.items():
        val = extracted_json.get(json_key)
        if val is None:
            val = default
        try:
            row[raw_col] = float(val)
        except (ValueError, TypeError):
            row[raw_col] = default

    # 2. Derived categories
    numeric_keys = list(numeric_col_map.keys())
    age = row.get(numeric_keys[0], 24.0) if numeric_keys else 24.0
    bmi = row.get(numeric_keys[1], 23.5) if len(numeric_keys) > 1 else 23.5
    row[f"Age Category_{_derive_age_category(age)}"] = 1
    row[f"BMI Category_{_derive_bmi_category(bmi)}"] = 1

    # 3. Categorical features — match against svm_cols candidates
    for raw_col, json_key in categorical_col_map.items():
        val = extracted_json.get(json_key)
        if val is None:
            continue
        val_str = str(val).strip()
        if not val_str:
            continue
        candidates = svm_col_index.get(raw_col, [])
        if not candidates:
            continue
        matched = _find_exact_match(val_str, candidates)
        if matched is not None:
            row[f"{raw_col}_{matched}"] = 1
        else:
            # Try direct column construction
            for sep in ["_", ":_"]:
                dummy_col = f"{raw_col}{sep}{val_str}"
                if dummy_col in svm_cols:
                    row[dummy_col] = 1
                    break

    # 4. Symptoms — keep as single joined string
    symptoms = extracted_json.get("symptoms", [])
    if isinstance(symptoms, str):
        symptoms = [s.strip() for s in symptoms.split(",")]
    if symptoms and symptom_col:
        symptom_val_str = ", ".join(str(s).strip() for s in symptoms)
        candidates = svm_col_index.get(symptom_col, [])
        matched = _find_exact_match(symptom_val_str, candidates)
        if matched:
            row[f"{symptom_col}_{matched}"] = 1
        else:
            dummy_col = f"{symptom_col}_{symptom_val_str}"
            if dummy_col in svm_cols:
                row[dummy_col] = 1

    # 5. Family history
    family = extracted_json.get("family_history", [])
    if isinstance(family, str):
        family = [f.strip() for f in family.split(",")]
    if family and family_dummy_prefix:
        family_val_str = ", ".join(str(f).strip() for f in family)
        candidates = svm_col_index.get("__family_dummy__", [])
        matched = _find_exact_match(family_val_str, candidates)
        if matched:
            row[f"{family_dummy_prefix}_{matched}"] = 1
        else:
            dummy_col = f"{family_dummy_prefix}_{family_val_str}"
            if dummy_col in svm_cols:
                row[dummy_col] = 1

    # 6. Test timings
    test_timings = extracted_json.get("test_timings", {})
    if not isinstance(test_timings, dict):
        test_timings = {}
    for i, tc in enumerate(timing_cols):
        if tc is None:
            continue
        timing_key = f"test_timing_{i}"
        user_val = test_timings.get(timing_key, "Never")
        candidates = svm_col_index.get(tc, [])
        matched = _find_exact_match(str(user_val), candidates)
        if matched:
            row[f"{tc}_{matched}"] = 1
        else:
            never_match = _find_exact_match("Never", candidates)
            if never_match:
                row[f"{tc}_{never_match}"] = 1

    df = pd.DataFrame([row])
    df_aligned = df.reindex(columns=svm_cols, fill_value=0)
    df_aligned = df_aligned.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df_aligned


def preprocess_for_svm_default(
    extracted_json: Dict[str, Any],
    svm_cols: List[str]
) -> pd.DataFrame:
    """
    Helper function to preprocess extracted JSON for the SVM model.
    Dynamically maps the input fields to the bilingual columns in svm_cols.
    """
    # Define the numeric mapping
    NUMERIC_COL_MAP = {
        "Age (years)": ("age", 24.0),
        "BMI (kg/m^2)": ("bmi", 23.5),
        "Duration of menstruation (days) / মাসিক চলার সময়কাল (দিন):": ("menstrual_cycle_length_days", 5.0),
        "Age when the first symptoms appeared / প্রথম লক্ষণ দেখা দেওয়ার সময় বয়স": ("age_first_symptoms", 18.0),
        "How often have you experienced the following in the last 6 months? ( 0 = Never, 1 = Rarely, 2 = Sometimes, 3 = Often, 4 = Always) / \nগত ৬ মাসে নিচের অভিজ্ঞতাগুলো কত ঘন ঘন হয়েছে? (০ = কখনও না, ১ = খুব কম, ২ = মাঝে মাঝে, ৩ = প্রায়ই, ৪ = সবসময়) [Feeling tired without reason / কারণ ছাড়া ক্লান্ত বোধ করা:]": ("feeling_tired", 2.0),
        "How often have you experienced the following in the last 6 months? ( 0 = Never, 1 = Rarely, 2 = Sometimes, 3 = Often, 4 = Always) / \nগত ৬ মাসে নিচের অভিজ্ঞতাগুলো কত ঘন ঘন হয়েছে? (০ = কখনও না, ১ = খুব কম, ২ = মাঝে মাঝে, ৩ = প্রায়ই, ৪ = সবসময়) [Feeling nervous or anxious / নার্ভাস বা উদ্বিগ্ন অনুভব:]": ("feeling_anxious", 2.0),
        "How often have you experienced the following in the last 6 months? ( 0 = Never, 1 = Rarely, 2 = Sometimes, 3 = Often, 4 = Always) / \nগত ৬ মাসে নিচের অভিজ্ঞতাগুলো কত ঘন ঘন হয়েছে? (০ = কখনও না, ১ = খুব কম, ২ = মাঝে মাঝে, ৩ = প্রায়ই, ৪ = সবসময়) [Trouble sleeping / ঘুমের সমস্যা:]": ("trouble_sleeping", 2.0),
        "How often have you experienced the following in the last 6 months? ( 0 = Never, 1 = Rarely, 2 = Sometimes, 3 = Often, 4 = Always) / \nগত ৬ মাসে নিচের অভিজ্ঞতাগুলো কত ঘন ঘন হয়েছে? (০ = কখনও না, ১ = খুব কম, ২ = মাঝে মাঝে, ৩ = প্রায়ই, ৪ = সবসময়) [Mood swings or irritability / মেজাজ পরিবর্তন বা বিরক্তিভাব]": ("mood_swings", 2.0),
        "How often have you experienced the following in the last 6 months? ( 0 = Never, 1 = Rarely, 2 = Sometimes, 3 = Often, 4 = Always) / \nগত ৬ মাসে নিচের অভিজ্ঞতাগুলো কত ঘন ঘন হয়েছে? (০ = কখনও না, ১ = খুব কম, ২ = মাঝে মাঝে, ৩ = প্রায়ই, ৪ = সবসময়) [Feeling depressed or hopeless / হতাশ বা নিরাশ অনুভব]": ("feeling_depressed", 2.0),
        "How often have you experienced the following in the last 6 months? ( 0 = Never, 1 = Rarely, 2 = Sometimes, 3 = Often, 4 = Always) / \nগত ৬ মাসে নিচের অভিজ্ঞতাগুলো কত ঘন ঘন হয়েছে? (০ = কখনও না, ১ = খুব কম, ২ = মাঝে মাঝে, ৩ = প্রায়ই, ৪ = সবসময়) [Loss of interest in activities / আগ্রহের কাজে মন না বসা]": ("loss_of_interest", 2.0),
        "Perceived stress level (1–10) / আপনার অনুভূত মানসিক চাপের মাত্রা (১–১০):": ("stress_level", 5.0),
        "Duration of lifestyle change (months) / এই পরিবর্তনের সময়কাল (মাস):": ("lifestyle_change_months", 0.0),
        "Motivation to maintain lifestyle changes (1–10) / পরিবর্তন চালিয়ে যাওয়ার ইচ্ছাশক্তি কত (১–১০):": ("motivation", 5.0),
        "How Helpful Social Media advices are?": ("social_media_helpfulness", 3.0),
        "How helpful AI advices are?": ("ai_helpfulness", 3.0),
    }

    numeric_col_map = {}
    for c in svm_cols:
        for possible_key, (json_key, default) in NUMERIC_COL_MAP.items():
            if possible_key.replace('\n', ' ').strip().lower() == c.replace('\n', ' ').strip().lower():
                numeric_col_map[c] = (json_key, default)
                break

    # Define the categorical mapping
    CATEGORICAL_COL_MAP = {
        "In which region or district, you have spent most of your life": "region",
        "Have you had these tests?": "tests_done",
        "Did any methods helped": "helpful_methods",
        "Are you using alternative treatments": "alternative_treatments",
        "Which Social Media platforms you usually follow for PCOS related advices?": "social_media_platforms",
        "Age Category": "age_category",
        "BMI Category": "bmi_category",
        "Menstrual cycle pattern": "cycle_pattern",
        "Cycle_Pattern": "cycle_pattern",
        "Delay between two periods": "period_delay",
        "Period_Delay": "period_delay",
        "Acne severity": "acne_severity",
        "Hair Loss": "hair_loss",
        "Vitamin D level": "vitamin_d",
        "Diagnosed with anxiety or depression?": "anxiety_depression_diagnosed",
        "Are you undergoing mental health treatment?": "mental_health_treatment",
        "Do you practice stress relief activities?": "stress_relief",
        "Number of meals per day": "meals_per_day",
        "Type of food consumed most often": "food_type",
        "Sleep duration per night": "sleep_hours",
        "Sleep quality": "sleep_quality",
        "Physical activity level": "activity_level",
        "Exercise type regularly performed": "exercise_type",
        "Tried to lose weight in the last 1 year?": "tried_weight_loss",
        "If yes, which methods?": "weight_loss_method",
        "Do you smoke?": "smoking",
        "Alcohol consumption": "alcohol",
        "Have you made any lifestyle modifications in the past year?": "lifestyle_modifications",
        "Noticed improvement in symptoms?": "symptom_improvement",
        "Are you on any fertility medications?": "fertility_meds",
        "Do you have children?": "has_children",
        "If yes, did you face infertility before?": "faced_infertility",
        "Have you ever asked any AI tool": "asked_ai_tool",
        "Do you follow any Social Media nutritionist for PCOS related advise?": "follow_social_media_nutritionist",
    }

    categorical_col_map = {}
    for c in svm_cols:
        for prefix, json_key in CATEGORICAL_COL_MAP.items():
            if c.startswith(prefix):
                categorical_col_map[prefix] = json_key
                break

    # Find symptom column
    symptom_col = None
    for c in svm_cols:
        if "Symptoms in the last 6 months" in c:
            symptom_col = c.split('_')[0]
            break

    # Find family history column
    family_dummy_prefix = None
    for c in svm_cols:
        if "Family history" in c:
            family_dummy_prefix = c.split('_')[0]
            break

    # Find test timings columns
    timing_cols = [None] * 7
    test_substrings = [
        "Fasting glucose",
        "HOMA-IR index",
        "OGTT",
        "Fasting insulin",
        "FSH",
        "LH",
        "Vitamin D level"
    ]
    for c in svm_cols:
        if "When were these tests done last time?" in c:
            prefix = c.split('_')[0]
            for i, substring in enumerate(test_substrings):
                if substring in prefix:
                    timing_cols[i] = prefix
                    break

    # Build the column index
    cat_prefixes = list(CATEGORICAL_COL_MAP.keys())
    svm_col_idx = build_svm_col_index(svm_cols, cat_prefixes)

    return preprocess_for_svm(
        extracted_json,
        svm_cols,
        svm_col_idx,
        numeric_col_map,
        categorical_col_map,
        symptom_col,
        family_dummy_prefix,
        timing_cols
    )

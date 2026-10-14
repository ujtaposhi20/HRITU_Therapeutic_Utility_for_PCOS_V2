"""
Features:
  • 18 Core questions (Age, Height, Weight -> BMI, Cycle, Symptoms, Family, Lifestyle, Stress) following Rotterdam Criteria
  • No countdown numbers in chat messages (clean, warm, elder-sisterly)
  • Milestone prompt after Question 18:
      "Apu, we now have all the essential details needed to evaluate your pattern!
       Would you like to share a few more lifestyle and daily wellness habits for
       even deeper insights & better accuracy, or would you like to see your results now?"
  • Optional extended lifestyle questions (Meals, Food Type, Sleep Quality, Exercise, Stress Relief)
  • Numbered review table with line-by-line correction capability
  • Generates complete extracted_features dict for the 2-out-of-3 ML ensemble & Google Sheets
"""

import re
from typing import Dict, Any, Optional, Tuple, List

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
YES = {"yes", "y", "ha", "haa", "hae", "ji", "hmm", "thik", "thik ache", "correct", "confirm", "confirmed", "ok", "okay", "done", "all good", "proceed", "looks good", "হ্যাঁ", "হ্যা", "জি", "হুম", "ঠিক", "ঠিক আছে", "হাঁ"}
NO = {"no", "n", "na", "nah", "naa", "ভুল", "না", "নাহ"}
SKIP = {"skip", "pass", "dont know", "don't know", "not sure", "jani na", "জানি না", "জানিনা", "বাদ"}
STOP = {"stop", "cancel", "exit", "quit", "thamo", "থামো", "বন্ধ", "বাতিল", "stop & view report", "view report", "থামো ও রিপোর্ট দেখুন", "রিপোর্ট দেখুন"}

def clean_text(text: str) -> str:
    """Normalize input text: strip whitespace, lowercase, and convert Bangla digits."""
    return text.strip().lower().translate(BN_DIGITS)

def to_number(text: str) -> Optional[float]:
    """Extract first numeric value from string."""
    cleaned = clean_text(text)
    match = re.search(r"\d+(\.\d+)?", cleaned)
    return float(match.group()) if match else None

def parse_yes_no(text: str) -> Optional[bool]:
    """Parse a boolean affirmative/negative response in English or Bangla."""
    words = set(re.split(r"[\s,.!?।]+", clean_text(text)))
    if words & YES:
        return True
    if words & NO:
        return False
    return None

def parse_height(text: str) -> Optional[float]:
    """
    Parse height in cm or feet/inches (e.g. 155, 5.2, 5'2", 5 feet 2 inches).
    Returns height in centimeters (rounded).
    """
    cleaned = clean_text(text)
    # Check feet.inches notation e.g. 5.2 or 5.4 or 5'2
    feet_match = re.search(r"(\d+)\s*(?:feet|ft|'|\.)\s*(\d+)?", cleaned)
    if feet_match:
        feet = int(feet_match.group(1))
        inches = int(feet_match.group(2) or 0)
        if 3 <= feet <= 7 and 0 <= inches < 12:
            return round((feet * 12 + inches) * 2.54)
    # Plain number
    num = to_number(text)
    if num is not None:
        if 4 <= num < 7.5: # e.g. 5.2 or 5.5
            feet = int(num)
            decimal_part = round((num - feet) * 10)
            return round((feet * 12 + decimal_part) * 2.54)
        if 120 <= num <= 220:
            return round(num)
    return None


# ── Core 18 Questions ────────────────────────────────────────────────────────
CORE_QUESTIONS = [
    {
        "key": "age", "kind": "number", "min": 10, "max": 65,
        "en": "How old are you?",
        "bn": "তোমার বয়স কত?",
        "name_en": "Age", "name_bn": "বয়স", "unit": ""
    },
    {
        "key": "height", "kind": "height",
        "en": "What is your height? (in cm like 155, or feet like 5.2)",
        "bn": "তোমার উচ্চতা কত? (সেমি যেমন ১৫৫, বা ফুট যেমন ৫.২)",
        "name_en": "Height", "name_bn": "উচ্চতা", "unit": " cm"
    },
    {
        "key": "weight", "kind": "number", "min": 25, "max": 200,
        "en": "What is your weight in kg?",
        "bn": "তোমার ওজন কত কেজি?",
        "name_en": "Weight", "name_bn": "ওজন", "unit": " kg"
    },
    {
        "key": "age_first_period", "kind": "number", "min": 8, "max": 22,
        "en": "How old were you at your first period?",
        "bn": "প্রথম মাসিক কত বছর বয়সে হয়েছিল?",
        "name_en": "Age at first period", "name_bn": "প্রথম মাসিকের বয়স", "unit": " yrs"
    },
    {
        "key": "period_length_days", "kind": "number", "min": 1, "max": 15,
        "en": "How many days does your period usually last?",
        "bn": "মাসিক সাধারণত কত দিন থাকে?",
        "name_en": "Period duration", "name_bn": "মাসিকের সময়কাল", "unit": " days"
    },
    {
        "key": "cycle_pattern", "kind": "choice",
        "values": ["Regular", "Irregular", "Absent"],
        "en": "Are your periods usually:\n1) Regular\n2) Irregular\n3) Stopped / Absent",
        "bn": "তোমার মাসিক সাধারণত:\n১) নিয়মিত\n২) অনিয়মিত\n৩) বন্ধ",
        "labels_en": ["Regular", "Irregular", "Stopped/Absent"],
        "labels_bn": ["নিয়মিত", "অনিয়মিত", "বন্ধ"],
        "name_en": "Cycle regularity", "name_bn": "মাসিক চক্র", "unit": ""
    },
    {
        "key": "gap_between_periods", "kind": "choice",
        "values": ["1 month", "2 months", "3 months or more"],
        "en": "Usual gap between your periods:\n1) About 1 month\n2) About 2 months\n3) 3 months or more",
        "bn": "দুই মাসিকের মাঝে সাধারণত ব্যবধান কত:\n১) প্রায় ১ মাস\n২) প্রায় ২ মাস\n৩) ৩ মাস বা বেশি",
        "labels_en": ["~1 month", "~2 months", "3+ months"],
        "labels_bn": ["~১ মাস", "~২ মাস", "৩+ মাস"],
        "name_en": "Period delay gap", "name_bn": "মাসিকের ব্যবধান", "unit": ""
    },
    {
        "key": "acne_severity", "kind": "choice",
        "values": ["None", "Mild", "Moderate", "Severe"],
        "en": "Acne severity on face or body:\n1) None\n2) Mild\n3) Moderate\n4) Severe",
        "bn": "মুখ বা শরীরে ব্রণের মাত্রা:\n১) নেই\n২) হালকা\n৩) মাঝারি\n৪) বেশি",
        "labels_en": ["None", "Mild", "Moderate", "Severe"],
        "labels_bn": ["নেই", "হালকা", "মাঝারি", "বেশি"],
        "name_en": "Acne severity", "name_bn": "ব্রণ", "unit": ""
    },
    {
        "key": "hair_loss", "kind": "choice",
        "values": ["None", "Mild", "Moderate", "Severe"],
        "en": "Hair thinning or scalp hair loss:\n1) None\n2) Mild\n3) Moderate\n4) Severe",
        "bn": "মাথার চুল পড়া বা চুল পাতলা হয়ে যাওয়া:\n১) নেই\n২) হালকা\n৩) মাঝারি\n৪) বেশি",
        "labels_en": ["None", "Mild", "Moderate", "Severe"],
        "labels_bn": ["নেই", "হালকা", "মাঝারি", "বেশি"],
        "name_en": "Hair loss", "name_bn": "চুল পড়া", "unit": ""
    },
    {
        "key": "sym_excess_hair", "kind": "yesno",
        "en": "Do you have excess or coarse hair on your face, chin, or body? (Yes/No)",
        "bn": "মুখে, থুতনিতে বা শরীরে কি অতিরিক্ত বা শক্ত লোম আছে? (হ্যাঁ/না)",
        "name_en": "Excess facial/body hair", "name_bn": "অতিরিক্ত লোম", "unit": ""
    },
    {
        "key": "sym_weight_gain", "kind": "yesno",
        "en": "Have you noticed sudden or stubborn weight gain in the last 6 months? (Yes/No)",
        "bn": "গত ৬ মাসে কি হঠাৎ বা দ্রুত ওজন বেড়ে যাওয়ার লক্ষণ দেখেছো? (হ্যাঁ/না)",
        "name_en": "Recent weight gain", "name_bn": "সাম্প্রতিক ওজন বৃদ্ধি", "unit": ""
    },
    {
        "key": "sym_acne_oily", "kind": "yesno",
        "en": "Have you had persistent oily skin or frequent breakouts in the last 6 months? (Yes/No)",
        "bn": "গত ৬ মাসে কি অতিরিক্ত তৈলাক্ত ত্বক বা বারবার ব্রণ ওঠার সমস্যা ছিল? (হ্যাঁ/না)",
        "name_en": "Oily skin / breakouts", "name_bn": "তৈলাক্ত ত্বক/ব্রণ", "unit": ""
    },
    {
        "key": "sym_irregular", "kind": "yesno",
        "en": "Have your periods been irregular, delayed, or missed in the last 6 months? (Yes/No)",
        "bn": "গত ৬ মাসে কি মাসিক অনিয়মিত, দেরিতে বা বন্ধ ছিল? (হ্যাঁ/না)",
        "name_en": "Irregular periods in last 6 mos", "name_bn": "গত ৬ মাসে অনিয়মিত মাসিক", "unit": ""
    },
    {
        "key": "fam_pcos", "kind": "yesno",
        "en": "Does anyone in your biological family (mother, sister, aunt) have PCOS? (Yes/No)",
        "bn": "রক্তের সম্পর্কের কারো (মা, বোন, খালা, ফুফু) কি PCOS আছে? (হ্যাঁ/না)",
        "name_en": "Family history of PCOS", "name_bn": "পরিবারে PCOS", "unit": ""
    },
    {
        "key": "fam_diabetes", "kind": "yesno",
        "en": "Does anyone in your immediate family have Type-2 Diabetes? (Yes/No)",
        "bn": "পরিবারে রক্তের কারো কি টাইপ-২ ডায়াবেটিস আছে? (হ্যাঁ/না)",
        "name_en": "Family history of Diabetes", "name_bn": "পরিবারে ডায়াবেটিস", "unit": ""
    },
    {
        "key": "sleep_hours", "kind": "choice",
        "values": ["<5 hours", "5-6 hours", "7-8 hours", ">8 hours"],
        "en": "Average sleep per night:\n1) Less than 5 hours\n2) 5-6 hours\n3) 7-8 hours\n4) More than 8 hours",
        "bn": "রাতে গড়ে কত সময় ঘুমাও:\n১) ৫ ঘণ্টার কম\n২) ৫-৬ ঘণ্টা\n৩) ৭-৮ ঘণ্টা\n৪) ৮ ঘণ্টার বেশি",
        "labels_en": ["<5 hours", "5-6 hours", "7-8 hours", ">8 hours"],
        "labels_bn": ["<৫ ঘণ্টা", "৫-৬ ঘণ্টা", "৭-৮ ঘণ্টা", ">৮ ঘণ্টা"],
        "name_en": "Sleep duration", "name_bn": "ঘুমের সময়", "unit": ""
    },
    {
        "key": "activity_level", "kind": "choice",
        "values": ["Sedentary", "Light", "Moderate (2-4 days per week)", "High (5+ days per week)"],
        "en": "Daily physical activity level:\n1) Sedentary (very little)\n2) Light\n3) Moderate (2-4 days/wk)\n4) Active (5+ days/wk)",
        "bn": "শারীরিক পরিশ্রম বা অ্যাক্টিভিটি:\n১) খুব কম (বসে থাকা)\n২) হালকা\n৩) মাঝারি (সপ্তাহে ২-৪ দিন)\n৪) সক্রিয় (সপ্তাহে ৫+ দিন)",
        "labels_en": ["Sedentary", "Light", "Moderate", "Active"],
        "labels_bn": ["খুব কম", "হালকা", "মাঝারি", "সক্রিয়"],
        "name_en": "Physical activity", "name_bn": "শারীরিক পরিশ্রম", "unit": ""
    },
    {
        "key": "stress_level", "kind": "number", "min": 1, "max": 10,
        "en": "How would you rate your everyday stress from 1 (calm) to 10 (very stressed)?",
        "bn": "১ (খুব শান্ত) থেকে ১০ (অত্যধিক চাপ) — তোমার মানসিক চাপ কত?",
        "name_en": "Perceived stress (1-10)", "name_bn": "মানসিক চাপ (১-১০)", "unit": ""
    },
]

# ── Extended Lifestyle Questions (Optional Milestone Follow-up) ─────────────
EXTENDED_QUESTIONS = [
    {
        "key": "meals_per_day", "kind": "choice",
        "values": ["1", "2", "3", ">3"],
        "en": "How many main meals do you usually eat per day?\n1) 1 meal\n2) 2 meals\n3) 3 meals\n4) More than 3 meals",
        "bn": "প্রতিদিন সাধারণত কয়বার মূল খাবার খাওয়া হয়?\n১) ১ বার\n২) ২ বার\n৩) ৩ বার\n৪) ৩ বারের বেশি",
        "labels_en": ["1 meal", "2 meals", "3 meals", ">3 meals"],
        "labels_bn": ["১ বার", "২ বার", "৩ বার", ">৩ বার"],
        "name_en": "Meals per day", "name_bn": "দৈনিক খাবার সংখ্যা", "unit": ""
    },
    {
        "key": "food_type", "kind": "choice",
        "values": ["Home-cooked", "Fast food", "Balanced or mixed diet", "Highly processed foods"],
        "en": "What type of food do you eat most frequently?\n1) Home-cooked\n2) Fast food\n3) Balanced or mixed diet\n4) Processed/packaged",
        "bn": "কোন ধরণের খাবার সবচেয়ে বেশি খাওয়া হয়?\n১) বাড়ির রান্না\n২) ফাস্ট ফুড\n৩) সুষম বা মিশ্র খাবার\n৪) প্যাকেটজাত খাবার",
        "labels_en": ["Home-cooked", "Fast food", "Balanced", "Processed"],
        "labels_bn": ["বাড়ির রান্না", "ফাস্ট ফুড", "সুষম", "প্যাকেটজাত"],
        "name_en": "Dietary pattern", "name_bn": "খাবারের ধরণ", "unit": ""
    },
    {
        "key": "sleep_quality", "kind": "choice",
        "values": ["Good", "Moderate", "Poor"],
        "en": "How would you describe your overall sleep quality?\n1) Good & restful\n2) Moderate\n3) Poor / restless",
        "bn": "তোমার ঘুমের মান সামগ্রিকভাবে কেমন?\n১) ভালো ও গভীর\n২) মাঝারি\n৩) খারাপ / অস্বস্তিকর",
        "labels_en": ["Good", "Moderate", "Poor"],
        "labels_bn": ["ভালো", "মাঝারি", "খারাপ"],
        "name_en": "Sleep quality", "name_bn": "ঘুমের মান", "unit": ""
    },
    {
        "key": "exercise_type", "kind": "choice",
        "values": ["Walking", "Cardio", "Yoga", "Home workouts", "None"],
        "en": "What kind of exercise do you regularly do?\n1) Walking\n2) Cardio / Running\n3) Yoga\n4) Home workouts\n5) None",
        "bn": "নিয়মিত কী ধরণের শরীরচর্চা করো?\n১) হাঁটা\n২) কার্ডিও/দৌড়ানো\n৩) যোগব্যায়াম\n৪) ঘরে ব্যায়াম\n৫) কোনোটিই না",
        "labels_en": ["Walking", "Cardio", "Yoga", "Home workouts", "None"],
        "labels_bn": ["হাঁটা", "কার্ডিও", "যোগব্যায়াম", "ঘরে ব্যায়াম", "কোনোটিই না"],
        "name_en": "Exercise type", "name_bn": "ব্যায়ামের ধরণ", "unit": ""
    },
    {
        "key": "stress_relief", "kind": "choice",
        "values": ["Prayer", "Meditation", "Physical exercise", "Breathing exercises", "None"],
        "en": "What helps you relieve stress most?\n1) Prayer / spiritual\n2) Meditation / Yoga\n3) Exercise\n4) Deep breathing\n5) None",
        "bn": "মানসিক চাপ কমাতে কোনটি সবচেয়ে বেশি সাহায্য করে?\n১) প্রার্থনা / নামাজ\n২) মেডিটেশন / যোগ\n৩) ব্যায়াম\n৪) গভীর শ্বাস-প্রশ্বাস\n৫) কোনোটিই না",
        "labels_en": ["Prayer", "Meditation", "Exercise", "Breathing", "None"],
        "labels_bn": ["প্রার্থনা", "মেডিটেশন", "ব্যায়াম", "শ্বাস-প্রশ্বাস", "কোনোটিই না"],
        "name_en": "Stress relief method", "name_bn": "চাপ কমানোর উপায়", "unit": ""
    },
]


def parse_answer(q: Dict[str, Any], text: str) -> Optional[Any]:
    """Parse user text into a typed answer according to question kind."""
    cleaned = clean_text(text)
    if cleaned in SKIP:
        return None

    kind = q["kind"]
    if kind == "number":
        n = to_number(text)
        if n is not None and q["min"] <= n <= q["max"]:
            return round(n, 1) if not float(n).is_integer() else int(n)
        return None

    if kind == "height":
        return parse_height(text)

    if kind == "yesno":
        return parse_yes_no(text)

    if kind == "choice":
        values = q["values"]
        labels_en = q.get("labels_en", values)
        labels_bn = q.get("labels_bn", values)

        # 1. Prefix number check (e.g. "1", "2", "3", "2)", "2.", "২", "২)")
        prefix_num_match = re.match(r"^(\d+)[\s.)\-:]*", cleaned)
        if prefix_num_match:
            n_idx = int(prefix_num_match.group(1))
            if 1 <= n_idx <= len(values):
                return values[n_idx - 1]

        # 2. Match textual labels: sort candidates by length descending to prevent substring collisions
        # e.g., "irregular" (len 9) checked before "regular" (len 7)
        candidates = []
        for i, val in enumerate(values):
            en_lbl = labels_en[i] if i < len(labels_en) else str(val)
            bn_lbl = labels_bn[i] if i < len(labels_bn) else str(val)
            candidates.append((val, en_lbl.lower(), bn_lbl))
            if str(val).lower() != en_lbl.lower():
                candidates.append((val, str(val).lower(), bn_lbl))
            # Also split slashed labels e.g. 'Stopped/Absent' -> 'stopped', 'absent'
            for part in en_lbl.split("/"):
                part_s = part.strip().lower()
                if part_s:
                    candidates.append((val, part_s, bn_lbl))
            for part in bn_lbl.split("/"):
                part_s = part.strip()
                if part_s:
                    candidates.append((val, en_lbl.lower(), part_s))

        # Sort by candidate string length descending
        candidates.sort(key=lambda x: max(len(x[1]), len(x[2])), reverse=True)

        for val, en_cand, bn_cand in candidates:
            pattern = r"(?<!\w)" + re.escape(en_cand) + r"(?!\w)"
            if re.search(pattern, cleaned):
                return val
            if bn_cand and (bn_cand in text or bn_cand.lower() in text.lower()):
                return val

        # Substring fallback if not already matched
        for val, en_cand, bn_cand in candidates:
            if en_cand in cleaned:
                return val

        # General number inside text
        n = to_number(text)
        if n is not None and 1 <= int(n) <= len(values):
            return values[int(n) - 1]

        return None

    return None


def format_value_for_display(q: Dict[str, Any], val: Any, lang: str = "en") -> str:
    """Format stored value for the summary review table."""
    if val is None:
        return "—"
    if q["kind"] == "yesno":
        if lang == "bn":
            return "হ্যাঁ" if val is True else "না"
        return "Yes" if val is True else "No"
    if q["kind"] == "choice":
        if val in q["values"]:
            idx = q["values"].index(val)
            labels = q["labels_bn"] if lang == "bn" else q["labels_en"]
            return labels[idx]
        return str(val)
    unit = q.get("unit", "")
    return f"{val}{unit}"


def generate_review_table(answers: Dict[str, Any], questions_list: List[Dict[str, Any]], lang: str = "en") -> str:
    """Construct numbered summary review table."""
    lines = []
    # Calculate BMI if height and weight present
    bmi_str = ""
    h = answers.get("height")
    w = answers.get("weight")
    if h and w:
        try:
            bmi_val = round(w / ((h / 100) ** 2), 1)
            bmi_str = f" (BMI: {bmi_val})"
        except Exception:
            pass

    for i, q in enumerate(questions_list):
        name = q["name_bn"] if lang == "bn" else q["name_en"]
        val_display = format_value_for_display(q, answers.get(q["key"]), lang)
        extra = bmi_str if q["key"] == "weight" and bmi_str else ""
        lines.append(f"{i + 1}. **{name}**: {val_display}{extra}")

    table_body = "\n".join(lines)
    if lang == "bn":
        header = "🌸 **দেখে নাও তোমার দেওয়া তথ্যগুলো ঠিক আছে কিনা:**\n\n"
        footer = "\n\nসব ঠিক থাকলে লিখো **'হ্যাঁ'** (ফলাফল দেখার জন্য), অথবা কোনো তথ্য ঠিক করতে চাইলে সেই **লাইনের নম্বর** (যেমন: ২) লিখে পাঠাও।"
    else:
        header = "🌸 **Let me double-check that I got everything right:**\n\n"
        footer = "\n\nIs this correct? Reply **'yes'** to generate your analysis, or reply with the **number** of any line you'd like to fix (e.g. 3)."

    return f"{header}{table_body}{footer}"


def build_extracted_features_from_answers(answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw screening answers into the complete extracted_features dict
    expected by preprocessor.py, ML ensemble, and Google Sheets.
    """
    height = answers.get("height")
    weight = answers.get("weight")
    bmi = None
    if height and weight:
        try:
            bmi = round(float(weight) / ((float(height) / 100) ** 2), 2)
        except Exception:
            pass

    # Build symptoms list
    symptoms = []
    if answers.get("sym_excess_hair") is True:
        symptoms.append("Excess facial")
    if answers.get("sym_weight_gain") is True:
        symptoms.append("Weight gain")
    if answers.get("sym_acne_oily") is True:
        symptoms.append("Acne or oily skin")
    if answers.get("sym_irregular") is True:
        symptoms.append("Irregular periods")

    # Build family history list
    fam_hist = []
    if answers.get("fam_pcos") is True:
        fam_hist.append("PCOS")
    if answers.get("fam_diabetes") is True:
        fam_hist.append("Type-2 Diabetes")
    if not fam_hist:
        fam_hist = ["None"]

    age = answers.get("age", 24.0)
    age_first_sym = min(float(age or 24.0), 18.0)

    extracted = {
        "age": float(age) if age is not None else 24.0,
        "bmi": float(bmi) if bmi is not None else 23.5,
        "age_at_first_period": float(answers.get("age_first_period") or 12.0),
        "menstrual_cycle_length_days": float(answers.get("period_length_days") or 5.0),
        "age_first_symptoms": float(age_first_sym),
        "cycle_pattern": str(answers.get("cycle_pattern") or "Regular"),
        "period_delay": str(answers.get("gap_between_periods") or "1 month"),
        "symptoms": symptoms,
        "acne_severity": str(answers.get("acne_severity") or "None"),
        "hair_loss": str(answers.get("hair_loss") or "None"),
        "family_history": fam_hist,
        "sleep_hours": str(answers.get("sleep_hours") or "7–8 hours"),
        "activity_level": str(answers.get("activity_level") or "Light"),
        "stress_level": int(answers.get("stress_level") or 5),
        # Extended fields if answered
        "meals_per_day": str(answers.get("meals_per_day") or "3"),
        "food_type": str(answers.get("food_type") or "Home-cooked"),
        "sleep_quality": str(answers.get("sleep_quality") or "Moderate"),
        "exercise_type": str(answers.get("exercise_type") or "None"),
        "stress_relief": str(answers.get("stress_relief") or "None"),
    }
    return extracted


# ── In-Memory Screening State Manager & Serverless Reconstruction ─────────────
# Maps session_id -> dict state
_screening_sessions: Dict[str, Dict[str, Any]] = {}

def serialize_state_for_client(state: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize screening state to a JSON-safe dictionary for client echoing."""
    return {
        "stage": state.get("stage", "not_started"),
        "q_idx": state.get("q_idx", 0),
        "ext_q_idx": state.get("ext_q_idx", 0),
        "answers": dict(state.get("answers", {})),
        "fixing_idx": state.get("fixing_idx"),
        "lang": state.get("lang", "en"),
        "is_complete": bool(state.get("is_complete", False)),
    }

def deserialize_client_state(client_state: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize client screening state, re-attaching active question objects."""
    stage = client_state.get("stage", "not_started")
    has_ext = (stage in ("asking_extended", "confirm", "fixing", "done")) and (client_state.get("ext_q_idx") is not None)
    active_questions = list(CORE_QUESTIONS) + list(EXTENDED_QUESTIONS) if has_ext else list(CORE_QUESTIONS)
    return {
        "stage": stage,
        "q_idx": int(client_state.get("q_idx", 0)),
        "ext_q_idx": int(client_state.get("ext_q_idx", 0)),
        "answers": dict(client_state.get("answers", {})),
        "active_questions": active_questions,
        "fixing_idx": client_state.get("fixing_idx"),
        "lang": client_state.get("lang", "en"),
        "is_complete": bool(client_state.get("is_complete", False)),
    }

def find_question_in_assistant_message(msg: str) -> Tuple[Optional[str], Optional[int]]:
    """Determine which question was asked in an assistant message."""
    msg_lower = msg.lower()
    for i, q in enumerate(CORE_QUESTIONS):
        en_first = q["en"].split("\n")[0].strip().lower()
        bn_first = q["bn"].split("\n")[0].strip()
        if (en_first and en_first in msg_lower) or (bn_first and bn_first in msg):
            return ("core", i)
        if q["name_en"].lower() in msg_lower or q["name_bn"] in msg:
            return ("core", i)
    for i, q in enumerate(EXTENDED_QUESTIONS):
        en_first = q["en"].split("\n")[0].strip().lower()
        bn_first = q["bn"].split("\n")[0].strip()
        if (en_first and en_first in msg_lower) or (bn_first and bn_first in msg):
            return ("ext", i)
        if q["name_en"].lower() in msg_lower or q["name_bn"] in msg:
            return ("ext", i)
    if "milestone" in msg_lower or "lifestyle and daily wellness" in msg_lower or "গভীর বিশ্লেষণ" in msg:
        return ("milestone", None)
    if "| #" in msg or "summary review" in msg_lower or "সারসংক্ষেপ" in msg:
        return ("confirm", None)
    return (None, None)

def reconstruct_state_from_history(history: List[Dict[str, str]], lang: str = "en") -> Dict[str, Any]:
    """
    Deterministically reconstruct screening state from conversation history.
    Provides complete resilience against Vercel serverless container cold-starts.
    """
    state = {
        "stage": "not_started",
        "q_idx": 0,
        "ext_q_idx": 0,
        "answers": {},
        "active_questions": list(CORE_QUESTIONS),
        "fixing_idx": None,
        "lang": lang,
        "is_complete": False,
    }
    if not history:
        return state

    last_q_type = None
    last_q_idx = None

    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            q_type, q_i = find_question_in_assistant_message(content)
            if q_type is not None:
                last_q_type = q_type
                last_q_idx = q_i
                if q_type == "core":
                    state["stage"] = "asking_core"
                    state["q_idx"] = q_i
                elif q_type == "ext":
                    state["stage"] = "asking_extended"
                    state["ext_q_idx"] = q_i
                    state["active_questions"] = list(CORE_QUESTIONS) + list(EXTENDED_QUESTIONS)
                elif q_type == "milestone":
                    state["stage"] = "milestone"
                elif q_type == "confirm":
                    state["stage"] = "confirm"
        elif role == "user" and last_q_type is not None:
            cleaned = clean_text(content)
            if last_q_type == "core" and last_q_idx is not None and last_q_idx < len(CORE_QUESTIONS):
                curr_q = CORE_QUESTIONS[last_q_idx]
                if cleaned in SKIP:
                    state["answers"][curr_q["key"]] = None
                else:
                    parsed_val = parse_answer(curr_q, content)
                    if parsed_val is not None:
                        state["answers"][curr_q["key"]] = parsed_val
            elif last_q_type == "ext" and last_q_idx is not None and last_q_idx < len(EXTENDED_QUESTIONS):
                curr_ext_q = EXTENDED_QUESTIONS[last_q_idx]
                if cleaned in SKIP:
                    state["answers"][curr_ext_q["key"]] = None
                else:
                    parsed_val = parse_answer(curr_ext_q, content)
                    if parsed_val is not None:
                        state["answers"][curr_ext_q["key"]] = parsed_val

    return state

def get_or_create_screening_state(session_id: str, lang: str = "en") -> Dict[str, Any]:
    if session_id not in _screening_sessions:
        _screening_sessions[session_id] = {
            "stage": "not_started",
            "q_idx": 0,
            "ext_q_idx": 0,
            "answers": {},
            "active_questions": list(CORE_QUESTIONS),
            "fixing_idx": None,
            "lang": lang,
            "is_complete": False,
        }
    return _screening_sessions[session_id]

def reset_screening_state(session_id: str):
    if session_id in _screening_sessions:
        del _screening_sessions[session_id]


def handle_screening_turn(
    session_id: str,
    user_text: str,
    lang: str = "en",
    history: Optional[List[Dict[str, str]]] = None,
    client_state: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool, Optional[Dict[str, Any]], List[str], Dict[str, Any]]:
    """
    Process one message in the slot-filling screening wizard.
    
    Returns:
        (bot_reply, is_confirmed_complete, extracted_features_or_None, quick_reply_chips, serialized_state)
    """
    # 0. Resolve State: Priority (client_state -> in-memory cache -> reconstruct from history -> fresh)
    if client_state and isinstance(client_state, dict) and client_state.get("stage") and client_state.get("stage") != "not_started":
        state = deserialize_client_state(client_state)
        _screening_sessions[session_id] = state
    elif session_id in _screening_sessions and _screening_sessions[session_id].get("stage") != "not_started":
        state = _screening_sessions[session_id]
    elif history and len(history) > 0:
        state = reconstruct_state_from_history(history, lang)
        _screening_sessions[session_id] = state
    else:
        state = get_or_create_screening_state(session_id, lang)

    state["lang"] = lang
    cleaned = clean_text(user_text)

    # 1. Stop / Cancel check
    if cleaned in STOP:
        reset_screening_state(session_id)
        msg = (
            "কোনো সমস্যা নেই আপু, পরীক্ষা এখানেই থামালাম। অন্য কোনো বিষয় নিয়ে সাহায্য লাগলে বলো! 🌸"
            if lang == "bn"
            else "No problem at all, sister! We've paused the screening. I'm right here whenever you need me! 🌸"
        )
        return msg, False, None, [], serialize_state_for_client(state)

    # 2. Starting screening (first turn)
    if state["stage"] == "not_started":
        state["stage"] = "asking_core"
        state["q_idx"] = 0
        state["answers"] = {}
        state["active_questions"] = list(CORE_QUESTIONS)

        q0 = CORE_QUESTIONS[0]
        # Check if user already answered q0 (Age) in response to the frontend initial greeting
        if cleaned in SKIP:
            state["answers"][q0["key"]] = None
            state["q_idx"] = 1
            q1 = CORE_QUESTIONS[1]
            chips = _chips_for_question(q1, lang)
            return q1[lang], False, None, chips, serialize_state_for_client(state)

        val = parse_answer(q0, user_text)
        if val is not None:
            state["answers"][q0["key"]] = val
            state["q_idx"] = 1
            q1 = CORE_QUESTIONS[1]
            chips = _chips_for_question(q1, lang)
            return q1[lang], False, None, chips, serialize_state_for_client(state)

        # Otherwise, the user sent a general greeting or unparsed text, ask q0:
        intro = (
            "ঠিক আছে আপু! চলো ধাপে ধাপে তোমার লক্ষণের তথ্যগুলো জেনে নেই। কোনো প্রশ্নের উত্তর অজানা থাকলে 'বাদ' লিখতে পারো।\n\n"
            if lang == "bn"
            else "Alright sister! Let's walk through your health pattern step by step. If you're not sure about any answer, just type 'skip'.\n\n"
        )
        reply = intro + q0[lang]
        chips = _chips_for_question(q0, lang)
        return reply, False, None, chips, serialize_state_for_client(state)

    # 3. Answering Core Questions
    if state["stage"] == "asking_core":
        curr_q = CORE_QUESTIONS[state["q_idx"]]
        
        # Check skip
        if cleaned in SKIP:
            state["answers"][curr_q["key"]] = None
        else:
            val = parse_answer(curr_q, user_text)
            if val is None:
                # Retry prompt
                retry_text = (
                    "দুঃখিত আপু, উত্তরটি বুঝতে পারিনি। দয়া করে আবার বলো:\n\n"
                    if lang == "bn"
                    else "Sorry sister, I didn't quite catch that. Could you please share again:\n\n"
                )
                chips = _chips_for_question(curr_q, lang)
                return retry_text + curr_q[lang], False, None, chips, serialize_state_for_client(state)
            state["answers"][curr_q["key"]] = val

        # Advance question
        state["q_idx"] += 1
        if state["q_idx"] < len(CORE_QUESTIONS):
            next_q = CORE_QUESTIONS[state["q_idx"]]
            chips = _chips_for_question(next_q, lang)
            return next_q[lang], False, None, chips, serialize_state_for_client(state)
        else:
            # Reached milestone after 18 core questions!
            state["stage"] = "milestone"
            milestone_prompt = (
                "আপু, তোমার প্যাটার্ন মূল্যায়নের জন্য প্রয়োজনীয় মূল তথ্যগুলো আমরা পেয়েছি! "
                "আরও গভীর বিশ্লেষণ ও সঠিকতার জন্য তুমি কি জীবনযাত্রা ও দৈনন্দিন অভ্যাস সংক্রান্ত আরও কিছু তথ্য শেয়ার করতে চাও, "
                "নাকি এখনই ফলাফল দেখতে চাও?"
                if lang == "bn"
                else "Apu, we now have all the essential details needed to evaluate your pattern! "
                     "Would you like to share a few more lifestyle and daily wellness habits for even deeper insights & better accuracy, "
                     "or would you like to see your results now?"
            )
            chips = (
                ["আরও তথ্য দিন", "এখনই ফলাফল দেখুন"]
                if lang == "bn"
                else ["Share More Details", "See My Results Now"]
            )
            return milestone_prompt, False, None, chips, serialize_state_for_client(state)

    # 4. Milestone Decision: Share More vs See Results Now
    if state["stage"] == "milestone":
        wants_more = any(w in cleaned for w in ["share", "more", "aro", "details", "habit", "lifestyle", "হ্যাঁ", "আরও", "তথ্য"])
        wants_results = any(w in cleaned for w in ["result", "now", "see", "show", "ফলাফল", "এখনই", "দেখাও", "no", "না"])

        if wants_more and not wants_results:
            state["stage"] = "asking_extended"
            state["ext_q_idx"] = 0
            state["active_questions"] = list(CORE_QUESTIONS) + list(EXTENDED_QUESTIONS)
            q_ext = EXTENDED_QUESTIONS[0]
            bridge = (
                "দারুণ! চলো দৈনন্দিন জীবনযাত্রা নিয়ে কয়েকটি ছোট প্রশ্ন জেনে নেই:\n\n"
                if lang == "bn"
                else "Wonderful! Let's explore a few quick daily lifestyle questions:\n\n"
            )
            chips = _chips_for_question(q_ext, lang)
            return bridge + q_ext[lang], False, None, chips, serialize_state_for_client(state)
        else:
            # Proceed directly to review table
            state["stage"] = "confirm"
            table = generate_review_table(state["answers"], state["active_questions"], lang)
            chips = ["হ্যাঁ, ঠিক আছে", "১", "২", "৩"] if lang == "bn" else ["Yes, correct", "1", "2", "3"]
            return table, False, None, chips, serialize_state_for_client(state)

    # 5. Answering Extended Lifestyle Questions
    if state["stage"] == "asking_extended":
        ext_idx = state.get("ext_q_idx", 0)
        curr_ext_q = EXTENDED_QUESTIONS[ext_idx]

        if cleaned in SKIP:
            state["answers"][curr_ext_q["key"]] = None
        else:
            val = parse_answer(curr_ext_q, user_text)
            if val is None:
                retry_text = (
                    "দুঃখিত আপু, উত্তরটি বুঝতে পারিনি। দয়া করে আবার বলো:\n\n"
                    if lang == "bn"
                    else "Sorry sister, I didn't quite catch that. Could you please share again:\n\n"
                )
                chips = _chips_for_question(curr_ext_q, lang)
                return retry_text + curr_ext_q[lang], False, None, chips, serialize_state_for_client(state)
            state["answers"][curr_ext_q["key"]] = val

        ext_idx += 1
        state["ext_q_idx"] = ext_idx
        if ext_idx < len(EXTENDED_QUESTIONS):
            next_ext_q = EXTENDED_QUESTIONS[ext_idx]
            chips = _chips_for_question(next_ext_q, lang)
            return next_ext_q[lang], False, None, chips, serialize_state_for_client(state)
        else:
            # All extended done -> Go to review table
            state["stage"] = "confirm"
            table = generate_review_table(state["answers"], state["active_questions"], lang)
            chips = ["হ্যাঁ, ঠিক আছে", "১", "২", "৩"] if lang == "bn" else ["Yes, correct", "1", "2", "3"]
            return table, False, None, chips, serialize_state_for_client(state)

    # 6. Fixing a specific line
    if state["stage"] == "fixing":
        fix_i = state.get("fixing_idx", 0)
        q_to_fix = state["active_questions"][fix_i]
        val = parse_answer(q_to_fix, user_text)
        if val is None and cleaned not in SKIP:
            retry_text = (
                "সঠিক মানটি লিখে পাঠাও:\n\n"
                if lang == "bn"
                else "Please provide the corrected value:\n\n"
            )
            chips = _chips_for_question(q_to_fix, lang)
            return retry_text + q_to_fix[lang], False, None, chips, serialize_state_for_client(state)

        state["answers"][q_to_fix["key"]] = val
        state["stage"] = "confirm"
        ack = (
            "ধন্যবাদ আপু, তথ্যটি আপডেট করা হয়েছে! এবার পুরো তালিকাটি দেখো:\n\n"
            if lang == "bn"
            else "Updated! Here is your complete updated summary table:\n\n"
        )
        table = generate_review_table(state["answers"], state["active_questions"], lang)
        chips = ["হ্যাঁ, ঠিক আছে", "১", "২", "৩"] if lang == "bn" else ["Yes, correct", "1", "2", "3"]
        return ack + table, False, None, chips, serialize_state_for_client(state)

    # 7. Confirmation Review Table ("yes" or line number to fix)
    if state["stage"] == "confirm":
        # Check for inline one-shot fix: e.g. "change 1 to 26" or "1: 26" or "১: ২৬"
        inline_match = re.match(
            r'^(?:change\s+|লাইন\s+)?([0-9০-৯]+)\s*(?:to|হবে|:|—|-|=)\s*(.+)$',
            user_text.strip(),
            re.IGNORECASE
        )
        if inline_match:
            line_str, val_str = inline_match.groups()
            line_num = to_number(line_str)
            if line_num is not None and 1 <= int(line_num) <= len(state["active_questions"]):
                target_idx = int(line_num) - 1
                q_to_fix = state["active_questions"][target_idx]
                parsed_val = parse_answer(q_to_fix, val_str)
                if parsed_val is not None:
                    state["answers"][q_to_fix["key"]] = parsed_val
                    ack = (
                        "ধন্যবাদ আপু, তথ্যটি আপডেট করা হয়েছে! এবার পুরো তালিকাটি দেখো:\n\n"
                        if lang == "bn"
                        else "Updated! Here is your complete updated summary table:\n\n"
                    )
                    table = generate_review_table(state["answers"], state["active_questions"], lang)
                    chips = ["হ্যাঁ, ঠিক আছে", "১", "২", "৩"] if lang == "bn" else ["Yes, correct", "1", "2", "3"]
                    return ack + table, False, None, chips, serialize_state_for_client(state)

        # Check if user typed a line number to fix
        n = to_number(user_text)
        if n is not None and 1 <= int(n) <= len(state["active_questions"]):
            target_idx = int(n) - 1
            state["stage"] = "fixing"
            state["fixing_idx"] = target_idx
            q_fix = state["active_questions"][target_idx]
            prompt = (
                f"লাইন {int(n)} সংশোধন: {q_fix['name_bn']}\n\n{q_fix['bn']}"
                if lang == "bn"
                else f"Fixing line {int(n)}: {q_fix['name_en']}\n\n{q_fix['en']}"
            )
            chips = _chips_for_question(q_fix, lang)
            return prompt, False, None, chips, serialize_state_for_client(state)

        # Check if confirmed ("yes", "ha", "correct")
        is_yes = parse_yes_no(user_text)
        if is_yes is True or cleaned in YES or "ঠিক" in cleaned or "correct" in cleaned:
            # Finalized! Build extracted features
            extracted = build_extracted_features_from_answers(state["answers"])
            state["stage"] = "done"
            state["is_complete"] = True
            closing = (
                "ধন্যবাদ আপু! তোমার দেওয়া সব তথ্য পর্যালোচনা করে আমাদের এআই মডেলগুলো বিশ্লেষণ প্রস্তুত করছে... 🌸"
                if lang == "bn"
                else "Thank you so much, sister! All your details have been recorded. Evaluating your patterns with our AI ensemble now... 🌸"
            )
            return closing, True, extracted, [], serialize_state_for_client(state)

        # If unclear
        hint = (
            "সব তথ্য ঠিক থাকলে **'হ্যাঁ'** লিখো, অথবা যে লাইনটি বদলাতে চাও তার **নম্বর** (যেমন ২) লিখো।"
            if lang == "bn"
            else "If everything looks good, please reply **'yes'**, or reply with the **line number** you want to correct (e.g. 2)."
        )
        chips = ["হ্যাঁ, ঠিক আছে"] if lang == "bn" else ["Yes, correct"]
        return hint, False, None, chips, serialize_state_for_client(state)

    # Fallback
    return "Let me know how you'd like to proceed, sister! 🌸", False, None, [], serialize_state_for_client(state)


def _chips_for_question(q: Dict[str, Any], lang: str) -> List[str]:
    """Generate quick-reply chips for a question."""
    skip_lbl = "বাদ" if lang == "bn" else "Skip"
    stop_lbl = "থামো ও রিপোর্ট দেখুন" if lang == "bn" else "Stop & View Report"
    if q["kind"] == "yesno":
        yes_lbl = "হ্যাঁ" if lang == "bn" else "Yes"
        no_lbl = "না" if lang == "bn" else "No"
        return [yes_lbl, no_lbl, skip_lbl, stop_lbl]
    if q["kind"] == "choice":
        labels = q["labels_bn"] if lang == "bn" else q["labels_en"]
        chips = [f"{i+1}) {lbl}" for i, lbl in enumerate(labels)]
        chips.append(skip_lbl)
        chips.append(stop_lbl)
        return chips
    return [skip_lbl, stop_lbl]

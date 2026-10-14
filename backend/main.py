from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import os
import sys
import csv
import re
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

ROOT_DIR = os.path.dirname(BASE_DIR)
STATIC_DIR = os.path.join(ROOT_DIR, "frontend/static") if os.path.isdir(os.path.join(ROOT_DIR, "frontend/static")) else os.path.join(BASE_DIR, "frontend/static")
HTML_PATH = os.path.join(ROOT_DIR, "frontend/index.html") if os.path.isfile(os.path.join(ROOT_DIR, "frontend/index.html")) else os.path.join(BASE_DIR, "frontend/index.html")

from agent import PCOSAssistant
from preprocessor import preprocess_for_model, build_collected_data_row, preprocess_for_svm_default
from explainability import generate_explainability_report
import google_sheets
import safety
import privacy
import screening  # Module 1: PCOS Check (Slot-filling screening & ML analysis)
import sister     # Module 2: Talk Support (Empathy, health & fertility)
import lifestyle  # Module 3: Lifestyle (Nutrition, sleep, workout & stress care)
import learn      # Module 4: Learn Period Cycle (Curriculum & quiz)

app = FastAPI(title="HRITU AI Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    return FileResponse(HTML_PATH)

ML_DIR = os.path.join(BASE_DIR, "ml models")

rf_model = None
dt_model = None
svm_model = None
rf_training_columns = None
dt_training_columns = None
svm_training_columns = None
rf_label_encoder = None
dt_label_encoder = None
svm_scaler = None

try:
    rf_model = joblib.load(os.path.join(ML_DIR, "rf_trained_model.pkl"))
    rf_training_columns = list(joblib.load(os.path.join(ML_DIR, "rf_training_columns.pkl")))
    rf_label_encoder = joblib.load(os.path.join(ML_DIR, "rf_label_encoder.pkl"))
    print(f"✅ RF model loaded — {len(rf_training_columns)} features")
except Exception as e:
    print(f"⚠️  RF model load error: {e}")

try:
    dt_model = joblib.load(os.path.join(ML_DIR, "dt_pcos_model.pkl"))
    dt_training_columns = list(joblib.load(os.path.join(ML_DIR, "dt_training_columns.pkl")))
    dt_label_encoder = joblib.load(os.path.join(ML_DIR, "dt_label_encoder.pkl"))
    print(f"✅ DT model loaded — {len(dt_training_columns)} features")
except Exception as e:
    print(f"⚠️  DT model load error: {e}")

try:
    svm_model = joblib.load(os.path.join(ML_DIR, "svm_pcos_model.pkl"))
    svm_training_columns = list(joblib.load(os.path.join(ML_DIR, "svm_training_columns.pkl")))
    svm_scaler = joblib.load(os.path.join(ML_DIR, "svm_scaler.pkl"))
    print(f"✅ SVM model loaded — {len(svm_training_columns)} features")
except Exception as e:
    print(f"⚠️  SVM model load error: {e}")

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GROQ_API_KEY", "")
assistant = PCOSAssistant(api_key=api_key) if api_key else None

session_features: Dict[str, Dict] = {}
session_learn_state: Dict[str, Dict] = {}

EXTRACTION_INTERVAL = 3


async def _async_background_extract(session_id: str, history: List[Dict[str, str]], message_count: int):
    """Run feature extraction in the background asynchronously using to_thread."""
    if not assistant:
        return
    try:
        features = await asyncio.to_thread(assistant.extract_full_features, history)
        session_features[session_id] = {
            "features": features,
            "updated_at": datetime.now(),
            "message_count": message_count
        }
        print(f"🔄 Async background extraction cached for session {session_id} (count: {message_count})")
    except Exception as e:
        print(f"⚠️ Async background extraction error: {e}")

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    session_id: Optional[str] = None
    message_count: Optional[int] = 0
    current_mode: Optional[str] = "prediction"
    mode: Optional[str] = None
    language: Optional[str] = "en"
    learn_step: Optional[str] = None
    screening_state: Optional[Dict[str, Any]] = None
    learn_state: Optional[Dict[str, Any]] = None

class EndChatRequest(BaseModel):
    history: List[Dict[str, str]]
    session_id: Optional[str] = None
    screening_state: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int  # 1-5
    comment: Optional[str] = ""


def _is_informative_message(message: str) -> bool:
    """Detect if the message is short, non-informative feedback/compliments/chitchat."""
    cleaned = message.strip().lower()
    cleaned_stripped = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()?]', '', cleaned)
    
    if len(cleaned_stripped) < 4:
        return False
        
    non_info_words = {
        "wow", "impressive", "awesome", "perfect", "ok", "okay", "acha", "achha",
        "thik ache", "thik ase", "thank you", "thanks", "tq", "ty", "great", "nice",
        "good", "hello", "hi", "hey", "asalamu alaikum", "assalamu alaikum", "slm",
        "আপু", "হাই", "হ্যালো", "আসসালামু আলাইকুম", "ঠিক আছে", "ধন্যবাদ", "ok apu",
        "okay apu", "acha apu", "yes", "no", "haa", "nah", "ha", "na", "হ্যাঁ", "না", "wow apu"
    }
    if cleaned_stripped in non_info_words:
        return False
        
    has_letters = bool(re.search(r'[a-zA-Z0-9\u0980-\u09FF]', cleaned))
    if not has_letters:
        return False
        
    return True

@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    sid = request.session_id or str(uuid.uuid4())
    req_mode = (request.mode or request.current_mode or "prediction").lower()
    if req_mode in ("pcos_assessment", "assessment", "predict"):
        mode = "prediction"
    elif req_mode in ("talk", "pregnancy"):
        mode = "talk"
    else:
        mode = req_mode
    lang = request.language or "en"
    hist = request.history or []

    # 1. PII Redaction: In-flight anonymization before any processing
    raw_user_msg = request.message
    redacted_msg = privacy.redact(raw_user_msg)
    request.message = redacted_msg

    # 2. Suicide / Crisis Interceptor: Zero-LLM Fast Path
    if safety.is_crisis(raw_user_msg) or safety.is_crisis(redacted_msg):
        crisis_info = safety.get_crisis_reply(raw_user_msg)
        return {
            "response": crisis_info["response"],
            "is_crisis": True,
            "chips": []
        }

    # 3. Off-Topic / Scope Guardrail: Zero-LLM Fast Path
    if safety.is_off_topic(raw_user_msg) or safety.is_off_topic(redacted_msg):
        off_topic_reply = safety.get_off_topic_reply(raw_user_msg)
        chips = ["PCOS Symptoms", "Period Cycle", "Healthy Diet"] if lang == "en" else ["PCOS লক্ষণ", "মাসিক চক্র", "স্বাস্থ্যকর খাবার"]
        return {
            "response": off_topic_reply,
            "is_off_topic": True,
            "chips": chips
        }

    # 4. Module 4: Learn Your Period Cycle (interactive curriculum + quiz + RAG)
    if mode == "learn":
        if not assistant:
            return {"response": "Hi sis! 💕 Please configure the API key to start learning!", "chips": []}
        l_state = request.learn_state or session_learn_state.setdefault(sid, {})
        if request.learn_step:
            if request.learn_step.startswith("phase_"):
                try:
                    idx = int(request.learn_step.split("_")[1]) - 1
                    l_state["learn_step"] = max(0, min(3, idx))
                except Exception:
                    pass
            elif request.learn_step == "quiz":
                l_state["learn_step"] = "quiz"
                lang_code = "bn" if lang == "bn" else "en"
                chips = ["১", "২", "৩"] if lang == "bn" else ["1", "2", "3"]
                return {"response": assistant.QUIZ[lang_code], "chips": chips, "learn_state": l_state}
            elif request.learn_step == "open":
                l_state["learn_step"] = "open"
        try:
            response, updated_state = await asyncio.to_thread(
                assistant.chat_learn, hist, request.message, l_state, lang
            )
            session_learn_state[sid] = updated_state
            step = updated_state.get("learn_step")
            chips = []
            if step == "quiz":
                chips = ["১", "২", "৩"] if lang == "bn" else ["1", "2", "3"]
            elif isinstance(step, int):
                chips = ["পরেরটা"] if lang == "bn" else ["Next"]
            else:
                chips = ["পরেরটা"] if lang == "bn" else ["Next"]
            return {"response": response, "chips": chips, "learn_state": updated_state}
        except Exception as e:
            print(f"Module 4 error: {e}")
            fallback_learn = (
                "মাসিক চক্রের যেকোনো ধাপ নিয়ে আমাকে জিজ্ঞেস করতে পারো, আপু! 🌸"
                if lang == "bn"
                else "Feel free to ask me anything about period cycle phases and self-care, sister! 🌸"
            )
            return {"response": fallback_learn, "chips": ["Next" if lang == "en" else "পরেরটা"], "learn_state": l_state}

    # 4. Module 1: PCOS Prediction Assessment (Slot-Filling Wizard)
    if mode == "prediction":
        bot_reply, is_confirmed, extracted_features, chips, current_screening_state = screening.handle_screening_turn(
            session_id=sid,
            user_text=request.message,
            lang=lang,
            history=hist,
            client_state=request.screening_state,
        )

        if not is_confirmed:
            return {
                "response": bot_reply,
                "chips": chips,
                "screening_state": current_screening_state
            }

        # User confirmed the review table! Proceed to ML Ensemble & Explainability
        session_features[sid] = {
            "features": extracted_features,
            "updated_at": datetime.now(),
            "message_count": len(hist)
        }

        rf_label = "Insufficient Data"
        dt_label = "Insufficient Data"
        svm_label = "Insufficient Data"
        final_verdict = "Not Suspected / Minimal chance"
        explainability_data = None

        if rf_model and rf_training_columns:
            try:
                rf_df = preprocess_for_model(extracted_features, rf_training_columns)
                rf_pred = rf_model.predict(rf_df)[0]
                rf_label = rf_label_encoder.inverse_transform([rf_pred])[0] if rf_label_encoder else str(rf_pred)
            except Exception as e:
                print(f"RF prediction error: {e}")

        if dt_model and dt_training_columns:
            try:
                dt_df = preprocess_for_model(extracted_features, dt_training_columns)
                dt_pred = dt_model.predict(dt_df)[0]
                dt_label = dt_label_encoder.inverse_transform([dt_pred])[0] if dt_label_encoder else str(dt_pred)
            except Exception as e:
                print(f"DT prediction error: {e}")

        if svm_model and svm_training_columns and svm_scaler:
            try:
                svm_df = preprocess_for_svm_default(extracted_features, svm_training_columns)
                svm_scaled = svm_scaler.transform(svm_df.values.astype(float))
                svm_pred = svm_model.predict(svm_scaled)[0]
                svm_mapping = {0: "No", 1: "Suspected", 2: "Yes"}
                svm_label = svm_mapping.get(svm_pred, "No")
            except Exception as e:
                print(f"SVM prediction error: {e}")

        valid_labels = [l for l in [rf_label, dt_label, svm_label] if l not in ("Error", "Insufficient Data")]
        yes_suspected_count = sum(1 for l in valid_labels if l in ("Yes", "Suspected"))
        if yes_suspected_count >= 2:
            final_verdict = "Suspected / Chance of condition"
        else:
            final_verdict = "Not Suspected / Minimal chance"

        # SHAP & LIME Explainability
        try:
            rf_df_sample = preprocess_for_model(extracted_features, rf_training_columns) if rf_model and rf_training_columns else None
            dt_df_sample = preprocess_for_model(extracted_features, dt_training_columns) if dt_model and dt_training_columns else None
            svm_df_sample = preprocess_for_svm_default(extracted_features, svm_training_columns) if svm_model and svm_training_columns else None
            if rf_df_sample is not None:
                explainability_data = generate_explainability_report(
                    rf_model, dt_model, svm_model, rf_df_sample, dt_df_sample, svm_df_sample
                )
        except Exception as ex:
            print(f"Explainability generation error: {ex}")

        # Empathetic non-diagnostic result statement
        pattern_statement = None
        if assistant:
            try:
                pattern_statement = assistant.generate_result_statement(final_verdict, extracted_features, hist)
            except Exception as e:
                print(f"Statement generation error: {e}")
                pattern_statement = assistant._fallback_result_message(rf_label, dt_label)
        else:
            pattern_statement = (
                "Our observation indicates patterns consistent with PCOS. Please visit a gynecologist or endocrinologist for clinical confirmation. 🌸"
                if "Suspected" in final_verdict
                else "Our observation indicates minimal signs commonly linked with PCOS. Maintain your healthy habits and consult a doctor if symptoms arise! 🌸"
            )

        # Background save to Google Sheets (and local CSV fallback)
        background_tasks.add_task(
            _background_save_and_cleanup, extracted_features, rf_label, dt_label, svm_label, final_verdict, sid
        )

        return {
            "response": f"{bot_reply}\n\n---\n\n{pattern_statement}",
            "is_screening_complete": True,
            "pattern": pattern_statement,
            "rf_prediction": rf_label,
            "dt_prediction": dt_label,
            "svm_prediction": svm_label,
            "final_verdict": final_verdict,
            "explainability": explainability_data,
            "screening_state": current_screening_state,
            "chips": []
        }

    # 5. Module 2: Lifestyle, Diet & Routine Advice (Two-Tier Hybrid RAG)
    if mode == "lifestyle":
        if not assistant:
            return {"response": "Hi sis! 💕 Please configure the API key to get lifestyle advice!", "chips": []}
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(assistant.chat_lifestyle, hist, request.message, lang),
                timeout=18.0
            )
            return {"response": response, "chips": []}
        except asyncio.TimeoutError:
            fallback_msg = (
                "আপু, আমি তোমার পাশেই আছি। ডায়েট বা লাইফস্টাইল নিয়ে তোমার আজকের প্রশ্নটি আবার বলো তো? 🌸"
                if lang == "bn"
                else "I'm right here with you, sister! Could you please share your question about lifestyle or diet again? 🌸"
            )
            return {"response": fallback_msg, "chips": []}

    # 6. Module 3: Talk (PCOS, Women's Health, Medical Concerns, Pregnancy & Fertility)
    if mode in ("pregnancy", "talk"):
        if not assistant:
            return {"response": "Hi sis! 💕 Please configure the API key to chat!", "chips": []}
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(assistant.chat_talk, hist, request.message, lang),
                timeout=18.0
            )
            return {"response": response, "chips": []}
        except asyncio.TimeoutError:
            fallback_msg = (
                "আপু, আমি তোমার পাশেই আছি। তোমার প্রশ্ন বা মনের কথাটি আবার বলো তো? 🌸"
                if lang == "bn"
                else "I'm right here with you, sister. Please tell me what's on your mind! 🌸"
            )
            return {"response": fallback_msg, "chips": []}

    return {"response": "Hello! How can I help you today? 🌸", "chips": []}


# ── Mandatory Features for Prediction (Must ALL be present) ─────────────────
MANDATORY_PREDICTION_FEATURES = [
    "age",
    "bmi",
    "age_at_first_period",
    "menstrual_cycle_length_days",
    "age_first_symptoms",
    "cycle_pattern",
    "period_delay",
    "symptoms",
    "acne_severity",
    "hair_loss"
]


# ── Check Completeness ───────────────────────────────────────────────────────
@app.post("/check-completeness")
def check_completeness(request: EndChatRequest):
    session_id = request.session_id or ""
    cached = session_features.get(session_id)
    user_msg_count = sum(1 for m in request.history if m.get("role") == "user")
    
    if request.screening_state and request.screening_state.get("answers"):
        extracted = screening.build_extracted_features_from_answers(request.screening_state["answers"])
    elif cached and cached.get("features") and cached.get("message_count") == user_msg_count:
        extracted = cached["features"]
    elif assistant:
        try:
            print(f"Check completeness: Re-extracting features (cache count: {cached.get('message_count') if cached else None}, current user msg_count: {user_msg_count})")
            extracted = assistant.extract_full_features(request.history)
            session_features[session_id] = {
                "features": extracted,
                "updated_at": datetime.now(),
                "message_count": user_msg_count
            }
        except Exception as e:
            print(f"Check completeness extraction error: {e}")
            extracted = {}
    else:
        extracted = {}

    missing_mandatory = []
    for k in MANDATORY_PREDICTION_FEATURES:
        v = extracted.get(k)
        if v is None or v == "None" or v == [] or (isinstance(v, str) and v.strip() == ""):
            missing_mandatory.append(k)
                
    missing_percentage = (len(missing_mandatory) / len(MANDATORY_PREDICTION_FEATURES)) * 100.0 if missing_mandatory else 0.0
    return {"missing_percentage": missing_percentage, "missing_mandatory": missing_mandatory}


# ── End Chat → Extract → Predict → Report ─────────────────────────────────────────
@app.post("/end-chat")
def end_chat(request: EndChatRequest, background_tasks: BackgroundTasks):
    session_id = request.session_id or str(uuid.uuid4())

    cached = session_features.get(session_id)
    user_msg_count = sum(1 for m in request.history if m.get("role") == "user")

    if request.screening_state and request.screening_state.get("answers"):
        extracted = screening.build_extracted_features_from_answers(request.screening_state["answers"])
    elif cached and cached.get("features") and cached.get("message_count") == user_msg_count:
        extracted = cached["features"]
        print(f"Using cached features for session {session_id} (extracted at {cached['updated_at']})")
    elif assistant:
        try:
            print(f"End chat: Re-extracting features (cache count: {cached.get('message_count') if cached else None}, current user msg_count: {user_msg_count})")
            extracted = assistant.extract_full_features(request.history)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            extracted = assistant._get_default_features()
    else:
        extracted = {
            "age": 24.0, "bmi": 23.5, "cycle_pattern": "Regular",
            "symptoms": [], "acne_severity": "None", "hair_loss": "None",
        }

    # Strict check: ALL 10 mandatory features must be extracted for prediction to be possible
    missing_mandatory = []
    for k in MANDATORY_PREDICTION_FEATURES:
        v = extracted.get(k)
        if v is None or v == "None" or v == [] or (isinstance(v, str) and v.strip() == ""):
            missing_mandatory.append(k)

    is_short_conversation = len(missing_mandatory) > 0
    if is_short_conversation:
        print(f"Skipping ML prediction — missing mandatory core feature(s): {missing_mandatory}")

    # Fill in fallback values for any remaining secondary data so the CSV is fully populated
    fallback_values = assistant._get_fallback_values() if assistant else {}
    for k, v in fallback_values.items():
        if extracted.get(k) is None:
            extracted[k] = v

    # Step 3: Conditional ML execution — skip if mandatory features are missing
    rf_label = "Insufficient Data"
    dt_label = "Insufficient Data"
    svm_label = "Insufficient Data"
    final_verdict = "Insufficient Data"
    explainability_data = None

    if not is_short_conversation:
        # Data is complete — feed cached JSON into ML models
        if rf_model and rf_training_columns:
            try:
                rf_df = preprocess_for_model(extracted, rf_training_columns)
                rf_pred = rf_model.predict(rf_df)[0]
                if rf_label_encoder:
                    rf_label = rf_label_encoder.inverse_transform([rf_pred])[0]
                else:
                    rf_label = str(rf_pred)
                print(f"RF prediction: {rf_label}")
            except Exception as e:
                print(f"RF prediction error: {e}")
                import traceback
                traceback.print_exc()

        if dt_model and dt_training_columns:
            try:
                dt_df = preprocess_for_model(extracted, dt_training_columns)
                dt_pred = dt_model.predict(dt_df)[0]
                if dt_label_encoder:
                    dt_label = dt_label_encoder.inverse_transform([dt_pred])[0]
                else:
                    dt_label = str(dt_pred)
                print(f"DT prediction: {dt_label}")
            except Exception as e:
                print(f"DT prediction error: {e}")
                import traceback
                traceback.print_exc()

        if svm_model and svm_training_columns and svm_scaler:
            try:
                svm_df = preprocess_for_svm_default(extracted, svm_training_columns)
                svm_scaled = svm_scaler.transform(svm_df.values.astype(float))
                svm_pred = svm_model.predict(svm_scaled)[0]
                svm_mapping = {0: "No", 1: "Suspected", 2: "Yes"}
                svm_label = svm_mapping.get(svm_pred, "No")
                print(f"SVM prediction: {svm_label}")
            except Exception as e:
                print(f"SVM prediction error: {e}")
                import traceback
                traceback.print_exc()

        # Majority Voting: 2-out-of-3
        valid_labels = [l for l in [rf_label, dt_label, svm_label] if l not in ("Error", "Insufficient Data")]
        yes_suspected_count = sum(1 for l in valid_labels if l in ("Yes", "Suspected"))
        no_count = sum(1 for l in valid_labels if l == "No")

        if yes_suspected_count >= 2:
            final_verdict = "Suspected / Chance of condition"
        elif no_count >= 2:
            final_verdict = "Not Suspected / Minimal chance"
        else:
            if "Yes" in valid_labels or "Suspected" in valid_labels:
                final_verdict = "Suspected / Chance of condition"
            else:
                final_verdict = "Not Suspected / Minimal chance"
        print(f"Majority Voting Verdict: {final_verdict}")

        # Step 3.5: Execute SHAP & LIME explainability strictly AFTER voting completes
        rf_df_sample = preprocess_for_model(extracted, rf_training_columns) if rf_model and rf_training_columns else None
        dt_df_sample = preprocess_for_model(extracted, dt_training_columns) if dt_model and dt_training_columns else None
        svm_df_sample = preprocess_for_svm_default(extracted, svm_training_columns) if svm_model and svm_training_columns else None
        
        if rf_df_sample is not None:
            explainability_data = generate_explainability_report(
                rf_model, dt_model, svm_model, rf_df_sample, dt_df_sample, svm_df_sample
            )
            print(f"SHAP & LIME Explainability generated: {len(explainability_data.get('shap_factors', []))} SHAP factors, {len(explainability_data.get('lime_factors', []))} LIME factors.")
    else:
        print(f"Skipping ML prediction — mandatory features missing: {missing_mandatory}")

    # Step 4: Generate result statement or return insufficient data flag
    if is_short_conversation:
        print(f"Prediction skipped for session {session_id} due to missing mandatory core feature(s): {missing_mandatory}")
        return {
            "insufficient_data": True,
            "missing_mandatory": missing_mandatory,
            "pattern": None,
            "suggestion": None,
            "rf_prediction": "Insufficient Data",
            "dt_prediction": "Insufficient Data",
            "svm_prediction": "Insufficient Data",
            "final_verdict": "Insufficient Data"
        }
    elif assistant:
        try:
            pattern = assistant.generate_result_statement(final_verdict, extracted, request.history)
        except Exception as e:
            print(f"Statement generation error: {e}")
            pattern = assistant._fallback_result_message(rf_label, dt_label)
        suggestion = (
            "Remember: This is a pattern-matching observation, NOT a medical diagnosis. "
            "For proper evaluation, please visit a gynecologist or endocrinologist. "
            "They're the real experts and will take the best care of you! ✨"
        )
    else:
        pattern = (
            "I wasn't able to fully analyze our chat right now (my AI brain is sleeping 😴), "
            "but please do consult a gynecologist for proper evaluation! 💕"
        )
        suggestion = (
            "Remember: This is a pattern-matching observation, NOT a medical diagnosis. "
            "For proper evaluation, please visit a gynecologist or endocrinologist. "
            "They're the real experts and will take the best care of you! ✨"
        )

    # Step 5: Save to Google Sheets (or CSV fallback) — ASYNC via background task
    background_tasks.add_task(_background_save_and_cleanup, extracted, rf_label, dt_label, svm_label, final_verdict, session_id)

    # Return to UI with explainability findings
    return {
        "session_id": session_id,
        "pattern": pattern,
        "suggestion": suggestion,
        "explainability": explainability_data,
    }


def _background_save_and_cleanup(extracted: dict, rf_label: str, dt_label: str, svm_label: str, final_verdict: str, session_id: str):
    """Background task: save data to Google Sheets/CSV, then clean up session cache."""
    # Save data — CSV fallback ensures data is NEVER lost
    try:
        _save_collected_data(extracted, rf_label, dt_label, svm_label, final_verdict, session_id)
    except Exception as e:
        print(f"Data save error: {e}")
        # Emergency fallback: write directly to CSV to prevent data loss
        try:
            row = build_collected_data_row(extracted, rf_label, dt_label, svm_label, final_verdict)
            row["session_id"] = session_id
            row["timestamp"] = datetime.now().isoformat()
            csv_path = os.path.join(BASE_DIR, "collected_pcos_data.csv")
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            print(f"📊 Emergency CSV save successful for session {session_id}")
        except Exception as csv_err:
            print(f"❌ Critical: Could not save data anywhere for session {session_id}: {csv_err}")

    # Clean up session cache after data is safely persisted
    session_features.pop(session_id, None)
    screening.reset_screening_state(session_id)
    session_learn_state.pop(session_id, None)
    print(f"🧹 Session cache cleaned for {session_id}")


# ── Feedback Endpoint ────────────────────────────────────────────────────────
@app.post("/feedback")
def feedback(request: FeedbackRequest):
    try:
        row = {
            "timestamp": datetime.now().isoformat(),
            "session_id": request.session_id,
            "rating": request.rating,
            "comment": request.comment or "",
        }
        google_sheets.append_feedback(row)
        return {"status": "ok", "message": "Thank you for your feedback! 💕"}
    except Exception as e:
        print(f"Feedback save error: {e}")
        return {"status": "error", "message": "Could not save feedback."}


# ── Data Collection Helper ───────────────────────────────────────────────────
def _save_collected_data(extracted: dict, rf_label: str, dt_label: str, svm_label: str, final_verdict: str, session_id: str):
    """Append a row to Google Sheets (or CSV fallback) for future retraining."""
    row = build_collected_data_row(extracted, rf_label, dt_label, svm_label, final_verdict)
    row["session_id"] = session_id
    row["timestamp"] = datetime.now().isoformat()

    google_sheets.append_collected_data(row)


# ══════════════════════════════════════════════════════════════════════════════
# [WhatsApp Integration — COMMENTED OUT due to Meta Cloud API payment issue]
# Uncomment the block below (and the imports at the top) to restore WhatsApp.
# ══════════════════════════════════════════════════════════════════════════════
#
# # ── Meta WhatsApp Cloud API Webhook Integration ──────────────────────────────
# whatsapp_sessions: Dict[str, List[Dict[str, str]]] = {}
#
# async def process_hritu_response(sender_phone: str, message_text: str):
#     """
#     Asynchronous task to process incoming WhatsApp messages via HRITU AI Engine.
#     Maintains multi-turn conversation memory per user phone number, auto-detects language,
#     and dispatches responses back via Meta Graph API v22.0.
#     """
#     phone_number_id = (os.getenv("WHATSAPP_PHONE_NUMBER_ID") or "").strip().strip('"').strip("'")
#     access_token = (os.getenv("WHATSAPP_ACCESS_TOKEN") or "").strip().strip('"').strip("'")
#
#     print(f"📩 Processing inbound WhatsApp message from {sender_phone}: '{message_text}'")
#
#     clean_phone = re.sub(r'\D', '', sender_phone)
#
#     # ── Handle Reset Command ──────────────────────────────────────────────────
#     cleaned_input = message_text.strip().lower()
#     if cleaned_input in ("reset", "restart", "clear", "new", "নতুন চ্যাট", "আলাপ শুরু"):
#         whatsapp_sessions.pop(sender_phone, None)
#         whatsapp_sessions.pop(clean_phone, None)
#         has_bn = bool(re.search(r'[\u0980-\u09FF]', message_text))
#         ai_reply = "নতুন চ্যাট শুরু করা হলো! আপনার প্রশ্নটি লিখুন। 🌸" if has_bn else "Your conversation has been reset! How can HRITU support you today? 🌸"
#     else:
#         user_history = whatsapp_sessions.get(clean_phone, whatsapp_sessions.get(sender_phone, []))
#         has_bangla_script = bool(re.search(r'[\u0980-\u09FF]', message_text))
#         lang = "bn" if has_bangla_script else "en"
#
#         try:
#             if assistant:
#                 ai_reply = await asyncio.wait_for(
#                     asyncio.to_thread(assistant.chat_lifestyle, user_history, message_text, lang),
#                     timeout=8.0
#                 )
#             else:
#                 ai_reply = "Hello! I am HRITU, your Women's Health & PCOS AI Assistant. How can I support you today? 🌸"
#         except Exception as e:
#             print(f"❌ HRITU AI Engine error processing WhatsApp message: {e}")
#             ai_reply = "Hello! I am HRITU, your Women's Health & PCOS AI Assistant. How can I support you today? 🌸"
#
#         user_history.append({"role": "user", "content": message_text})
#         user_history.append({"role": "assistant", "content": ai_reply})
#         whatsapp_sessions[clean_phone] = user_history[-16:]
#
#     ai_reply_formatted = re.sub(r'<[^>]+>', '', ai_reply).strip()
#     if len(ai_reply_formatted) > 4000:
#         ai_reply_formatted = ai_reply_formatted[:3990] + "..."
#
#     if not phone_number_id or not access_token:
#         print("⚠️ Missing WHATSAPP_PHONE_NUMBER_ID or WHATSAPP_ACCESS_TOKEN in environment variables.")
#         return
#
#     graph_url = f"https://graph.facebook.com/v22.0/{phone_number_id}/messages"
#     headers = {
#         "Authorization": f"Bearer {access_token}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "messaging_product": "whatsapp",
#         "recipient_type": "individual",
#         "to": clean_phone,
#         "type": "text",
#         "text": {
#             "preview_url": False,
#             "body": ai_reply_formatted
#         }
#     }
#
#     try:
#         async with httpx.AsyncClient(timeout=10.0) as client:
#             resp = await client.post(graph_url, headers=headers, json=payload)
#             print(f"📤 Meta Graph API Dispatch Status: {resp.status_code}, Response: {resp.text}")
#     except Exception as req_err:
#         print(f"❌ Network exception while calling Meta Graph API: {req_err}")
#
#
# @app.get("/webhook")
# @app.get("/api/webhook")
# def verify_whatsapp_webhook(
#     hub_mode: Optional[str] = Query(None, alias="hub.mode"),
#     hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
#     hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
# ):
#     """
#     Meta WhatsApp Cloud API GET Webhook Verification Handshake.
#     Verifies that hub.mode == 'subscribe' and hub.verify_token matches WHATSAPP_VERIFY_TOKEN.
#     Returns hub.challenge as plain text string.
#     """
#     expected_verify_token = (os.getenv("WHATSAPP_VERIFY_TOKEN") or "hritu_secret_webhook_token_2026").strip()
#
#     if hub_mode == "subscribe" and hub_verify_token == expected_verify_token:
#         print(f"✅ Meta Webhook Verification Handshake successful! Returning challenge: {hub_challenge}")
#         return Response(content=str(hub_challenge or "OK"), media_type="text/plain")
#     else:
#         print(f"❌ Webhook Verification Failed: Received mode '{hub_mode}', token '{hub_verify_token}' (Expected: '{expected_verify_token}').")
#         raise HTTPException(status_code=403, detail="Webhook verification token mismatch.")
#
#
# @app.post("/webhook")
# @app.post("/api/webhook")
# async def receive_whatsapp_webhook(request: Request):
#     """
#     Meta WhatsApp Cloud API POST Inbound Webhook.
#     Receives real-time incoming messages, parses sender phone number and message text,
#     and dispatches processing via HRITU AI Engine and Meta Graph API v22.0.
#     """
#     try:
#         body = await request.json()
#         print(f"📥 Received Webhook Payload: {body}")
#
#         entry = body.get("entry", [])
#         for e in entry:
#             changes = e.get("changes", [])
#             for c in changes:
#                 value = c.get("value", {})
#                 messages = value.get("messages", [])
#                 if messages:
#                     for msg in messages:
#                         sender_phone = msg.get("from")
#                         msg_type = msg.get("type")
#
#                         if msg_type == "text":
#                             message_text = msg.get("text", {}).get("body", "")
#                         else:
#                             message_text = msg.get("interactive", {}).get("button_reply", {}).get("title", "") or "Hello"
#
#                         if sender_phone and message_text:
#                             print(f"📩 Processing WhatsApp message from {sender_phone}: '{message_text}'")
#                             await process_hritu_response(sender_phone, message_text)
#     except Exception as err:
#         print(f"❌ Error parsing WhatsApp webhook payload: {err}")
#
#     # Satisfy Meta's requirement for an immediate HTTP 200 response
#     return {"status": "ok"}

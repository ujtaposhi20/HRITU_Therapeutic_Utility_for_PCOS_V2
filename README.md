# HRITU (ঋতু)

<p align="center">
  <img src="frontend/static/logo.jpg" alt="HRITU logo" width="120" />
</p>

HRITU (Human-centered Responsive and Integrated Therapeutic Utility) is a web chatbot that helps young women in Bangladesh learn about PCOS (Polycystic Ovary Syndrome) and menstrual health. It speaks English, Bangla, and Banglish.

HRITU talks like a caring elder sister (আপু). It does not blame or judge the user. It does not diagnose PCOS and does not suggest medicines. When something needs medical attention, it tells the user to see a doctor.

HRITU was built for an HCI user study at BRAC University.

**Live demo:** https://hritu.vercel.app/

## What it can do

HRITU has four parts:

1. **PCOS Check** – Asks 18 short questions following Rotterdam criteria about age, height, weight, periods, and symptoms. The user can answer the questions. HRITU then shows all the answers so the user can fix any mistakes. After the user confirms, it gives a result that is a screening hint, not a diagnosis.
2. **Talk** – A space to talk about feelings, body changes, fertility worries, and family or social pressure.
3. **Lifestyle** – Food and exercise tips that fit everyday life in Bangladesh, such as red rice, leafy greens, and local fish.
4. **Learn Cycle** – Explains the four phases of the menstrual cycle, with a short quiz.

Replies can also be read aloud.

## How the PCOS Check works

Three machine learning models look at the user's answers: Random Forest, Decision Tree, and Support Vector Machine (SVM). If at least two of them say "yes", the result is *Suspected*. Otherwise it is *Not suspected*.

HRITU also shows which answers affected the result most, using SHAP and LIME.

The models use these answers: age, BMI (from height and weight), age at first period, cycle length, age when symptoms started, cycle pattern, period delay, symptoms, acne level, and hair loss.

## Safety and privacy

- **Crisis messages:** If a user writes about self-harm or suicide in English, Bangla, or Banglish, HRITU skips the AI completely. It replies straight away with the Kaan Pete Roi (কান পেতে রই) helpline and the national emergency number 999.
- **Personal details are removed:** Phone numbers, NID numbers, email addresses, names, and house addresses are removed before a message goes to the AI. Place names like Dhaka or Sylhet are kept, so the advice can fit the user's area.
- **Medicine check:** HRITU is told never to suggest medicines. As an extra check, every AI reply is scanned for medicine names such as metformin or letrozole. If one is found, a note is added telling the user to talk to a doctor before taking any medicine.
- **No chat logs:** Conversations are not saved. Only the anonymous PCOS Check answers are saved to Google Sheets. If Sheets is not set up, they go to a local CSV file instead.

## Where answers come from

HRITU uses Google Gemini to write replies. First it looks for a match in 16 health fact cards written in English and Bangla (`backend/knowledge_cards.json`). If none of the cards match, it searches DuckDuckGo for more information.

## Project structure

```
frontend/              Web page (index.html, styles.css, app.js, logo)
backend/
  main.py              Web server (FastAPI)
  agent.py             Connects to Gemini and handles web search
  screening.py         PCOS Check questions
  sister.py            Talk
  lifestyle.py         Lifestyle
  learn.py             Learn Cycle
  safety.py            Crisis check and medicine filter
  privacy.py           Removes personal details
  knowledge.py         Searches the fact cards
  knowledge_cards.json The 16 fact cards
  preprocessor.py      Prepares answers for the models
  explainability.py    SHAP and LIME
  google_sheets.py     Saves results to Google Sheets
  ml models/           Trained model files
requirements.txt       Python packages
vercel.json            Vercel settings
```

## Run it on your computer

You need Python 3.10 or newer (3.12 recommended) and a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

```bash
git clone https://github.com/ujtaposhi20/HRITU_Therapeutic_Utility_for_PCOS_V2.git
cd HRITU_Therapeutic_Utility_for_PCOS_V2

python -m venv backend/venv
# Windows:
backend\venv\Scripts\activate
# Mac/Linux:
source backend/venv/bin/activate

pip install -r requirements.txt
pip install shap lime    # optional, needed for the "which answers mattered" feature
```

Create a file called `.env` inside the `backend` folder:

```env
GEMINI_API_KEY=your-gemini-api-key
PRIMARY_MODEL=gemini-flash-latest
FALLBACK_MODEL=gemini-flash-lite-latest

# Optional: save PCOS Check results to Google Sheets
GOOGLE_SHEET_ID=your-sheet-id
GOOGLE_SERVICE_ACCOUNT_JSON={"type": "service_account", ...}
```

For `GOOGLE_SERVICE_ACCOUNT_JSON`, paste the full contents of your service account JSON file on one line. Do not put the file name there.

Start the app:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000.

Do not upload your `.env` file or any credentials file to GitHub. Both are already listed in `.gitignore`.

## Deploy on Vercel

1. Import this repository on [vercel.com](https://vercel.com).
2. Under **Settings → Environment Variables**, add the same values you put in `.env` (at least `GEMINI_API_KEY`).
3. Deploy. After that, every push to `main` updates the site automatically.

Vercel cannot save files, so on Vercel the CSV backup does not work. Set up Google Sheets if you want to keep PCOS Check results.

## Disclaimer

HRITU is for health awareness only. It is not a medical device and does not diagnose any condition. Its results depend only on what the user types in. Anyone with severe pain, low mood, or ongoing symptoms should see a gynecologist, an endocrinologist, or a mental health professional.

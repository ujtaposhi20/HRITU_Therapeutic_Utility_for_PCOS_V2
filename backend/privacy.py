"""
privacy.py — Bangladeshi PII Anonymizer.

Strips personal information (phone numbers, NIDs, names, addresses, emails)
from user messages BEFORE they are sent to external LLMs (Groq/OpenAI).
Supports:
  • Bangladeshi phone formats (01..., +8801...) with Bangla digits (০-৯)
  • National ID numbers (10, 13, or 17 digits)
  • Self-introductions in English, Bangla, and Banglish
  • House/road address fragments
  • Email addresses

No chat logs are saved to local disk. Only anonymous session summaries
are persisted to Google Sheets (and CSV fallback).
"""

import re


DIGIT = r"[0-9০-৯]"

PHONE_RE = re.compile(
    rf"(\+?[8৮][8৮][0০]?[\s\-]?)?[0০]?[1১][3-9৩-৯](?:[\s\-]?{DIGIT}){{8}}"
)

EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.]+")
LONG_NUMBER_RE = re.compile(rf"{DIGIT}{{10,17}}")


NAME_PATTERNS = [
    re.compile(
        r"(my name is|i am called|i'm called|call me)\s+"
        r"([A-Za-z]+(?:\s+[A-Z][a-z]+)?)",
        re.IGNORECASE,
    ),
    re.compile(r"(আমার নাম)\s+([^\s,।.]+)"),
    re.compile(
        r"(amar nam|amar naam|ami holi)\s+([A-Za-z]+)",
        re.IGNORECASE,
    ),
]

ADDRESS_RE = re.compile(
    r"\b(house|road|flat|block|holding|apt|apartment|বাসা|রোড|ফ্ল্যাট|গলি|হোল্ডিং)\s*(no\.?|নং|number)?\s*([0-9০-৯]+[a-zA-Z0-9০-৯/\-]*)",
    re.IGNORECASE,
)

GEO_WHITELIST = {
    # Countries
    "bangladesh", "bd", "india", "pakistan", "uk", "usa", "canada", "australia",
    "বাংলাদেশ", "ভারত",
    # Divisions & major districts/cities of Bangladesh
    "dhaka", "chittagong", "chattogram", "sylhet", "rajshahi", "khulna", "barishal", "barisal",
    "rangpur", "mymensingh", "comilla", "cumilla", "gazipur", "narayanganj", "bogra", "bogura",
    "cox's bazar", "coxsbazar", "jashore", "jessore", "dinajpur", "tangail", "feni", "noakhali",
    "ঢাকা", "চট্টগ্রাম", "সিলেট", "রাজশাহী", "খুলনা", "বরিশাল", "রংপুর", "ময়মনসিংহ",
    "কুমিল্লা", "গাজীপুর", "নারায়ণগঞ্জ", "বগুড়া", "কক্সবাজার", "যশোর", "দিনাজপুর",
    # Regional / Habitat words
    "village", "town", "city", "district", "division", "country", "habitat",
    "গ্রাম", "শহর", "জেলা", "বিভাগ", "দেশ",
}


def redact(text: str) -> str:
    """
    Return the text with personal info replaced by placeholder tags.
    Protects and preserves broader location (country/state/division/city)
    so lifestyle & diet suggestions align with the user's local habitat.
    """
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = LONG_NUMBER_RE.sub("[ID]", text)
    text = ADDRESS_RE.sub("[ADDRESS]", text)
    
    for pattern in NAME_PATTERNS:
        def _replace_name(m):
            prefix = m.group(1)
            name_candidate = m.group(2).strip()
            words = set(re.findall(r"[\w]+", name_candidate.lower()))
            if words & GEO_WHITELIST:
                return m.group(0)
            return f"{prefix} [NAME]"
        text = pattern.sub(_replace_name, text)
        
    return text

if __name__ == "__main__":
    tests = [
        "My name is Nusrat Jahan, call 01712-345678",
        "আমার নাম রিয়া, আমার নম্বর ০১৮১২৩৪৫৬৭৮",
        "amar naam Tania, email tania@gmail.com",
        "I live at house 12, road 5. NID 1234567890123",
        "I am 24 and my period is late",    
        "+8801712345678 is my number",
    ]
    for t in tests:
        print(f"  IN:  {t}")
        print(f"  OUT: {redact(t)}\n")

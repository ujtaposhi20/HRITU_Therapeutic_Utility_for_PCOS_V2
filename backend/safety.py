"""
safety.py — Pre-LLM Crisis Interceptor & Post-LLM Medicine Guardrail.

Runs BEFORE every LLM call (is_crisis, is_off_topic) and
AFTER every LLM reply (check_reply) to enforce patient safety.

Rule-based on purpose: safety must work even when the LLM is down.
"""

import re

CRISIS_WORDS = [
    # English
    "suicide", "kill myself", "killing myself", "end my life", "ending my life", "want to die", "wanna die", "self harm",
    "self harming", "hurt myself", "hurting myself", "no reason to live", "better off dead", "wish i was dead", "wish i were dead",
    # Bangla script
    "আত্মহত্যা", "মরে যেতে চাই", "মরতে চাই", "বাঁচতে চাই না", "বাঁচা না হলে", "বাঁচতে পারবো না",
    "বাঁচতে পারব না", "বাঁচতে ইচ্ছে করে না", "বাঁচতে ইচ্ছা করে না",
    "মারা যাব", "মরে যাব", "মরে যেতে ইচ্ছে করে", "মরে যেতে ইচ্ছে", "মরে যেতে ইচ্ছা", "মরে যাই",
    "বেঁচে থেকে লাভ নেই", "বেঁচে থেকে লাভ নাই", "বেঁচে থেকে কি লাভ", "বেঁচে থাকার ইচ্ছা নেই",
    "নিজেকে শেষ", "আর বাঁচতে চাই না", "জীবন শেষ করতে চাই", "জীবন শেষ করে দেব", "নিজেকে মেরে ফেলব", "মেরে ফেলব",
    # Banglish death & self-harm
    "mara jabo", "mora jabo", "more jabo", "more jete chai", "morte chai", "more jete icche",
    "more jete icche kore", "more jet e icche kore", "more jete ichhe", "more jete mon chay",
    "bachte chai na", "bachte chaina", "bachte icche kore na", "bachte parbo na",
    "bancha na hole", "beche theke labh nei", "beche theke labh nai", "beche theke lav nai",
    "beche theke ki hobe", "attohotta", "atmohotta", "suiside",
    "nijer jibon ses", "nijer jibon sesh", "ar bachte chai na", "mere felbo", "nijeke mere",
]

CRISIS_PATTERNS = [
    # English self-harm & suicide
    re.compile(r'\b(suicid(e|al)|kill(ing)?\s+myself|end(ing)?\s+my\s+life|want\s+to\s+die|wanna\s+die|self\s*harm(ing)?|hurt(ing)?\s+myself|better\s+off\s+dead|no\s+reason\s+to\s+live|wish\s+i\s+(was|were)\s+dead|take\s+my\s+(own\s+)?life|taking\s+my\s+(own\s+)?life)\b', re.IGNORECASE),
    # Bangla script
    re.compile(r'(আত্মহত্যা|মরে\s*যেতে|মরতে\s*চাই|মারা\s*যাব|মরে\s*যাব|বাঁচতে\s*চাই\s*না|বাঁচতে\s*(পারব|পারবো)\s*না|বাঁচা\s*না\s*হলে|বাঁচতে\s*ইচ্ছে\s*করে\s*না|বেঁচে\s*থেকে\s*লাভ\s*(নেই|নাই|কি)|বেঁচে\s*থাকার\s*ইচ্ছা\s*নেই|বেঁচে\s*থাকতে\s*চাই\s*না|নিজেকে\s*(শেষ|মেরে)|জীবন\s*শেষ)', re.IGNORECASE),
    # Banglish die / death
    re.compile(r'\b(m[ao]r[ae]\s*j[ao]b[oai]|m[ao]r[ae]\s*j[a-z]*\s*(icche|ichhe|chai|mon|kore)|morte\s*chai|more\s*j[a-z]*|mora\s*jete)\b', re.IGNORECASE),
    re.compile(r'\bmore\s*jet\s*e\s*(icche|ichhe|mon|chai|kore)?\b', re.IGNORECASE),
    re.compile(r'\b(mara|more|mora)\s+jabo\b', re.IGNORECASE),
    # Banglish no wish to live / giving up
    re.compile(r'\b(b[a-z]*ch[a-z]*\s*na\s*hole|b[a-z]*ch[a-z]*\s*(chai\s*na|chaina|parbo\s*na|icche\s*kore\s*na|pari\s*na))\b', re.IGNORECASE),
    re.compile(r'\b(beche|bache)\s*th[a-z]*\s*(labh|lav|ortho)?\s*(nei|nai|ki|hobe)\b', re.IGNORECASE),
    # Banglish self harm / ending life
    re.compile(r'\b(attohotta|atmohotta|suiside)\b', re.IGNORECASE),
    re.compile(r'\b(nijeke|amake)\s*(sesh|shesh|ses|mere\s*fel[a-z]*)\b', re.IGNORECASE),
    re.compile(r'\b(jibon|jeebon)\s*(sesh|shesh|ses)\b', re.IGNORECASE),
]

CRISIS_REPLY = {
    "en": (
        "Apu, I'm really glad you told me. You matter, and you don't have to carry this alone.\n\n"
        "🔴 **Please reach out now:**\n"
        "• **Kaan Pete Roi (কান পেতে রই)** — Free emotional support helpline for Bangladesh\n"
        "• **National Emergency: 999**\n\n"
        "If you have someone you trust — a family member, friend, or teacher — please talk to them too. "
        "I'm still here with you. 💛"
    ),
    "bn": (
        "আপু, আমাকে বলেছো বলে আমি সত্যিই খুশি। তুমি অনেক গুরুত্বপূর্ণ, একা এটা বয়ে বেড়াতে হবে না।\n\n"
        "🔴 **এখনই কারো সাথে কথা বলো:**\n"
        "• **কান পেতে রই** — বিনামূল্যে মানসিক সহায়তা হেল্পলাইন\n"
        "• **জরুরি সেবা: ৯৯৯**\n\n"
        "তোমার বিশ্বাসের কেউ থাকলে — পরিবার, বন্ধু, বা শিক্ষক — তাদের সাথেও কথা বলো। "
        "আমি তোমার পাশেই আছি। 💛"
    ),
}

# Medicine names the bot must never recommend (checked in LLM replies).
MEDICINE_WORDS = [
    "metformin", "letrozole", "clomiphene", "clomid", "spironolactone",
    "birth control pill", "inositol", "provera", "diane-35", "yasmin",
    "মেটফরমিন", "লেট্রোজোল", "ক্লোমিফেন", "ক্লোমিড",
    "dose", "mg ", "dosage", "prescription",
]

DOCTOR_NOTE = {
    "en": (
        "\n\n💡 *I want to gently remind you that I can't suggest specific medicines or dosages — "
        "please consult a gynaecologist or endocrinologist for proper treatment.* 💕"
    ),
    "bn": (
        "\n\n💡 *আমি ওষুধ বা ডোজের পরামর্শ দিতে পারি না — "
        "চিকিৎসার জন্য একজন গাইনি বা এন্ডোক্রাইনোলজিস্টের সাথে কথা বলো।* 💕"
    ),
}

# Stay on topic. These rules are injected into EVERY LLM call.
SCOPE_RULES = """STRICT RULES (always follow):
- Only talk about: periods, PCOS, women's health, lifestyle (sleep, activity, stress, food habits), and emotional support.
- If asked about anything else (coding, politics, religion debates, celebrities, homework, news), say kindly that you can only help with health and wellbeing, and offer the 4 module options.
- Never name medicines, doses, or supplements. Never diagnose. Suggest a doctor for treatment.
- Never ask for her name, phone number, address, or any ID. If she shares them, do not repeat them.
- Keep replies short and focused. Ask at most one question per response."""


# Quick rule-based check for clearly off-topic requests (works even without the LLM).
OFF_TOPIC_WORDS = [
    "python", "javascript", "coding", "programming", "homework", "math equation",
    "solve math", "election", "politics", "cricket score", "movie review",
    "song lyrics", "bitcoin", "cryptocurrency", "রাজনীতি", "নির্বাচন", "কোডিং", "প্রোগ্রামিং",
]

OFF_TOPIC_PATTERNS = [
    # Programming & coding requests
    re.compile(r'\b(write|create|generate|give\s+me|fix|debug)\s+.*(code|program|script|function|class|algorithm|regex)\b', re.IGNORECASE),
    re.compile(r'\b(python|javascript|typescript|c\+\+|java\b|html|css|php|ruby|rust|sql|golang)\b', re.IGNORECASE),
    re.compile(r'\b(coding|programming|developer|github|terminal|bash\s+script|compiler)\b', re.IGNORECASE),
    re.compile(r'\b(কোডিং|প্রোগ্রামিং|কোড\s*লিখ|পাইথন)\b', re.IGNORECASE),
    # Homework, math, academic problems
    re.compile(r'\b(homework|solve\s+(this\s+)?(math|equation|problem)|algebra|calculus|derivative|integral)\b', re.IGNORECASE),
    re.compile(r'\b(write\s+(an\s+)?essay|write\s+(a\s+)?poem\s+about\s+(?!pcos|period|health))\b', re.IGNORECASE),
    # Finance, crypto, politics, sports
    re.compile(r'\b(bitcoin|crypto|cryptocurrency|stock\s+market|forex|invest\s+in)\b', re.IGNORECASE),
    re.compile(r'\b(election|politics|political|vote\s+for|government\s+policy)\b', re.IGNORECASE),
    re.compile(r'\b(cricket\s+score|football\s+score|fifa|ipl\s+match)\b', re.IGNORECASE),
    re.compile(r'\b(movie\s+review|song\s+lyrics|tell\s+me\s+a\s+joke)\b', re.IGNORECASE),
    re.compile(r'\b(রাজনীতি|নির্বাচন|ক্রিকেট|ফুটবল|বিটকয়েন)\b', re.IGNORECASE),
]

OFF_TOPIC_REPLY = {
    "en": (
        "Sister, I am HRITU, a specialized health companion for PCOS, menstrual cycles, fertility, and daily wellness. 🌸\n\n"
        "I cannot help with coding, software programming, math homework, or general non-health topics. "
        "Please feel free to ask me anything about your cycle, symptoms, diet, or reproductive wellness!"
    ),
    "bn": (
        "আপু, আমি ঋতু — PCOS, মাসিক স্বাস্থ্য, প্রজনন এবং জীবনযাপন সচেতনতার বিশেষায়িত স্বাস্থ্য সহকারী। 🌸\n\n"
        "আমি প্রোগ্রামিং, কোডিং, হোমওয়ার্ক বা স্বাস্থ্য বহির্ভূত বিষয়ে সাহায্য করতে পারি না। "
        "তোমার মাসিক, শারীরিক লক্ষণ, পুষ্টি বা প্রজনন স্বাস্থ্য নিয়ে যেকোনো প্রশ্ন থাকলে আমাকে নিঃসংকোচে বলতে পারো!"
    ),
}


def is_crisis(text: str) -> bool:
    """Return True if the message contains any crisis/self-harm keyword or pattern."""
    if not text:
        return False
    norm = " ".join(text.lower().split())
    # 1. Regex pattern check (handles spaces, word boundaries, variations)
    for pattern in CRISIS_PATTERNS:
        if pattern.search(norm):
            return True
    # 2. Exact substring check across extended vocabulary
    if any(word in norm for word in CRISIS_WORDS):
        return True
    return False


def get_crisis_reply(text: str) -> dict:
    """
    Return a crisis response dict with 'response', 'is_crisis' flag, and
    detected language. Ready to be sent directly to the frontend.
    """
    has_bangla = bool(re.search(r'[\u0980-\u09FF]', text))
    lang = "bn" if has_bangla else "en"
    return {
        "response": CRISIS_REPLY[lang],
        "is_crisis": True,
        "lang": lang,
    }


def is_off_topic(text: str) -> bool:
    """Return True if the message contains clearly off-topic patterns or keywords."""
    if not text:
        return False
    norm = " ".join(text.lower().split())
    for pattern in OFF_TOPIC_PATTERNS:
        if pattern.search(norm):
            return True
    return any(word in norm for word in OFF_TOPIC_WORDS)


def get_off_topic_reply(text: str) -> str:
    """Return a polite off-topic refusal message."""
    has_bangla = bool(re.search(r'[\u0980-\u09FF]', text))
    lang = "bn" if has_bangla else "en"
    return OFF_TOPIC_REPLY[lang]


def check_reply(reply: str, lang: str = "en") -> str:
    """
    Post-LLM safety check: if the reply mentions a medicine name,
    append a doctor-referral note.
    """
    lower = reply.lower()
    if any(word in lower for word in MEDICINE_WORDS):
        return reply + DOCTOR_NOTE.get(lang, DOCTOR_NOTE["en"])
    return reply

if __name__ == "__main__":
    test_messages = [
        "I want to die",
        "আমি মরে যেতে চাই",
        "bachte chai na",
        "I'm tired today",
        "write me a python code",
    ]
    for msg in test_messages:
        print(f"  {msg!r:30} → crisis={is_crisis(msg)}, off_topic={is_off_topic(msg)}")

    print("\nMedicine check:")
    print(check_reply("You could try metformin 500 mg.", "en"))

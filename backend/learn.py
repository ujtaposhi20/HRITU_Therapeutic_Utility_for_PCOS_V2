from langchain_core.prompts import ChatPromptTemplate
import safety as safety_module
import knowledge as knowledge_module


LESSONS = [
    {
        "title": {"en": "Phase 1: Menstrual Phase (Period) — Days 1-5", "bn": "ধাপ ১: মাসিক (পিরিয়ড) — প্রায় ১-৫ দিন"},
        "en": "The lining of the uterus comes out as blood. Tiredness and cramps are common.",
        "bn": "জরায়ুর ভেতরের আস্তরণ রক্ত হিসেবে বের হয়ে আসে। ক্লান্তি ও পেটব্যথা স্বাভাবিক।",
        "care": {
            "en": "Rest when you can. A warm water bottle on the tummy can ease cramps. Drink water, and eat iron-rich foods like spinach (shak), lentils (dal), fish, and eggs. Change pads every 4-6 hours.",
            "bn": "সুযোগ পেলে বিশ্রাম নাও। পেটে গরম পানির বোতল দিলে ব্যথা কমতে পারে। পানি খাও, আর আয়রনযুক্ত খাবার খাও যেমন শাক, ডাল, মাছ, ডিম। ৪-৬ ঘণ্টা পরপর প্যাড বদলাও।"
        },
    },
    {
        "title": {"en": "Phase 2: Follicular Phase — Days 1-13", "bn": "ধাপ ২: ফলিকুলার ধাপ — প্রায় ১-১৩ দিন"},
        "en": "An egg starts to grow inside the ovary. Energy often goes up in this phase.",
        "bn": "ডিম্বাশয়ের ভেতরে একটা ডিম বড় হতে শুরু করে। এই সময়ে সাধারণত শক্তি বাড়ে।",
        "care": {
            "en": "A good time to be more active: walking, dancing, or exercise. Plan tasks that need energy for these days.",
            "bn": "বেশি সক্রিয় হওয়ার ভালো সময়: হাঁটা, নাচ বা ব্যায়াম। যে কাজে শক্তি লাগে, সেগুলো এই দিনগুলোতে রাখো।"
        },
    },
    {
        "title": {"en": "Phase 3: Ovulation — Around Day 14", "bn": "ধাপ ৩: ওভুলেশন — প্রায় ১৪তম দিনে"},
        "en": "The ovary releases the egg. With PCOS, ovulation often doesn't happen regularly. That is why periods can come late, or skip months.",
        "bn": "ডিম্বাশয় থেকে ডিম বের হয়। PCOS থাকলে প্রায়ই ওভুলেশন নিয়মিত হয় না। এজন্যই মাসিক দেরিতে হয় বা কয়েক মাস বাদ যায়।",
        "care": {
            "en": "Track your period dates in a calendar or app. If periods stop for 3 months or more, see a gynaecologist.",
            "bn": "ক্যালেন্ডার বা অ্যাপে মাসিকের তারিখ লিখে রাখো। ৩ মাস বা বেশি মাসিক না হলে গাইনি ডাক্তার দেখাও।"
        },
    },
    {
        "title": {"en": "Phase 4: Luteal Phase — Days 15-28", "bn": "ধাপ ৪: লুটিয়াল ধাপ — প্রায় ১৫-২৮ দিন"},
        "en": "The body prepares for a possible pregnancy. If it doesn't happen, hormones drop and the next period starts. Mood changes and cravings are common before a period.",
        "bn": "শরীর সম্ভাব্য গর্ভধারণের জন্য প্রস্তুত হয়। না হলে হরমোন কমে যায় আর পরের মাসিক শুরু হয়। মাসিকের আগে মেজাজ বদলানো ও খাওয়ার ইচ্ছা বাড়া স্বাভাবিক।",
        "care": {
            "en": "Be gentle with yourself. Sleep well, eat regular meals, and choose fruit or nuts when sweet cravings come. Light exercise can lift your mood.",
            "bn": "নিজের প্রতি নরম থাকো। ভালো ঘুমাও, নিয়মিত খাও, মিষ্টি খেতে ইচ্ছা করলে ফল বা বাদাম নাও। হালকা ব্যায়াম মন ভালো করতে পারে।"
        },
    },
]

QUIZ = {
    "en": "Quick quiz! Why can periods be late with PCOS?\n1) Too much water  2) Ovulation doesn't happen regularly  3) Too much sleep",
    "bn": "ছোট্ট কুইজ! PCOS থাকলে মাসিক দেরিতে হয় কেন?\n১) বেশি পানি খাওয়া  ২) ওভুলেশন নিয়মিত হয় না  ৩) বেশি ঘুম",
}
QUIZ_ANSWERS = {"2", "২"}

NEXT_WORDS = {"next", "n", "continue", "porer", "porer ta", "পরের", "পরেরটা", "এগোও", "skip", "বাদ"}


def chat_learn(history, message, learn_state, language="en", assistant=None):
    """
    Module 4: Learn Your Period Cycle.
    Interactive curriculum with 4 phases, quiz, and free-form Q&A.
    Uses Two-Tier Hybrid RAG for Q&A after curriculum.

    Args:
        learn_state: dict with 'learn_step' key tracking progress.
                     None/missing → start; 0-3 → lesson index; 'quiz' → quiz time;
                     'open' → free Q&A mode.
    Returns:
        (response_text, updated_learn_state)
    """
    if assistant is None:
        raise ValueError("assistant instance must be provided to chat_learn")

    lang = "bn" if language == "bn" else "en"
    step = learn_state.get("learn_step")
    wants_next = message.strip().lower() in NEXT_WORDS

    if step is None:
        learn_state["learn_step"] = 0
        lesson = LESSONS[0]
        text = f"**{lesson['title'][lang]}**\n{lesson[lang]}\n\n💡 {lesson['care'][lang]}"
        hint = "\n\n*(Ask me anything about this phase, or type 'next' to continue)*" if lang == "en" else "\n\n*(এই ধাপ নিয়ে যা খুশি জিজ্ঞেস করো, অথবা 'পরের' লিখো এগোতে)*"
        return text + hint, learn_state

    if step == "quiz":
        correct = message.strip() in QUIZ_ANSWERS
        learn_state["learn_step"] = "open"
        if correct:
            reply = "Correct! 🌸 With PCOS, ovulation is often irregular, so periods come late. Well done, apu!" if lang == "en" else "সঠিক! 🌸 PCOS থাকলে ওভুলেশন প্রায়ই অনিয়মিত হয়, তাই মাসিক দেরিতে হয়। দারুণ আপু!"
        else:
            reply = "Not quite. The answer is 2: ovulation doesn't happen regularly. That's okay, now you know! 🌸" if lang == "en" else "পুরোপুরি না। সঠিক উত্তর ২: ওভুলেশন নিয়মিত হয় না। কোনো সমস্যা নেই, এখন তো জানলে! 🌸"
        open_hint = "\n\nYou can still ask me anything about periods and self-care." if lang == "en" else "\n\nমাসিক আর নিজের যত্ন নিয়ে এখনও যা খুশি জিজ্ঞেস করতে পারো।"
        return reply + open_hint, learn_state

    if step == "open" or (isinstance(step, int) and not wants_next):
        extra_context = None
        knowledge_context = knowledge_module.get_card_context(message, lang=lang)
        if knowledge_context:
            extra_context = knowledge_context
            print(f"📚 Tier-1 Knowledge Hit for learn Q&A: {extra_context[:80]}...")
        else:
            from agent import run_web_search
            web_search_res = run_web_search(message, max_results=3)
            if web_search_res and web_search_res != "No additional web search context found.":
                extra_context = web_search_res
                print(f"🔍 Tier-2 DuckDuckGo fallback for learn Q&A")
            elif isinstance(step, int):
                lesson = LESSONS[step]
                extra_context = f"{lesson['title'][lang]}\n{lesson[lang]}\n💡 {lesson['care'][lang]}"
            else:
                extra_context = "No additional context."

        system_prompt = (
            f"You are HRITU, a warm elder-sister AI teaching about period cycle phases and self-care.\n"
            f"NON-NEGOTIABLE DOMAIN LOCK: You only discuss period cycle phases, menstrual health, and self-care. NEVER write programming code (Python, etc.), solve math/homework, or answer general non-health questions. If asked about off-topic subjects, politely refuse and redirect to period cycle learning.\n"
            f"Answer the user's question using the context below. Keep it short (2-3 sentences), warm, and simple.\n"
            f"Context:\n{extra_context}\n"
            f"Language: {'Bangla' if lang == 'bn' else 'English'}. Use 'tumi' (তুমি) forms in Bangla."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        try:
            response = assistant._invoke_llm_resilient(prompt, {"input": message})
        except Exception:
            if isinstance(step, int):
                response = "💡 " + LESSONS[step]["care"][lang]
            else:
                response = "💡 " + LESSONS[0]["care"][lang]

        response = safety_module.check_reply(response, lang)
        hint = ""
        if isinstance(step, int):
            hint = "\n\n*(Ask me anything about this phase, or type 'next' to continue)*" if lang == "en" else "\n\n*(এই ধাপ নিয়ে যা খুশি জিজ্ঞেস করো, অথবা 'পরের' লিখো এগোতে)*"
        return response + hint, learn_state

    if isinstance(step, int) and step < len(LESSONS) - 1:
        learn_state["learn_step"] = step + 1
        lesson = LESSONS[step + 1]
        text = f"**{lesson['title'][lang]}**\n{lesson[lang]}\n\n💡 {lesson['care'][lang]}"
        hint = "\n\n*(Ask me anything about this phase, or type 'next' to continue)*" if lang == "en" else "\n\n*(এই ধাপ নিয়ে যা খুশি জিজ্ঞেস করো, অথবা 'পরের' লিখো এগোতে)*"
        return text + hint, learn_state
    learn_state["learn_step"] = "quiz"
    return QUIZ[lang], learn_state

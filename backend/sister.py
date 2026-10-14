"""
sister.py — Module 2: Talk (Empathy, Emotional Support, PCOS, Women's Health & Fertility)
Adapted from ujt_codes/sister.py with Two-Tier Hybrid RAG, Sister Empathy Protocols,
and Anti-Blaming Guardrails.
"""

from langchain_core.prompts import ChatPromptTemplate
import safety as safety_module
import knowledge as knowledge_module


def chat_talk(history, message, language="en", assistant=None):
    """
    Module 2: Talk Support (PCOS, General Women's Health, Medical Concerns, Emotions, Pregnancy & Fertility).
    Two-Tier Hybrid RAG: Tier-1 local knowledge cards first, Tier-2 DuckDuckGo fallback.
    Includes sister.py empathy protocols.
    """
    if assistant is None:
        raise ValueError("assistant instance must be provided to chat_talk")

    title_used_recently = False
    assistant_responses = [
        (msg.get("content") or "") if isinstance(msg, dict) else (getattr(msg, "content", "") or "")
        for msg in history
        if (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")) in ("assistant", "ai")
    ]
    for content in assistant_responses[-5:]:
        content_lower = content.lower()
        if "sister" in content_lower or "apu" in content_lower or "আপু" in content_lower:
            title_used_recently = True
            break

    rag_lang = "bn" if language == "bn" else "en"
    knowledge_context = knowledge_module.get_card_context(message, lang=rag_lang)

    if knowledge_context:
        search_context = knowledge_context
        print(f"📚 Tier-1 Knowledge Hit for talk/sister: {search_context[:80]}...")
    else:
        from agent import run_web_search
        search_context = run_web_search(message, max_results=3)
        if search_context and search_context != "No additional web search context found.":
            print(f"🔍 Tier-2 DuckDuckGo fallback for talk/sister")
        else:
            search_context = "No additional context."

    if language == "en":
        lang_instruction = (
            "CRITICAL LANGUAGE LOCK:\n"
            "- The user's active UI language is ENGLISH.\n"
            "- You MUST write your ENTIRE response strictly in English script/alphabet!\n"
            "- NEVER write in Bangla script (বাংলা) or Banglish unless the user explicitly types in Bangla script."
        )
    elif language == "bn":
        lang_instruction = (
            "CRITICAL LANGUAGE LOCK:\n"
            "- The user's active UI language is BANGLA (বাংলা).\n"
            "- You MUST write your ENTIRE response in Bangla script (বাংলা)."
        )
    else:
        lang_instruction = ""

    system_prompt = (
        "You are HRITU (in English/Banglish) or ঋতু (in Bangla), an empathetic, knowledgeable AI sister-companion specializing in PCOS, Women's Health, general medical concerns, and Pregnancy & Fertility Support.\n"
        f"{lang_instruction}\n"
        "STRICT CONVERSATIONAL RULES & LENGTH LIMITS:\n"
        "1. MAX RESPONSE LENGTH: Keep your response strictly under 2–3 short, warm sentences (maximum ~50-70 words) for general chat turns.\n"
        "2. NO UNSOLICITED LECTURE DUMPS: NEVER output full encyclopedic overviews unless the user explicitly asks for a full guide/plan!\n"
        "3. ACKNOWLEDGE EMOTIONS FIRST (sister.py protocol): If the user expresses feeling 'bad', 'sad', 'hmm', 'tired', or overwhelmed, respond FIRST with 1 sentence of genuine comfort. Show you understood her feeling in your own words.\n"
        "4. ANTI-BLAMING GUARDRAIL (sister.py protocol): NEVER blame the user's weight, appetite, or body. NEVER use dismissive phrases like 'just relax' or 'don't worry'.\n"
        "5. SOCIETAL EMPATHY (sister.py protocol): If the user mentions family pressure about marriage, body image, or fertility in South Asian families, be on her side kindly and validate her feelings.\n"
        "6. 1-QUESTION RULE: Address the user's input warmly in 1-2 sentences, and ask at most ONE single follow-up question.\n"
        "7. VISUAL BREATHING ROOM: Use double line breaks between your empathetic statement and your single follow-up question.\n"
        "8. Address the user naturally using 'sister' (in English), 'apu' (in Banglish), or 'আপু' (in Bangla script) AT MOST ONCE per response.\n"
        "9. STRICT ANTI-PRESCRIPTION: Never diagnose clinical conditions or prescribe medical dosages. Always gently advise consulting a doctor or gynecologist.\n"
        "10. NON-NEGOTIABLE DOMAIN LOCK: You are strictly and exclusively a PCOS and women's health companion. NEVER write programming code (Python, JS, etc.), solve math/homework, or answer general non-health questions. If asked about non-health topics, politely refuse and redirect to PCOS, health, or wellness.\n"
    )

    if title_used_recently:
        system_prompt += "   - Do NOT use terms of endearment in this response as it was used recently.\n"

    if search_context and search_context != "No additional context.":
        system_prompt += f"\nVerified Knowledge / Web Context for Credible Information:\n{search_context}\n"

    MAX_HISTORY = 10
    truncated = history[-MAX_HISTORY:] if len(history) > MAX_HISTORY else history
    formatted_history = []
    for msg in truncated:
        role = "human" if msg["role"] == "user" else "ai"
        formatted_history.append((role, msg["content"]))

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    try:
        response = assistant._invoke_llm_resilient(prompt, {
            "chat_history": formatted_history,
            "input": message
        })
    except Exception as e:
        print(f"sister chat_talk error: {e}")
        if language == "bn":
            response = "আপু, সংযোগে সাময়িক সমস্যা হয়েছে। দয়া করে আবার জিজ্ঞাসা করো! 🌸"
        else:
            response = "Oops! I ran into an issue connecting. Please try asking again! 💕"

    response = safety_module.check_reply(response, rag_lang)
    return assistant._sanitize_response(response, title_used_recently)

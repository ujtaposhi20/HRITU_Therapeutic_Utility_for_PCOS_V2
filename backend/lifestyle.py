from langchain_core.prompts import ChatPromptTemplate
import safety as safety_module
import knowledge as knowledge_module


def chat_lifestyle(history, message, language="en", assistant=None):
    """
    Module 3: Lifestyle, Diet & Routine Advice.
    Two-Tier Hybrid RAG: Tier-1 local knowledge cards first, Tier-2 DuckDuckGo fallback.
    Includes sister.py empathy protocols (acknowledge first, anti-blaming, societal empathy).
    """
    if assistant is None:
        raise ValueError("assistant instance must be provided to chat_lifestyle")

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
    #RAG
    rag_lang = "bn" if language == "bn" else "en"
    knowledge_context = knowledge_module.get_card_context(message, lang=rag_lang)

    if knowledge_context:
        # Tier 1 hit: use verified local knowledge card
        search_context = knowledge_context
        print(f"📚 Tier-1 Knowledge Hit for lifestyle: {search_context[:80]}...")
    else:
        # Tier 2 fallback: live DuckDuckGo search
        from agent import run_web_search
        search_context = run_web_search(message, max_results=3)
        if search_context and search_context != "No additional web search context found.":
            print(f"🔍 Tier-2 DuckDuckGo fallback for lifestyle")
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
        "You are HRITU (in English/Banglish) or ঋতু (in Bangla), a caring, empathetic elder-sisterly AI health companion specializing in Lifestyle, Diet & Routine Advice for Women's Health and PCOS.\n"
        f"{lang_instruction}\n"
        "STRICT CONVERSATIONAL RULES & LENGTH LIMITS:\n"
        "1. MAX RESPONSE LENGTH: Keep your response strictly under 2–3 short, warm sentences (maximum ~50-70 words) for general chat turns.\n"
        "2. NO UNSOLICITED LECTURE DUMPS: NEVER output full encyclopedic overviews unless the user explicitly asks for a full guide/plan!\n"
        "3. ACKNOWLEDGE EMOTIONS FIRST (sister.py protocol): If the user expresses feeling 'bad', 'sad', 'hmm', 'tired', or overwhelmed, respond FIRST with 1 sentence of genuine comfort. Show you understood her feeling in your own words. Then offer one small comfort or idea.\n"
        "4. ANTI-BLAMING GUARDRAIL (sister.py protocol): NEVER blame the user's weight, appetite, food habits, or willpower. NEVER use dismissive phrases like 'just relax', 'don't worry', 'it's easy', or 'just eat less'.\n"
        "5. SOCIETAL EMPATHY (sister.py protocol): If the user mentions family pressure about marriage, body image, weight, or skin complexion in South Asian families, be on her side kindly and validate her feelings. Remind her she is not alone.\n"
        "6. 1-QUESTION RULE: Address the user's input warmly in 1-2 sentences, and ask at most ONE single follow-up question.\n"
        "7. VISUAL BREATHING ROOM: Use double line breaks between your empathetic statement and your single follow-up question.\n"
        "8. Address the user naturally using 'sister' (in English), 'apu' (in Banglish), or 'আপু' (in Bangla script) AT MOST ONCE per response.\n"
        "9. ALWAYS maintain an anti-prescription guardrail: never prescribe medication or medical dosages. Encourage consulting a nutritionist or gynecologist for clinical treatments.\n"
        "10. NON-NEGOTIABLE DOMAIN LOCK: You are strictly and exclusively a women's health companion. NEVER write programming code (Python, JS, etc.), solve homework or math problems, or answer general non-health questions. If asked about non-health topics, politely refuse and invite the user to discuss health, diet, or PCOS.\n"
    )

    if title_used_recently:
        system_prompt += "   - Do NOT use terms of endearment ('sister', 'apu', or 'আপু') in this response as it was used recently.\n"

    if search_context and search_context != "No additional context.":
        system_prompt += f"\nVerified Knowledge / Web Context for Accuracy:\n{search_context}\n"

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
        print(f"lifestyle chat_lifestyle error: {e}")
        if language == "bn":
            response = "আপু, সংযোগে একটি সমস্যা হয়েছে। ডায়েট ও লাইফস্টাইল সংক্রান্ত প্রশ্নটি আবার জিজ্ঞাসা করো! 🌸"
        else:
            response = "Oops! I had a connection hiccup while fetching lifestyle advice. Please try again! 💕"

    # Post-LLM medicine guardrail
    response = safety_module.check_reply(response, rag_lang)
    return assistant._sanitize_response(response, title_used_recently)

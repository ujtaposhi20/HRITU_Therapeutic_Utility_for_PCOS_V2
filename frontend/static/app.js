/* ═══════════════════════════════════════════════════════════════════════════
   HRITU — PCOS Awareness & Health Assistant Frontend Logic
   Unified Interface with Eye-Soothing UI, Compact Module Selector Grid,
   Empathetic Talk Module, and In-Chat Report Triggering
   ═══════════════════════════════════════════════════════════════════════════ */

// ── DOM Elements ──────────────────────────────────────────────────────────
const chatContainer = document.getElementById('chat-container');
const appHeaderTitle = document.getElementById('app-header-title');
const appHeaderSubtitle = document.getElementById('app-header-subtitle');
const headerNewChatBtn = document.getElementById('header-new-chat-btn');
const headerNewChatText = document.getElementById('header-new-chat-text');

// Centralized Language Toggle
const langToggleBtn = document.getElementById('landing-lang-toggle');
const langEn = document.getElementById('landing-lang-en');
const langBn = document.getElementById('landing-lang-bn');

// Settings & Controls
const voiceToggle = document.getElementById('voice-toggle');
const voiceToggleLabel = document.getElementById('voice-toggle-label');
const disclaimerBar = document.getElementById('disclaimer-bar');
const chatMessages = document.getElementById('chat-messages');
const micBtn = document.getElementById('mic-btn');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const sendLoading = document.getElementById('send-loading');
const chatInputArea = document.getElementById('chat-input-area');

// Module Selection Buttons
const m1Btn = document.getElementById('m1-btn');
const m2Btn = document.getElementById('m2-btn');
const m3Btn = document.getElementById('m3-btn');
const m4Btn = document.getElementById('m4-btn');
const moduleButtons = [m1Btn, m2Btn, m3Btn, m4Btn].filter(Boolean);

// Modals
const confirmModal = document.getElementById('confirm-modal');
const confirmModalTitle = document.getElementById('confirm-modal-title');
const confirmModalText = document.getElementById('confirm-modal-text');
const confirmYes = document.getElementById('confirm-yes');
const confirmNo = document.getElementById('confirm-no');

const earlyExitModal = document.getElementById('early-exit-modal');
const earlyModalTitle = document.getElementById('early-modal-title');
const earlyModalText = document.getElementById('early-modal-text');
const earlyKeepChatting = document.getElementById('early-keep-chatting');
const earlyEndAnyway = document.getElementById('early-end-anyway');

const resultModal = document.getElementById('result-modal');
const resultCloseBtn = document.getElementById('result-close-btn');
const resultLoading = document.getElementById('result-loading');
const resultContent = document.getElementById('result-content');
const patternText = document.getElementById('analysis-pattern');
const suggestionText = document.getElementById('analysis-suggestion');
const explainabilitySection = document.getElementById('explainability-section');
const explainabilityContent = document.getElementById('explainability-content');
const predictionBadges = document.getElementById('prediction-badges');
const rfBadge = document.getElementById('rf-badge');
const dtBadge = document.getElementById('dt-badge');
const downloadBtn = document.getElementById('download-btn');
const newChatBtn = document.getElementById('new-chat-btn');

const feedbackModal = document.getElementById('feedback-modal');
const starRating = document.getElementById('star-rating');
const starLabel = document.getElementById('star-label');
const feedbackComment = document.getElementById('feedback-comment');
const submitFeedbackBtn = document.getElementById('submit-feedback-btn');
const skipFeedbackBtn = document.getElementById('skip-feedback-btn');

const thankyouModal = document.getElementById('thankyou-modal');
const thankyouCloseBtn = document.getElementById('thankyou-close-btn');

let history = [];
let sessionId = crypto.randomUUID ? crypto.randomUUID() : _generateUUID();
let selectedRating = 0;
let chatEnded = false;
let userMessageCount = 0;
let lastResponseText = '';
let isProcessing = false;
let currentLanguage = 'en'; // 'en' | 'bn'
let currentMode = null;     // null | 'prediction' | 'talk' | 'lifestyle' | 'learn'
let currentScreeningState = null;
let currentLearnState = null;

const translations = {
    en: {
        appTitle: "HRITU",
        appSubtitle: "Health Assistant",
        newChatText: "New chat",
        voiceToggleLabel: "Read Aloud",
        disclaimerBar: "Note: This is not a medical diagnosis. Please consult a healthcare professional.",
        inputPlaceholder: "Type a message...",
        welcomePrompt: "Hi! I'm HRITU, your PCOS sister-friend. I'm an AI, not a doctor. Please do not share personal details.\n\nWhat would you like to do today? Tap one of the buttons above, or just ask me a question!",
        moduleLabels: {
            prediction: "PCOS check",
            talk: "Talk",
            lifestyle: "Lifestyle",
            learn: "Learn cycle"
        },
        moduleGreetings: {
            prediction: "Hello, sister! I am HRITU, your health awareness companion. Let's do a step-by-step check to understand your symptoms and patterns. To begin, how old are you?",
            talk: "Hello sister! I'm here for you. You can talk to me about your PCOS symptoms, general health questions, emotions, or pregnancy & fertility concerns. What's on your mind? 🌸",
            lifestyle: "Hello sister! Welcome to Lifestyle, Diet & Routine Advice. How can I help with your nutrition, workout, or sleep routine today? 🌸",
            learn: "Welcome sister! Let's learn about your 4 period cycle phases and simple self-care tips. Type 'next' or tap below to start Phase 1! 🌸"
        },
        confirmTitle: "End Conversation?",
        confirmText: "Your responses will be processed to analyze pattern observations. Proceed?",
        confirmYes: "Yes, Proceed",
        confirmNo: "Not Yet",
        earlyTitle: "More Information Needed",
        earlyText: "To generate an accurate pattern observation, mandatory core health features are required. Please continue the conversation to provide these details.",
        earlyKeep: "Keep Chatting",
        earlyEnd: "End Anyway",
        stopReportChip: "Stop & View Report",
        skipChip: "Skip",
        nextChip: "Next",
        thinking: "Thinking...",
        offline: "Connection error. Please try sending your message again."
    },
    bn: {
        appTitle: "ঋতু",
        appSubtitle: "স্বাস্থ্য সহকারী",
        newChatText: "নতুন চ্যাট",
        voiceToggleLabel: "পড়ে শোনাও",
        disclaimerBar: "নোট: এটি কোনো চিকিৎসা নিদান বা ডায়াগনোসিস নয়। সঠিক পরামর্শের জন্য চিকিৎসকের পরামর্শ নিন।",
        inputPlaceholder: "একটি বার্তা লিখুন...",
        welcomePrompt: "হাই! আমি ঋতু, তোমার PCOS বন্ধু। আমি একটি এআই, ডাক্তার নই। দয়া করে ব্যক্তিগত তথ্য দিও না।\n\nআজ কী করতে চাও? উপরের যেকোনো একটি বাটনে ট্যাপ করো, অথবা সরাসরি প্রশ্ন করো!",
        moduleLabels: {
            prediction: "PCOS পরীক্ষা",
            talk: "মনের কথা",
            lifestyle: "লাইফস্টাইল",
            learn: "মাসিক শিখি"
        },
        moduleGreetings: {
            prediction: "হ্যালো আপু! আমি ঋতু, তোমার স্বাস্থ্য সহায়িকা। চলো একটি একটি করে সহজ প্রশ্নের মাধ্যমে তোমার লক্ষণ ও স্বাস্থ্য সংক্রান্ত প্যাটার্ন মূল্যায়ন করি। শুরুতে বলো, তোমার বয়স কত?",
            talk: "হ্যালো আপু! আমি তোমার পাশেই আছি। তোমার PCOS এর লক্ষণ, যেকোনো স্বাস্থ্য ভাবনা, আবেগ বা গর্ভধারণ ও প্রজনন নিয়ে যেকোনো কথা বলতে পারো। কী জানতে চাও? 🌸",
            lifestyle: "হ্যালো আপু! জীবনধারা ও খাবার পরামর্শে স্বাগতম। ডায়েট, শরীরচর্চা বা ঘুম নিয়ে কী জানতে ইচ্ছে করছে? 🌸",
            learn: "স্বাগতম আপু! চলো সহজ ভাষায় মাসিক চক্রের ৪টি ধাপ এবং নিজের যত্নের উপায়গুলো জেনে নেই। প্রথম ধাপ দেখতে 'পরেরটা' লিখো বা নিচের বাটনে ট্যাপ করো! 🌸"
        },
        confirmTitle: "চ্যাট শেষ করতে চান?",
        confirmText: "আপনার উত্তরগুলো বিশ্লেষণ করে একটি স্বাস্থ্য পর্যবেক্ষণ তৈরি করা হবে। এগিয়ে যাবেন?",
        confirmYes: "হ্যাঁ, দেখুন",
        confirmNo: "এখনও না",
        earlyTitle: "আরও তথ্য প্রয়োজন",
        earlyText: "সঠিক প্যাটার্ন পর্যবেক্ষণের জন্য মূল স্বাস্থ্য সংক্রান্ত তথ্য প্রয়োজন। অনুগ্রহ করে আরও কিছু তথ্য দিয়ে সহযোগিতা করুন।",
        earlyKeep: "কথোপকথন চালিয়ে যান",
        earlyEnd: "তবুও শেষ করুন",
        stopReportChip: "থামো ও রিপোর্ট দেখুন",
        skipChip: "বাদ",
        nextChip: "পরেরটা",
        thinking: "ভাবছি...",
        offline: "সংযোগ বিচ্ছিন্ন হয়েছে। আবার চেষ্টা করুন।"
    }
};

function _generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Voice Synthesis (Text-to-Speech)
let isVoiceEnabled = sessionStorage.getItem('isVoiceEnabled') === 'true';

document.addEventListener('DOMContentLoaded', () => {
    // Theme Switcher Handlers
    const themeToggle = document.getElementById('theme-toggle');
    const savedTheme = localStorage.getItem('theme') || 'light';
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
    } else {
        document.body.classList.remove('dark-mode');
    }

    const toggleTheme = () => {
        const isDark = document.body.classList.toggle('dark-mode');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    };
    if (themeToggle) themeToggle.addEventListener('click', toggleTheme);
    if (voiceToggle) {
        voiceToggle.checked = isVoiceEnabled;
        voiceToggle.addEventListener('change', (e) => {
            isVoiceEnabled = e.target.checked;
            sessionStorage.setItem('isVoiceEnabled', isVoiceEnabled);
            if (!isVoiceEnabled) {
                window.speechSynthesis.cancel();
            } else if (lastResponseText) {
                speakText(lastResponseText);
            }
        });
    }
    if (langToggleBtn) langToggleBtn.addEventListener('click', toggleLanguage);

    if (headerNewChatBtn) {
        headerNewChatBtn.addEventListener('click', () => {
            resetAndStartNewChat();
        });
    }

    [
        { btn: m1Btn, mode: 'prediction' },
        { btn: m3Btn, mode: 'talk' },
        { btn: m2Btn, mode: 'lifestyle' },
        { btn: m4Btn, mode: 'learn' }
    ].forEach(({ btn, mode }) => {
        if (btn) {
            btn.addEventListener('click', () => selectModuleAndStartChat(mode));
        }
    });

    updateLanguageUI();
    showInitialWelcome();
});

function toggleLanguage() {
    currentLanguage = currentLanguage === 'en' ? 'bn' : 'en';
    updateLanguageUI();
    if (history.length <= 1) {
        showInitialWelcome();
    }
}

function updateLanguageUI() {
    const t = translations[currentLanguage];

    // Language Toggle Badges
    if (currentLanguage === 'en') {
        if (langEn) langEn.classList.add('active');
        if (langBn) langBn.classList.remove('active');
    } else {
        if (langBn) langBn.classList.add('active');
        if (langEn) langEn.classList.remove('active');
    }

    if (appHeaderTitle) appHeaderTitle.textContent = t.appTitle;
    if (appHeaderSubtitle) appHeaderSubtitle.textContent = t.appSubtitle;
    if (headerNewChatText) headerNewChatText.textContent = t.newChatText;
    if (voiceToggleLabel) voiceToggleLabel.textContent = t.voiceToggleLabel;
    if (disclaimerBar) disclaimerBar.textContent = t.disclaimerBar;
    if (messageInput && !isRecording) messageInput.placeholder = t.inputPlaceholder;
    if (micBtn && !isRecording) micBtn.title = currentLanguage === 'bn' ? 'ভয়েস ইনপুট' : 'Voice Input (Speech-to-Text)';

    // Update Module Card Labels
    document.querySelectorAll('.feat-label').forEach(el => {
        const enVal = el.getAttribute('data-en');
        const bnVal = el.getAttribute('data-bn');
        el.textContent = currentLanguage === 'bn' ? bnVal : enVal;
    });

    if (confirmModalTitle) confirmModalTitle.textContent = t.confirmTitle;
    if (confirmModalText) confirmModalText.textContent = t.confirmText;
    if (confirmYes) confirmYes.textContent = t.confirmYes;
    if (confirmNo) confirmNo.textContent = t.confirmNo;

    if (earlyModalTitle) earlyModalTitle.textContent = t.earlyTitle;
    if (earlyModalText) earlyModalText.textContent = t.earlyText;
    if (earlyKeepChatting) earlyKeepChatting.textContent = t.earlyKeep;
    if (earlyEndAnyway) earlyEndAnyway.textContent = t.earlyEnd;
}

function showInitialWelcome() {
    chatMessages.innerHTML = '';
    history = [];
    const t = translations[currentLanguage];
    const welcome = t.welcomePrompt;
    lastResponseText = welcome;
    appendMessage(welcome, 'ai', false, []);
    speakText(welcome);
}

function resetAndStartNewChat() {
    sessionId = crypto.randomUUID ? crypto.randomUUID() : _generateUUID();
    history = [];
    currentScreeningState = null;
    currentLearnState = null;
    currentMode = null;
    chatEnded = false;
    userMessageCount = 0;
    isProcessing = false;

    moduleButtons.forEach(btn => btn.classList.remove('active'));
    chatInputArea.classList.remove('disabled');
    messageInput.disabled = false;
    if (micBtn) micBtn.disabled = false;
    sendBtn.classList.remove('hidden');
    if (sendLoading) sendLoading.classList.add('hidden');
    messageInput.value = '';

    showInitialWelcome();
    messageInput.focus();
}

function selectModuleAndStartChat(mode) {
    currentMode = mode;

    moduleButtons.forEach(btn => {
        if (btn && btn.getAttribute('data-mode') === mode) {
            btn.classList.add('active');
        } else if (btn) {
            btn.classList.remove('active');
        }
    });

    const t = translations[currentLanguage];
    const greeting = t.moduleGreetings[mode];
    lastResponseText = greeting;

    let initialChips = [];
    if (mode === 'prediction') {
        initialChips = currentLanguage === 'bn'
            ? [t.skipChip, t.stopReportChip]
            : [t.skipChip, t.stopReportChip];
    } else if (mode === 'learn') {
        initialChips = [t.nextChip];
    }

    appendMessage(greeting, 'ai', false, initialChips);
    speakText(greeting);
    messageInput.focus();
}

// Voice Output (TTS)- Empathetic, Strictly Female Voice Engine
let cachedVoices = [];

function loadVoices() {
    if ('speechSynthesis' in window) {
        cachedVoices = window.speechSynthesis.getVoices() || [];
    }
}

if ('speechSynthesis' in window) {
    loadVoices();
    window.speechSynthesis.onvoiceschanged = () => {
        loadVoices();
    };
}

// Strict blacklist of male voice identifiers (names, tags, prefixes)
const MALE_VOICE_BLACKLIST = [
    'male', 'alex', 'fred', 'daniel', 'david', 'george', 'oliver', 'tom',
    'bruce', 'junior', 'ralph', 'albert', 'guy', 'man', 'boy', 'mark',
    'ravi', 'stefan', 'martin', 'richard', 'james', 'john', 'william', 'paul',
    'lee', 'bernard', 'kurt', 'yuri', 'kabir', 'deep'
];

function isMaleVoice(voice) {
    if (!voice || !voice.name) return false;
    const nameLower = voice.name.toLowerCase();
    return MALE_VOICE_BLACKLIST.some(m => {
        const regex = new RegExp(`(^|[^a-z])${m}([^a-z]|$)`, 'i');
        return regex.test(nameLower);
    });
}

function selectBestFemaleVoice(isBangla) {
    const voices = (cachedVoices && cachedVoices.length > 0) ? cachedVoices : (window.speechSynthesis ? window.speechSynthesis.getVoices() : []);
    if (!voices || voices.length === 0) return null;

    if (isBangla) {
        // 1. Look for Bangla voices that are definitely not male
        const bnVoices = voices.filter(v => 
            v.lang && (v.lang.toLowerCase().startsWith('bn') || v.lang.toLowerCase().includes('bangla')) &&
            !isMaleVoice(v)
        );
        // Preference for known Bangla female voices
        const preferredBn = bnVoices.find(v => {
            const nl = v.name.toLowerCase();
            return nl.includes('female') || nl.includes('tania') || nl.includes('bangla') || nl.includes('google');
        });
        if (preferredBn) return preferredBn;
        if (bnVoices.length > 0) return bnVoices[0];
    }

    // For English (or fallback for Bangla if no bn voice is installed on OS):
    // MUST be strictly an English voice (en-US, en-GB, en-AU, en-IN, en-CA, etc.) and NOT male!
    const englishNonMaleVoices = voices.filter(v => 
        v.lang && v.lang.toLowerCase().startsWith('en') &&
        !isMaleVoice(v)
    );

    const candidates = englishNonMaleVoices.length > 0 ? englishNonMaleVoices : voices.filter(v => !isMaleVoice(v));

    // Priority 1: High-fidelity natural female English voices
    const priority1 = [
        /\bgoogle uk english female\b/i,
        /\bgoogle us english female\b/i,
        /\bmicrosoft jenny\b/i,
        /\bmicrosoft aria\b/i,
        /\bmicrosoft michelle\b/i,
        /\bmicrosoft sonia\b/i,
        /\bgoogle us english\b/i,
        /\bnatural\b/i
    ];
    for (const pat of priority1) {
        const found = candidates.find(v => pat.test(v.name));
        if (found) return found;
    }

    // Priority 2: High quality macOS / iOS / Windows female English voices
    const priority2 = [
        /\bsamantha\b/i,
        /\bvictoria\b/i,
        /\bkaren\b/i,
        /\bmoira\b/i,
        /\btessa\b/i,
        /\bfiona\b/i,
        /\bava\b/i,
        /\ballison\b/i,
        /\bsusan\b/i,
        /\bzoe\b/i,
        /\bnicky\b/i,
        /\bserena\b/i,
        /\bzira\b/i,
        /\beva\b/i
    ];
    for (const pat of priority2) {
        const found = candidates.find(v => pat.test(v.name));
        if (found) return found;
    }

    // Priority 3: Any candidate explicitly containing 'female' in name
    const explicitFemale = candidates.find(v => /\bfemale\b/i.test(v.name));
    if (explicitFemale) return explicitFemale;

    // Priority 4: Any English non-male candidate
    if (englishNonMaleVoices.length > 0) return englishNonMaleVoices[0];

    // Priority 5: Fallback to first non-male candidate
    if (candidates.length > 0) return candidates[0];

    return null;
}

function cleanTextForSpeech(text, isBangla) {
    if (!text) return '';

    let clean = text;

    // 1. Extract <tts> content if present, or strip <tts> tags
    const ttsMatch = clean.match(/<tts[^>]*>([\s\S]*?)<\/tts>/i);
    if (ttsMatch && ttsMatch[1]) {
        clean = ttsMatch[1];
    } else {
        clean = clean.replace(/<tts[^>]*>[\s\S]*?<\/tts>/gi, '');
    }

    // 2. Strip HTML tags
    clean = clean.replace(/<[^>]+>/g, ' ');

    // 3. Strip Markdown formatting
    // Code blocks and backticks
    clean = clean.replace(/```[\s\S]*?```/g, '');
    clean = clean.replace(/`([^`]+)`/g, '$1');
    // Markdown links: [text](url) -> text
    clean = clean.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
    // Headers (#, ##, etc.)
    clean = clean.replace(/^#{1,6}\s+/gm, '');
    // Bold / italic formatting (**, *, __, _)
    clean = clean.replace(/\*\*([^*]+)\*\*/g, '$1');
    clean = clean.replace(/\*([^*]+)\*/g, '$1');
    clean = clean.replace(/__([^_]+)__/g, '$1');
    clean = clean.replace(/_([^_]+)_/g, '$1');
    clean = clean.replace(/~~([^~]+)~~/g, '$1');
    // Blockquotes (> text)
    clean = clean.replace(/^>\s+/gm, '');
    // Horizontal lines (---, ***, ___)
    clean = clean.replace(/^[\s-*_]{3,}$/gm, ' ');
    // Tables: remove pipe characters
    clean = clean.replace(/\|/g, ', ');
    // Bullet points (-, *, +, 1., 2.)
    clean = clean.replace(/^[\s]*[-*+]\s+/gm, '');
    clean = clean.replace(/^[\s]*\d+\.\s+/gm, '');

    // 4. Strip emojis and unicode decorative symbols thoroughly
    clean = clean.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{2300}-\u{23FF}\u{2B50}\u{1F495}\u{2728}\u{1F338}\u{1F49B}\u{1F496}\u{1F48C}\u{1F4AC}\u{1F9E1}\u{1F49A}\u{1F499}\u{1F49C}\u{1F49D}\u{1F49E}\u{1FA77}\u{1FA78}\u{1FA79}\u{1F97A}]/gu, '');

    // 5. Special pronunciation & natural speech cadence adjustments:
    if (isBangla) {
        clean = clean.replace(/\bHRITU\b/gi, 'ঋতু');
        clean = clean.replace(/\bH\.R\.I\.T\.U\b/gi, 'ঋতু');
        clean = clean.replace(/\bH R I T U\b/gi, 'ঋতু');
        clean = clean.replace(/\bPCOS\b/gi, 'পিসিওএস');
        clean = clean.replace(/\bBMI\b/gi, 'বিএমআই');
    } else {
        // In English, spell HRITU phonetically as "Ritu" or "Reetu" for natural female English pronunciation
        clean = clean.replace(/\bHRITU\b/gi, 'Reetu');
        clean = clean.replace(/\bH\.R\.I\.T\.U\b/gi, 'Reetu');
        clean = clean.replace(/\bH R I T U\b/gi, 'Reetu');
        // PCOS -> P. C. O. S. (avoids pronouncing it as "peecoss")
        clean = clean.replace(/\bPCOS\b/g, 'P. C. O. S.');
        clean = clean.replace(/\bBMI\b/g, 'B. M. I.');
        clean = clean.replace(/\bkg\b/gi, 'kilograms');
        clean = clean.replace(/\bcm\b/gi, 'centimeters');
        clean = clean.replace(/\bRF\b/g, 'Random Forest');
        clean = clean.replace(/\bDT\b/g, 'Decision Tree');
        clean = clean.replace(/\bSVM\b/g, 'S. V. M.');
    }

    // Remove remaining stray asterisks, hashes, brackets, etc.
    clean = clean.replace(/[*#~^`]/g, '');
    clean = clean.replace(/[()[\]{}]/g, ', ');

    // Normalize whitespace and pauses
    clean = clean.replace(/\n\s*\n/g, '. ');
    clean = clean.replace(/\n/g, ' ');
    clean = clean.replace(/\s+/g, ' ').trim();

    return clean;
}

function speakText(text) {
    if (!('speechSynthesis' in window) || !isVoiceEnabled) return;
    window.speechSynthesis.cancel();

    if (!text || typeof text !== 'string') return;

    // Detect language
    const hasBangla = /[\u0980-\u09FF]/.test(text);

    // Clean text thoroughly
    const speechContent = cleanTextForSpeech(text, hasBangla);
    if (!speechContent || speechContent.length < 2) return;

    const utterance = new SpeechSynthesisUtterance(speechContent);
    const selectedVoice = selectBestFemaleVoice(hasBangla);

    if (selectedVoice) {
        utterance.voice = selectedVoice;
        utterance.lang = selectedVoice.lang;
    } else {
        utterance.lang = hasBangla ? 'bn-BD' : 'en-US';
    }

    // Conversational, empathetic pacing (NOT robotic, NO squeaky chipmunk pitch)
    if (hasBangla) {
        utterance.rate = 0.95;
        utterance.pitch = 1.0;
    } else {
        // 0.93 rate = calm, empathetic sister cadence; 1.0 pitch = natural grounded tone
        utterance.rate = 0.93;
        utterance.pitch = 1.0;
    }

    window.speechSynthesis.speak(utterance);
}

// ── Messages & Dynamic Chips ────────────────────────────────────────────────
if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true, gfm: true });
}

function appendMessage(content, sender, isCrisis = false, chips = []) {
    const div = document.createElement('div');
    div.classList.add('message', sender);
    if (isCrisis) {
        div.classList.add('crisis');
    }

    if (sender === 'ai' && typeof marked !== 'undefined') {
        div.innerHTML = marked.parse(content);

        // Add inline PDF download button for any Markdown tables in AI response
        const tables = div.querySelectorAll('table');
        if (tables.length > 0) {
            tables.forEach((table) => {
                const btnWrapper = document.createElement('div');
                btnWrapper.className = 'table-pdf-btn-wrapper';

                const pdfBtn = document.createElement('button');
                pdfBtn.className = 'btn-download-table-pdf';
                pdfBtn.title = currentLanguage === 'bn' ? 'পিডিএফ ডাউনলোড' : 'Download PDF';
                pdfBtn.setAttribute('aria-label', pdfBtn.title);
                pdfBtn.innerHTML = `
                    <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                `;

                pdfBtn.onclick = () => downloadTableAsPDF(table);
                btnWrapper.appendChild(pdfBtn);
                table.parentNode.insertBefore(btnWrapper, table.nextSibling);
            });
        }
    } else {
        div.textContent = content;
    }

    // Inline selection chips directly inside/under the AI message bubble
    if (sender === 'ai' && Array.isArray(chips) && chips.length > 0) {
        const chipsWrapper = document.createElement('div');
        chipsWrapper.className = 'message-chips-container';
        chips.forEach(chipText => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'message-chip';
            btn.textContent = chipText;

            // Highlight Stop & View Report chip subtly
            if (chipText === 'Stop & View Report' || chipText === 'থামো ও রিপোর্ট দেখুন') {
                btn.classList.add('report-trigger-chip');
            }

            btn.addEventListener('click', () => {
                if (isProcessing || chatEnded) return;

                // Dim and disable all chips in this container to prevent duplicate taps
                chipsWrapper.querySelectorAll('.message-chip').forEach(c => {
                    c.classList.add('disabled');
                    c.disabled = true;
                });
                btn.classList.add('selected');

                // If Stop & View Report chip is clicked, trigger completeness and report flow
                if (chipText === 'Stop & View Report' || chipText === 'থামো ও রিপোর্ট দেখুন') {
                    handleStopOrEndChat();
                    return;
                }

                messageInput.value = chipText;
                sendMessage();
            });
            chipsWrapper.appendChild(btn);
        });
        div.appendChild(chipsWrapper);
    }

    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    if (sender === 'user') {
        history.push({ role: 'user', content: content });
    } else {
        history.push({ role: 'assistant', content: content });
    }
}

function downloadTableAsPDF(tableElement) {
    const reportContainer = document.createElement('div');
    reportContainer.style.fontFamily = "'Outfit', sans-serif";
    reportContainer.style.padding = '24px';
    reportContainer.style.color = '#301a3e';
    reportContainer.style.background = '#ffffff';

    const titleText = currentLanguage === 'bn' ? 'ঋতু — স্বাস্থ্য ও লাইফস্টাইল প্ল্যান' : 'HRITU — Health & Lifestyle Plan';
    const dateStr = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });

    const tableClone = tableElement.cloneNode(true);
    tableClone.style.width = '100%';
    tableClone.style.borderCollapse = 'collapse';
    tableClone.style.marginTop = '16px';
    tableClone.style.marginBottom = '16px';
    tableClone.style.fontSize = '12px';

    const cells = tableClone.querySelectorAll('th, td');
    cells.forEach(cell => {
        cell.style.border = '1px solid #ede3f6';
        cell.style.padding = '8px 12px';
        cell.style.textAlign = 'left';
    });
    const headers = tableClone.querySelectorAll('th');
    headers.forEach(h => {
        h.style.background = '#8a5ca8';
        h.style.color = '#ffffff';
    });

    reportContainer.innerHTML = `
        <div style="text-align:center; margin-bottom:20px;">
            <h1 style="color:#6e4389; font-size:18px; margin:0 0 4px 0;">${titleText}</h1>
            <p style="font-size:11px; color:#8a5ca8; margin:0;">Generated on ${dateStr}</p>
        </div>
        <div style="background:#f9f6fc; padding:16px; border-radius:12px; margin-bottom:16px; border:1px solid #ede3f6;">
            ${tableClone.outerHTML}
        </div>
        <div style="text-align:center; margin-top:20px; padding-top:12px; border-top:1px solid #ede3f6;">
            <p style="font-size:11px; color:#8a5ca8; margin:0;">
                © 2026 - HRITU TN. | Health Assistant
            </p>
        </div>
    `;

    if (typeof html2pdf !== 'undefined') {
        html2pdf().from(reportContainer).set({
            margin: 0.6,
            filename: `HRITU_Plan_${new Date().getTime()}.pdf`,
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2 },
            jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
        }).save();
    }
}

function showTyping() {
    const div = document.createElement('div');
    div.classList.add('typing');
    div.id = 'typing-indicator';
    div.innerHTML = '<span></span><span></span><span></span>';
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTyping() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

// ── Send Message ──────────────────────────────────────────────────────────
async function sendMessage() {
    if (isRecording) stopRecording();
    const text = messageInput.value.trim();
    if (!text || chatEnded || isProcessing) return;

    // Check if user manually typed stop or view report command in chat
    const normalizedText = text.toLowerCase();
    if (currentMode === 'prediction' && (
        normalizedText === 'stop' ||
        normalizedText === 'view report' ||
        normalizedText === 'stop & view report' ||
        normalizedText === 'report' ||
        normalizedText === 'থামো' ||
        normalizedText === 'রিপোর্ট দেখুন' ||
        normalizedText === 'থামো ও রিপোর্ট দেখুন'
    )) {
        appendMessage(text, 'user');
        messageInput.value = '';
        handleStopOrEndChat();
        return;
    }

    // If no module was explicitly clicked yet, default mode to 'talk'
    if (!currentMode) {
        currentMode = 'talk';
        if (m3Btn) m3Btn.classList.add('active');
    }

    isProcessing = true;
    messageInput.disabled = true;
    if (micBtn) micBtn.disabled = true;
    messageInput.placeholder = translations[currentLanguage].thinking;
    sendBtn.classList.add('hidden');
    if (sendLoading) sendLoading.classList.remove('hidden');

    appendMessage(text, 'user');
    messageInput.value = '';
    userMessageCount++;
    showTyping();

    try {
        const payload = {
            message: text,
            history: history.slice(0, -1),
            session_id: sessionId,
            message_count: userMessageCount,
            current_mode: currentMode,
            language: currentLanguage
        };
        if (currentMode === 'prediction' && currentScreeningState) {
            payload.screening_state = currentScreeningState;
        }
        if (currentMode === 'learn' && currentLearnState) {
            payload.learn_state = currentLearnState;
        }

        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        hideTyping();
        lastResponseText = data.response;
        const isCrisis = Boolean(data.is_crisis);

        if (data.screening_state) {
            currentScreeningState = data.screening_state;
        }
        if (data.learn_state) {
            currentLearnState = data.learn_state;
        }

        // Extract inline selection chips (crisis alerts have no chips)
        const chips = (data.chips && data.chips.length > 0) ? data.chips : (isCrisis ? [] : chipsFor(data.response));
        appendMessage(data.response, 'ai', isCrisis, chips);
        speakText(data.response);

        // If screening finished in Module 1:
        if (data.is_screening_complete) {
            chatEnded = true;
            chatInputArea.classList.add('disabled');
            messageInput.disabled = true;
            if (micBtn) { micBtn.disabled = true; stopRecording(); }

            if (data.pattern) {
                patternText.innerHTML = marked.parse(data.pattern);
            }
            if (data.suggestion) {
                suggestionText.textContent = data.suggestion;
            }
            if (rfBadge && data.rf_prediction) {
                rfBadge.textContent = `Random Forest: ${data.rf_prediction}`;
            }
            if (dtBadge && data.dt_prediction) {
                dtBadge.textContent = `Decision Tree: ${data.dt_prediction}`;
            }
            if (data.explainability && data.explainability.shap_factors) {
                explainabilitySection.style.display = 'block';
                let expHtml = '';
                data.explainability.shap_factors.forEach(item => {
                    const isPos = item.direction === 'positive';
                    const badgeClass = isPos ? 'pos' : 'neg';
                    const badgeText = isPos ? 'Key Indicator' : 'Balanced Factor';
                    expHtml += `
                        <div class="explain-item">
                            <span><strong>${item.feature}</strong></span>
                            <span class="explain-badge ${badgeClass}">${badgeText}</span>
                        </div>
                    `;
                });
                explainabilityContent.innerHTML = expHtml;
            }

            // Auto-display Result Modal after brief pause
            setTimeout(() => {
                resultModal.style.display = 'block';
                resultLoading.style.display = 'none';
                resultContent.style.display = 'block';
            }, 2500);
        }
    } catch (err) {
        hideTyping();
        appendMessage(translations[currentLanguage].offline, 'ai');
    } finally {
        isProcessing = false;
        messageInput.disabled = false;
        if (micBtn && !chatEnded) micBtn.disabled = false;
        messageInput.placeholder = translations[currentLanguage].inputPlaceholder;
        sendBtn.classList.remove('hidden');
        if (sendLoading) sendLoading.classList.add('hidden');
        messageInput.focus();
    }
}

// ── Dynamic Quick-Reply Chips Logic ─────────────────────────────────────────
function chipsFor(replyText) {
    if (!replyText) return [];
    const chips = [];
    const t = translations[currentLanguage];

    // 1. Numbered options e.g. "1) ... 2) ..." or "১) ... ২) ..."
    const optionMatches = replyText.match(/(?:[1-9১-৯]\))\s*([^1-9১-৯\n\r]+)/g);
    if (optionMatches && optionMatches.length >= 2) {
        optionMatches.forEach(opt => {
            const cleanOpt = opt.replace(/^[1-9১-৯]\)\s*/, '').trim();
            const label = cleanOpt.length > 25 ? cleanOpt.substring(0, 22) + '...' : cleanOpt;
            chips.push(label);
        });
        if (/skip|বাদ|জানিনা/i.test(replyText) || currentMode === 'prediction') {
            chips.push(t.skipChip);
            if (currentMode === 'prediction') {
                chips.push(t.stopReportChip);
            }
        }
        return chips;
    }

    // 2. Yes/No prompts
    if (/\(yes\/no\)/i.test(replyText) || /\(yes\s*\/\s*no\)/i.test(replyText)) {
        return ['Yes', 'No', t.skipChip, t.stopReportChip];
    }
    if (/\(হ্যাঁ\/না\)/i.test(replyText) || /\(হ্যাঁ\s*\/\s*না\)/i.test(replyText)) {
        return ['হ্যাঁ', 'না', t.skipChip, t.stopReportChip];
    }

    // 3. Milestone decision
    if (/lifestyle.*see.*results/i.test(replyText) || /evaluate your pattern/i.test(replyText)) {
        return ['Share More Details', 'See My Results Now'];
    }
    if (/জীবনযাত্রা.*ফলাফল/i.test(replyText) || /প্যাটার্ন মূল্যায়নের/i.test(replyText)) {
        return ['আরও তথ্য দিন', 'এখনই ফলাফল দেখুন'];
    }

    // 4. Confirmation review table
    if (/is this correct\?.*reply.*yes/i.test(replyText)) {
        return ['Yes, correct'];
    }
    if (/সব ঠিক থাকলে.*'হ্যাঁ' লিখো/i.test(replyText)) {
        return ['হ্যাঁ, সব ঠিক আছে'];
    }

    // 5. Next lesson (Module 4: Strictly 'Next', NEVER 'Skip')
    if (/type\s+['"]?next['"]?/i.test(replyText)) {
        return ['Next'];
    }
    if (/['"]?পরের['"]?\s*লিখো/i.test(replyText) || /পরেরটা/i.test(replyText)) {
        return ['পরেরটা'];
    }

    return chips;
}

// ── Voice-to-Text (Speech Recognition / STT) ─────────────────────────────
let recognition = null;
let isRecording = false;
const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognitionAPI) {
    recognition = new SpeechRecognitionAPI();
    recognition.continuous = false;
    recognition.interimResults = true;

    recognition.onstart = () => {
        isRecording = true;
        if (micBtn) {
            micBtn.classList.add('recording');
            micBtn.title = currentLanguage === 'bn' ? 'রেকর্ডিং বন্ধ করুন' : 'Stop Listening';
        }
        if (messageInput) {
            messageInput.placeholder = currentLanguage === 'bn' ? 'শুনছি... বলুন...' : 'Listening... Speak now...';
        }
    };

    recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        if (messageInput && transcript) {
            messageInput.value = transcript;
        }
    };

    recognition.onerror = (event) => {
        console.warn('Speech recognition error:', event.error);
        stopRecording();
    };

    recognition.onend = () => {
        stopRecording();
    };
}

function startRecording() {
    if (!recognition) {
        alert(currentLanguage === 'bn'
            ? 'আপনার ব্রাউজারে ভয়েস-টু-টেক্সট সমর্থন করে না। অনুগ্রহ করে Google Chrome বা Edge ব্যবহার করুন।'
            : 'Voice-to-Text is not supported in this browser. Please try Google Chrome or Edge.'
        );
        return;
    }
    try {
        recognition.lang = currentLanguage === 'bn' ? 'bn-BD' : 'en-US';
        recognition.start();
    } catch (err) {
        console.warn('Recognition start error:', err);
        stopRecording();
    }
}

function stopRecording() {
    isRecording = false;
    if (micBtn) {
        micBtn.classList.remove('recording');
        micBtn.title = currentLanguage === 'bn' ? 'ভয়েস ইনপুট' : 'Voice Input (Speech-to-Text)';
    }
    if (messageInput) {
        messageInput.placeholder = translations[currentLanguage].inputPlaceholder;
        messageInput.focus();
    }
    if (recognition) {
        try {
            recognition.stop();
        } catch (e) {}
    }
}

if (micBtn) {
    micBtn.addEventListener('click', () => {
        if (isProcessing || chatEnded) return;
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });
}

sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// ── In-Chat Stop & End Chat Triggering Flow ───────────────────────────────
async function handleStopOrEndChat() {
    if (chatEnded) return;
    if (isRecording) stopRecording();

    // Modules 2, 3, 4: Friendly closing message and feedback modal
    if (currentMode !== 'prediction') {
        chatEnded = true;
        chatInputArea.classList.add('disabled');
        messageInput.disabled = true;
        if (micBtn) micBtn.disabled = true;

        const closingMsg = currentLanguage === 'bn'
            ? "আজকের মতো চ্যাট সম্পন্ন করার জন্য ধন্যবাদ! ভালো থেকো এবং নিজের স্বাস্থ্যের যত্ন নিও। 🌸"
            : "Thank you for chatting with me today! Stay healthy and take good care of yourself. 🌸";

        appendMessage(closingMsg, 'ai');
        speakText(closingMsg);

        setTimeout(() => {
            feedbackModal.style.display = 'block';
        }, 800);
        return;
    }

    // Module 1 (Prediction Assessment): Run completeness check
    if (history.length < 2) {
        appendMessage(currentLanguage === 'bn' ? "রিপোর্ট দেখার পূর্বে আরও কিছু প্রশ্নের উত্তর দিন।" : "Please answer a few screening questions before viewing the report.", 'ai');
        return;
    }

    showTyping();

    try {
        const compPayload = {
            history: history,
            session_id: sessionId
        };
        if (currentMode === 'prediction' && currentScreeningState) {
            compPayload.screening_state = currentScreeningState;
        }

        const response = await fetch('/check-completeness', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(compPayload)
        });
        const data = await response.json();
        hideTyping();

        if (data.missing_percentage > 0) {
            earlyExitModal.style.display = 'block';
        } else {
            confirmModal.style.display = 'block';
        }
    } catch (err) {
        hideTyping();
        confirmModal.style.display = 'block';
    }
}

earlyKeepChatting.addEventListener('click', () => {
    earlyExitModal.style.display = 'none';
    messageInput.focus();
});

earlyEndAnyway.addEventListener('click', () => {
    earlyExitModal.style.display = 'none';
    executeEndChat();
});

confirmNo.addEventListener('click', () => {
    confirmModal.style.display = 'none';
    messageInput.focus();
});

confirmYes.addEventListener('click', () => {
    confirmModal.style.display = 'none';
    executeEndChat();
});

async function executeEndChat() {
    chatEnded = true;
    chatInputArea.classList.add('disabled');
    messageInput.disabled = true;
    if (micBtn) micBtn.disabled = true;

    appendMessage(currentLanguage === 'bn' ? "তথ্য বিশ্লেষণের কাজ চলছে..." : "Processing conversation metrics and pattern analysis...", 'ai');

    await new Promise(resolve => setTimeout(resolve, 800));

    try {
        const endPayload = {
            history: history,
            session_id: sessionId
        };
        if (currentMode === 'prediction' && currentScreeningState) {
            endPayload.screening_state = currentScreeningState;
        }

        const response = await fetch('/end-chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(endPayload)
        });

        const data = await response.json();
        if (data.session_id) sessionId = data.session_id;

        if (data.insufficient_data) {
            resultModal.style.display = 'none';
            const insufficientMsg = currentLanguage === 'bn'
                ? "প্রয়োজনীয় তথ্যের ঘাটতি থাকায় প্রেডিকশন সম্পন্ন করা সম্ভব হয়নি। প্রেডিকশন মডেলের জন্য অন্তত ১০টি মূল স্বাস্থ্য সংক্রান্ত তথ্য প্রদান করা আবশ্যক।"
                : "Prediction cancelled due to insufficient health details. At least 10 mandatory features are required to run the prediction models.";
            appendMessage(insufficientMsg, 'ai');
            speakText(insufficientMsg);
            setTimeout(() => {
                feedbackModal.style.display = 'block';
            }, 800);
            return;
        }

        resultModal.style.display = 'block';
        resultLoading.style.display = 'none';
        resultContent.style.display = 'block';

        if (typeof marked !== 'undefined') {
            patternText.innerHTML = marked.parse(data.pattern);
            suggestionText.innerHTML = marked.parse(data.suggestion);
        } else {
            patternText.textContent = data.pattern;
            suggestionText.textContent = data.suggestion;
        }

        // Render SHAP & LIME Explainability Factors
        if (data.explainability && data.explainability.shap_factors) {
            explainabilitySection.style.display = 'block';
            let expHtml = '';
            data.explainability.shap_factors.forEach(item => {
                const isPos = item.direction === 'positive';
                const badgeClass = isPos ? 'pos' : 'neg';
                const badgeText = isPos ? 'Key Indicator' : 'Balanced Factor';
                expHtml += `
                    <div class="explain-item">
                        <span><strong>${item.feature}</strong></span>
                        <span class="explain-badge ${badgeClass}">${badgeText}</span>
                    </div>
                `;
            });
            explainabilityContent.innerHTML = expHtml;
        } else {
            explainabilitySection.style.display = 'none';
        }

        if (data.rf_prediction && data.dt_prediction) {
            rfBadge.textContent = `RF: ${data.rf_prediction}`;
            dtBadge.textContent = `DT: ${data.dt_prediction}`;
            predictionBadges.style.display = 'flex';
        }

    } catch (err) {
        resultLoading.style.display = 'none';
        resultContent.style.display = 'block';
        patternText.textContent = "Error performing pattern analysis. Please try again.";
        suggestionText.textContent = "";
    }
}

resultCloseBtn.addEventListener('click', () => {
    resultModal.style.display = 'none';
    setTimeout(() => {
        feedbackModal.style.display = 'block';
    }, 400);
});

// ── Star Rating & Feedback ──────────────────────────────────────────────────
const stars = starRating.querySelectorAll('.star');
const starLabels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'];

stars.forEach((star) => {
    star.addEventListener('click', () => {
        selectedRating = parseInt(star.getAttribute('data-value'));
        updateStars(selectedRating);
        submitFeedbackBtn.disabled = false;
        starLabel.textContent = starLabels[selectedRating - 1];
    });

    star.addEventListener('mouseenter', () => {
        const val = parseInt(star.getAttribute('data-value'));
        highlightStars(val);
    });

    star.addEventListener('mouseleave', () => {
        highlightStars(0);
        updateStars(selectedRating);
    });
});

function highlightStars(count) {
    stars.forEach((s) => {
        const v = parseInt(s.getAttribute('data-value'));
        s.classList.toggle('hover', v <= count);
    });
}

function updateStars(count) {
    stars.forEach((s) => {
        const v = parseInt(s.getAttribute('data-value'));
        s.classList.toggle('active', v <= count);
        s.classList.remove('hover');
    });
}

submitFeedbackBtn.addEventListener('click', async () => {
    if (selectedRating < 1) return;
    submitFeedbackBtn.disabled = true;
    submitFeedbackBtn.textContent = 'Sending...';

    try {
        await fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                rating: selectedRating,
                comment: feedbackComment.value.trim()
            })
        });
    } catch (err) {
        console.error('Feedback send error:', err);
    }

    feedbackModal.style.display = 'none';
    thankyouModal.style.display = 'block';
    setTimeout(() => {
        thankyouModal.style.display = 'none';
        resetAndStartNewChat();
    }, 2000);
});

skipFeedbackBtn.addEventListener('click', () => {
    feedbackModal.style.display = 'none';
    resetAndStartNewChat();
});

thankyouCloseBtn.addEventListener('click', () => {
    thankyouModal.style.display = 'none';
    resetAndStartNewChat();
});

// ── Download Report ───────────────────────────────────────────────────────
downloadBtn.addEventListener('click', () => {
    const report = document.createElement('div');
    report.style.fontFamily = "'Outfit', sans-serif";
    report.style.padding = '20px';
    report.style.color = '#301a3e';

    report.innerHTML = `
        <div style="text-align:center; margin-bottom:24px;">
            <h1 style="color:#6e4389; font-size:20px;">${currentLanguage === 'bn' ? 'ঋতু — PCOS প্যাটার্ন পর্যবেক্ষণ' : 'HRITU — PCOS Pattern Summary'}</h1>
            <p style="font-size:11px; color:#8a5ca8;">Generated on ${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</p>
        </div>
        <div style="background:#f9f6fc; padding:16px; border-radius:12px; margin-bottom:16px; border:1px solid #ede3f6;">
            <h3 style="color:#6e4389; margin-bottom:8px;">${currentLanguage === 'bn' ? 'পর্যবেক্ষণ' : 'Pattern Observation'}</h3>
            <p style="font-size:13px; line-height:1.6;">${patternText.textContent}</p>
        </div>
        <div style="background:#ede3f6; padding:14px; border-radius:12px; margin-bottom:16px;">
            <p style="font-size:12px; font-style:italic; line-height:1.5;">${suggestionText.textContent}</p>
        </div>
        <div style="text-align:center; margin-top:20px; padding-top:16px; border-top:1px solid #ede3f6;">
            <p style="font-size:11px; color:#6b557a;">
                <strong>Disclaimer:</strong> This report is NOT a medical diagnosis.<br>
                It is a pattern-matching observation for awareness purposes only.<br>
                Please consult a gynecologist or endocrinologist for proper evaluation.<br><br>
                © 2026 - HRITU TN. | Health Assistant
            </p>
        </div>
    `;

    html2pdf().from(report).set({
        margin: 0.8,
        filename: 'PCOS_Pattern_Summary.pdf',
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2 },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    }).save();
});

window.onclick = function (event) {
    if (event.target === confirmModal) confirmModal.style.display = 'none';
    if (event.target === earlyExitModal) earlyExitModal.style.display = 'none';
};

newChatBtn.addEventListener('click', () => {
    resultModal.style.display = 'none';
    resetAndStartNewChat();
});

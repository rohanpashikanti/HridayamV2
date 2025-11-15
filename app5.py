# app.py
import os
import time
import requests
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, timedelta
import json
import re

app = Flask(__name__)

# -------------------------------------------------
#           Gemini (RAG) + MongoDB Setup
# -------------------------------------------------
# --- Gemini API config ---
# Prefer environment variables in production; fallback to literals for local/testing convenience.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCelsbBGkMcws5qSv49Qt4g4xjNwp9xof8")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# --- MongoDB config ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://rohanpash:rohanpash@cluster0.p7d0opf.mongodb.net/")
MONGO_DBNAME = os.getenv("MONGO_DBNAME", "hridhayam")
# collections: datasets, user_data, user_chats, quiz_details
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_db = mongo_client[MONGO_DBNAME]
    datasets_col = mongo_db.get_collection("datasets")
    user_data_col = mongo_db.get_collection("user_data")
    user_chats_col = mongo_db.get_collection("user_chats")
    quiz_details_col = mongo_db.get_collection("quiz_details")
    therapy_insights_col = mongo_db.get_collection("therapy_insights")
    # quick ping to ensure connection - will raise exception if unreachable
    mongo_client.admin.command("ping")
    # indexes for performance
    try:
        user_chats_col.create_index([("userid", 1), ("timestamp", -1)])
        therapy_insights_col.create_index([("userid", 1)], unique=True)
    except Exception as e:
        print(f"[MongoDB] Index creation warning: {e}")
    print("[MongoDB] Connected successfully.")
except Exception as e:
    print(f"[MongoDB] Connection failed: {e}")
    datasets_col = None
    user_data_col = None
    user_chats_col = None
    quiz_details_col = None
    therapy_insights_col = None

def query_gemini_api(prompt: str, max_output_tokens: int = 512, temperature: float = 0.2):
    """
    Query the gemini-2.0-flash:generateContent endpoint.
    Uses the 'contents' payload shape (text pieces). Adjust if your API expects different schema.
    Returns the model text (or entire JSON string if parsing fails).
    """
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens
        }
    }

    try:
        resp = requests.post(GEMINI_ENDPOINT, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Parse Gemini API response format
        text = None
        if isinstance(data, dict):
            # Standard Gemini response format: data['candidates'][0]['content']['parts'][0]['text']
            if "candidates" in data and isinstance(data["candidates"], list) and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if isinstance(candidate, dict) and "content" in candidate:
                    content = candidate["content"]
                    if isinstance(content, dict) and "parts" in content:
                        parts = content["parts"]
                        if isinstance(parts, list) and len(parts) > 0:
                            # join parts if multiple
                            text = " ".join(p.get("text", "") for p in parts)
        
        # Fallback: try other common fields
        if not text:
            if isinstance(data, dict):
                # try 'output' or 'content' top-level
                top = data.get("output") or data.get("content")
                if isinstance(top, str):
                    text = top
                elif isinstance(top, dict) and "parts" in top:
                    text = " ".join(p.get("text", "") for p in top.get("parts", []))
                elif isinstance(top, list):
                    text = " ".join([str(x) for x in top])
        # Fallback: stringify full JSON if parsing still fails
        if not text:
            text = str(data)
        return text
    except requests.exceptions.RequestException as e:
        print(f"[Gemini] Request failed: {e}")
        return None

# -------------------------------------------------
# Simple helper: enforce short replies (trim to N sentences)
# -------------------------------------------------
def trim_to_sentences(text: str, max_sentences: int = 2):
    """
    Naive sentence splitter that respects common sentence-ending punctuation.
    Returns up to max_sentences worth of text joined.
    """
    if not text:
        return text
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(parts) <= max_sentences:
        return text.strip()
    return " ".join(parts[:max_sentences]).strip()

# -------------------------------------------------
# Retrieval helpers (RAG)
# -------------------------------------------------
def retrieve_user_chats_context(userid: str, limit: int = 5):
    """
    Fetch most recent user chats for a given userid.
    Returns list of strings (message + response) in reverse chronological order (most recent first).
    Excludes deleted chats.
    """
    if user_chats_col is None or not userid:
        return []
    try:
        cursor = user_chats_col.find({
            "userid": userid,
            "is_deleted": {"$ne": True}
        }).sort("timestamp", -1).limit(limit)
        snippets = []
        for doc in cursor:
            msg = doc.get("message", "")
            resp = doc.get("response", "")
            combined = f"User: {msg}\nAssistant: {resp}"
            snippets.append(combined)
        return snippets
    except Exception as e:
        print(f"[MongoDB] Error retrieving user chats: {e}")
        return []

def retrieve_datasets_context(query_text: str, limit: int = 3):
    """
    Retrieve relevant dataset snippets from 'datasets' collection.
    Tries a text index search first; falls back to regex search on 'content' field.
    Each document is expected to have a 'content' field (adjust if you use a different key).
    Returns list of strings.
    """
    if datasets_col is None or not query_text:
        return []
    snippets = []
    try:
        try:
            cursor = datasets_col.find({"$text": {"$search": query_text}}, {"score": {"$meta": "textScore"}, "content": 1}).sort([("score", {"$meta": "textScore"})]).limit(limit)
            for doc in cursor:
                content = doc.get("content") or doc.get("text") or str(doc)
                snippets.append(content if isinstance(content, str) else str(content))
        except Exception:
            regex = {"content": {"$regex": query_text, "$options": "i"}}
            cursor = datasets_col.find(regex).limit(limit)
            for doc in cursor:
                content = doc.get("content") or doc.get("text") or str(doc)
                snippets.append(content if isinstance(content, str) else str(content))
    except Exception as e:
        print(f"[MongoDB] Error retrieving datasets: {e}")
    return snippets

# -------------------------------------------------
# Simple analysis over previous chats for therapist personalization
# -------------------------------------------------
def analyze_user_patterns(user_chats_snippets):
    """
    Analyze user's chat history to identify patterns, concerns, and progress.
    """
    if not user_chats_snippets:
        return "No previous conversation history available."
    
    # Extract key themes and emotions from chat history
    themes = []
    
    for snippet in user_chats_snippets:
        snippet_lower = snippet.lower()
        if any(word in snippet_lower for word in ['anxiety', 'worried', 'nervous', 'panic']):
            themes.append('anxiety')
        if any(word in snippet_lower for word in ['depressed', 'sad', 'down', 'hopeless']):
            themes.append('depression')
        if any(word in snippet_lower for word in ['stress', 'overwhelmed', 'pressure']):
            themes.append('stress')
        if any(word in snippet_lower for word in ['sleep', 'insomnia', 'tired']):
            themes.append('sleep issues')
        if any(word in snippet_lower for word in ['relationship', 'family', 'friend', 'social']):
            themes.append('relationships')
        if any(word in snippet_lower for word in ['work', 'job', 'career']):
            themes.append('work/career')
    
    analysis = "User's previous conversation themes: " + ", ".join(set(themes)) if themes else "No specific themes identified"
    return analysis

# -------------------------------------------------
# Conversational RAG prompt builder (listen-first + summarization mode)
# -------------------------------------------------
def build_conversational_rag_prompt(instruction: str,
                                    user_chats_snippets,
                                    datasets_snippets,
                                    user_message: str,
                                    analysis: dict = None,
                                    therapist_profile: dict = None,
                                    summarize_mode: bool = False,
                                    stress_fear_mode: bool = False):
    """
    Build a RAG prompt optimized for a conversational therapist reply.
    If summarize_mode == False: ask model to reflect and ask 2 gentle questions, no unsolicited exercises.
    If summarize_mode == True: ask model to (after a short reflection) produce a concise summary + 3 simple steps.
    """
    parts = []

    if stress_fear_mode and not summarize_mode:
        # Micro-CBT mode for stress/fear: 4 numbered steps, very concise
        conv_instr = (
            "You are a warm CBT helper. The user is sharing stress or fear."
            " Respond with EXACTLY the following 4 compact, numbered lines (1-2 short clauses each):\n"
            "1) Identify the likely negative thought (quote it briefly).\n"
            "2) Question it gently (one short Socratic question).\n"
            "3) Reframe it with logic and kindness (one short sentence).\n"
            "4) Suggest one tiny next step for relief (one short action).\n"
            "Optionally add one final short question on a new line starting with 'Q:' to check preference."
            " Keep language empathetic, non-judgmental, and specific to the user's message. Do not add any extra lines."
            "\n\n"
        )
        parts.append(conv_instr)
    elif not summarize_mode:
        conv_instr = (
            "You are a warm, conversational Cognitive Behavioral Therapist (CBT)."
            " IMPORTANT: Keep responses extremely short and focused. Reply in at most 1-2 sentences, "
            "then ask exactly 1 short open-ended question (one line). Do NOT add exercises or long explanations."
            "\n\n"
            "Your first priority is to build rapport and understand the user's experience. "
            "When replying, strictly follow these rules:\n"
            "1) Start with a 1-2 sentence empathic reflection of what the user said (validate feelings).\n"
            "2) Then ask 1 open-ended, gentle question that explores feelings or context.\n"
            "3) Do NOT give exercises, coping steps, or behavioural suggestions at this stage unless the user explicitly asks for them.\n"
            "4) Keep tone warm, curious, and non-judgemental. Keep initial reply concise (1-2 sentences plus the 1 question).\n"
            "5) Use the conversation history and analysis flags to personalize the single question if appropriate.\n"
            "6) If the analysis indicates 'suicidal_ideation', do NOT produce a normal reply here; indicate crisis instead.\n"
            "\n"
        )
        parts.append(conv_instr)
    else:
        # Summarization / action mode: used after several conversational turns
        summ_instr = (
            "You are a warm, experienced Cognitive Behavioral Therapist. IMPORTANT: Keep output concise — "
            "summary + steps must fit in 3 short sentences total (max). Use numbered steps but keep each step one short sentence."
            "\n\n"
            "You should now: \n"
            "A) Start with a 1-2 sentence empathic reflection (validate & normalize feelings).\n"
            "B) Provide a concise (1 short paragraph) summary that captures the core problem and common maintaining thoughts/behaviors.\n"
            "C) Offer one short hope/encouragement sentence.\n"
            "D) Provide up to 3 very specific, achievable steps the user can try in the next 24 hours (each step one short sentence, numbered).\n"
            "E) End with one open invitation question that checks user's preference.\n"
            "Important: keep language validating, non-judgmental, and empowering. Do NOT give long psychoeducation or many exercises—keep it focused and doable.\n"
            "\n"
        )
        parts.append(summ_instr)

    # Attach context snippets (RAG)
    if user_chats_snippets:
        parts.append("Recent user chat snippets (most recent first):\n" + "\n\n---\n".join(user_chats_snippets[:8]))
    if datasets_snippets:
        parts.append("Relevant knowledge base snippets:\n" + "\n\n---\n".join(datasets_snippets[:3]))

    # Provide analysis for personalization
    if analysis:
        parts.append(f"Structured analysis (flags/confidence/summary): {analysis}")

    # Add therapist profile (if any)
    if therapist_profile:
        parts.append(f"Therapist profile (use to tailor tone): {therapist_profile}")

    # User message and final instruction to respond as described
    parts.append(f"User message:\n{user_message}")
    parts.append("\nNow produce a single reply following the rules above. Keep it concise and conversational.")

    return "\n\n".join(parts)

# -------------------------------------------------
# New: Analysis helpers & listen-first flagging logic
# -------------------------------------------------
def analyze_message_with_gemini(message: str, userid: str = None):
    """
    Ask Gemini for a short, structured analysis.
    Expected output: JSON with keys 'flags', 'confidence', 'short_summary'.
    Attempts to extract JSON from Gemini response; if fails, returns None.
    """
    analysis_instruction = (
        "You are an assistant that ONLY analyzes the user's emotional state and behavior. "
        "Do NOT give advice in this step. Read the user message and return a single JSON object "
        "with three keys: 'flags' (a dict of booleans for anger, sadness, rumination, anxiety, "
        "withdrawal, suicidal_ideation), 'confidence' (0.0-1.0), and 'short_summary' (one sentence). "
        "Example output strictly as JSON (no extra text):\n"
        '{\n'
        '  "flags": { "anger": true, "sadness": false, "rumination": true, "anxiety": false, "withdrawal": false, "suicidal_ideation": false },\n'
        '  "confidence": 0.85,\n'
        '  "short_summary": "User is angry and ruminating about friends not valuing them."\n'
        '}\n\n'
        "Now analyze the following message:\n\n"
    )
    prompt = analysis_instruction + message

    raw = query_gemini_api(prompt, max_output_tokens=200, temperature=0.0)
    if not raw:
        return None

    # Try to extract JSON blob(s) from the response
    json_blob = None
    try:
        text = str(raw)
        brace_starts = [m.start() for m in re.finditer(r'\{', text)]
        brace_ends = [m.start() for m in re.finditer(r'\}', text)]
        for s in brace_starts:
            for e in brace_ends:
                if e <= s:
                    continue
                candidate = text[s:e+1]
                try:
                    parsed = json.loads(candidate)
                    json_blob = parsed
                    break
                except Exception:
                    continue
            if json_blob:
                break
    except Exception:
        json_blob = None

    if json_blob:
        flags = json_blob.get("flags", {})
        confidence = float(json_blob.get("confidence", 0.0)) if json_blob.get("confidence", None) is not None else 0.0
        short_summary = json_blob.get("short_summary", "")
        return {
            "flags": {
                "anger": bool(flags.get("anger", False)),
                "sadness": bool(flags.get("sadness", False)),
                "rumination": bool(flags.get("rumination", False)),
                "anxiety": bool(flags.get("anxiety", False)),
                "withdrawal": bool(flags.get("withdrawal", False)),
                "suicidal_ideation": bool(flags.get("suicidal_ideation", False))
            },
            "confidence": confidence,
            "short_summary": short_summary
        }
    return None

def local_emotion_heuristics(message: str):
    """
    Fallback heuristic emotion detection. Keyword-based.
    Returns a dict with flags, confidence, short_summary.
    """
    text = message.lower()
    flags = {
        "anger": False,
        "sadness": False,
        "rumination": False,
        "anxiety": False,
        "withdrawal": False,
        "suicidal_ideation": False
    }

    anger_kw = ["angry", "furious", "mad", "hate", "pissed", "irritat"]
    sadness_kw = ["sad", "depressed", "unhappy", "tears", "cry"]
    rumination_kw = ["can't forget", "cant forget", "keep thinking", "replay", "won't let go", "wont let go", "can't stop thinking", "cant stop thinking", "ruminate"]
    anxiety_kw = ["anxious", "panic", "nervous", "tensed", "tension", "worry", "worried"]
    withdrawal_kw = ["alone", "withdraw", "isolat", "don't want to talk", "dont want to talk", "avoid"]
    suicidal_kw = ["suicide", "kill myself", "end my life", "want to die", "i wana die", "i want to die"]

    if any(k in text for k in anger_kw):
        flags["anger"] = True
    if any(k in text for k in sadness_kw):
        flags["sadness"] = True
    if any(k in text for k in rumination_kw):
        flags["rumination"] = True
    if any(k in text for k in anxiety_kw):
        flags["anxiety"] = True
    if any(k in text for k in withdrawal_kw):
        flags["withdrawal"] = True
    if any(k in text for k in suicidal_kw):
        flags["suicidal_ideation"] = True

    positive_counts = sum(1 for v in flags.values() if v)
    confidence = min(0.6 + 0.12 * positive_counts, 0.95)
    summary_items = [k for k, v in flags.items() if v]
    short_summary = "User shows: " + (", ".join(summary_items) if summary_items else "no strong signals") + "."

    return {"flags": flags, "confidence": confidence, "short_summary": short_summary}

# -------------------------------------------------
# Utility: decide when to switch to summarization/action mode
# -------------------------------------------------
def should_summarize_for_user(userid: str, threshold: int = 3):
    """
    Return True when user has had >= threshold prior messages (so assistant should switch from exploratory Qs
    to a summarizing, hope-building, actionable reply).
    """
    if not userid or user_chats_col is None:
        return False
    try:
        # count user's messages saved in user_chats (excluding deleted)
        count = user_chats_col.count_documents({
            "userid": userid,
            "is_deleted": {"$ne": True}
        })
        return count >= threshold
    except Exception as e:
        print(f"[MongoDB] Error counting user chats: {e}")
        return False

# -------------------------------------------------
#            User Data & Quiz Management Functions
# (unchanged)
# -------------------------------------------------
def save_user_data(userid: str, username: str, email: str = None, firebase_uid: str = None):
    if user_data_col is None or not userid or not username:
        return False
    try:
        user_doc = {
            "userid": userid,
            "username": username,
            "email": email,
            "firebase_uid": firebase_uid,
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow()
        }
        user_data_col.update_one(
            {"userid": userid},
            {"$set": user_doc},
            upsert=True
        )
        return True
    except Exception as e:
        print(f"[MongoDB] Error saving user data: {e}")
        return False

def get_user_data(userid: str):
    if user_data_col is None or not userid:
        return None
    try:
        user_doc = user_data_col.find_one({"userid": userid})
        return user_doc
    except Exception as e:
        print(f"[MongoDB] Error retrieving user data: {e}")
        return None

def get_user_by_firebase_uid(firebase_uid: str):
    if user_data_col is None or not firebase_uid:
        return None
    try:
        user_doc = user_data_col.find_one({"firebase_uid": firebase_uid})
        return user_doc
    except Exception as e:
        print(f"[MongoDB] Error retrieving user data by Firebase UID: {e}")
        return None

def save_quiz_results(userid: str, responses: list, category_scores: dict, category_percentages: dict):
    if quiz_details_col is None or not userid or not responses:
        return False
    try:
        quiz_doc = {
            "userid": userid,
            "responses": responses,
            "category_scores": category_scores,
            "category_percentages": category_percentages,
            "total_questions": len(responses),
            "completed_at": datetime.utcnow()
        }
        quiz_details_col.insert_one(quiz_doc)
        return True
    except Exception as e:
        print(f"[MongoDB] Error saving quiz results: {e}")
        return False

def get_user_quiz_history(userid: str, limit: int = 10):
    if quiz_details_col is None or not userid:
        return []
    try:
        cursor = quiz_details_col.find({"userid": userid}).sort("completed_at", -1).limit(limit)
        return list(cursor)
    except Exception as e:
        print(f"[MongoDB] Error retrieving quiz history: {e}")
        return []

def analyze_quiz_responses_for_therapist_profile(userid: str):
    if quiz_details_col is None or not userid:
        return None
    try:
        latest_quiz = quiz_details_col.find_one({"userid": userid}, sort=[("completed_at", -1)])
        if not latest_quiz:
            return None
        category_percentages = latest_quiz.get("category_percentages", {})
        responses = latest_quiz.get("responses", [])
        mood_score = category_percentages.get("Mood", 50)
        relationships_score = category_percentages.get("Relationships", 50)
        habits_score = category_percentages.get("Habits", 50)
        focus_score = category_percentages.get("Focus", 50)
        therapist_profile = {
            "communication_style": "empathetic",
            "tone": "warm",
            "approach": "supportive",
            "focus_areas": [],
            "special_techniques": [],
            "response_length": "concise"
        }
        if mood_score < 30:
            therapist_profile["communication_style"] = "gentle"
            therapist_profile["tone"] = "very supportive"
            therapist_profile["special_techniques"].append("validation")
        elif mood_score > 70:
            therapist_profile["communication_style"] = "encouraging"
            therapist_profile["tone"] = "motivational"
            therapist_profile["special_techniques"].append("goal-setting")
        if relationships_score < 30:
            therapist_profile["focus_areas"].append("social connections")
            therapist_profile["special_techniques"].append("social skills")
        elif relationships_score > 70:
            therapist_profile["focus_areas"].append("maintaining healthy boundaries")
        if habits_score < 30:
            therapist_profile["focus_areas"].append("routine building")
            therapist_profile["special_techniques"].append("habit formation")
        elif habits_score > 70:
            therapist_profile["focus_areas"].append("optimizing existing habits")
        if focus_score < 30:
            therapist_profile["focus_areas"].append("concentration techniques")
            therapist_profile["special_techniques"].append("mindfulness")
        elif focus_score > 70:
            therapist_profile["focus_areas"].append("advanced productivity")
        anxiety_indicators = 0
        depression_indicators = 0
        stress_indicators = 0
        for response in responses:
            response_lower = response.lower()
            if any(word in response_lower for word in ['anxiety', 'worried', 'nervous', 'panic']):
                anxiety_indicators += 1
            if any(word in response_lower for word in ['depressed', 'sad', 'down', 'hopeless']):
                depression_indicators += 1
            if any(word in response_lower for word in ['stress', 'overwhelmed', 'pressure']):
                stress_indicators += 1
        if anxiety_indicators > 2:
            therapist_profile["special_techniques"].append("breathing exercises")
            therapist_profile["special_techniques"].append("grounding techniques")
        if depression_indicators > 2:
            therapist_profile["special_techniques"].append("behavioral activation")
            therapist_profile["special_techniques"].append("positive reframing")
        if stress_indicators > 2:
            therapist_profile["special_techniques"].append("stress management")
            therapist_profile["special_techniques"].append("time management")
        return therapist_profile
    except Exception as e:
        print(f"[MongoDB] Error analyzing quiz responses: {e}")
        return None

# -------------------------------------------------
#            CBT Quiz Data & Logic (unchanged)
# -------------------------------------------------
questions = {
    "How often have you felt down, depressed, or hopeless in the last two weeks?":
        ['Rarely or none of the time', 'Some of the time', 'Often', 'Nearly every day'],
    "Over the last two weeks, how often have you felt nervous, anxious, or on edge?":
        ['Not at all', 'Several days', 'More than half the days', 'Nearly every day'],
    "How comfortable do you feel in social situations?":
        ['Very comfortable', 'Somewhat comfortable', 'Uncomfortable', 'Very uncomfortable'],
    "How would you describe your recent sleep patterns?":
        ['Regular and restful', 'Occasionally restless', 'Frequently interrupted', 'Poor and unsatisfying'],
    "Have you experienced any physical symptoms of stress or anxiety, such as headaches, stomach upset, or rapid heartbeat?":
        ['No symptoms', 'Mild symptoms', 'Moderate symptoms', 'Severe symptoms'],
    "When you feel overwhelmed, which of the following coping strategies are you most likely to use?":
        ['Physical activity', 'Talking to friends/family', 'Avoiding the situation', 'Substance use'],
    "How do you generally perceive yourself?":
        ['Mostly positive', 'Mixed feelings', 'Mostly negative', 'I struggle with self-perception'],
    "Overall, how satisfied are you with your life?":
        ['Very satisfied', 'Somewhat satisfied', 'Not very satisfied', 'Not satisfied at all'],
    "Have you noticed a change in your interest or pleasure in doing things?":
        ['No change', 'Slight change', 'Noticeable change', 'Significant change'],
    "How would you rate your relationships with family, friends, and coworkers?":
        ['Very healthy', 'Generally healthy', 'Strained', 'Very strained'],
    "How do you usually handle stress in your life?":
        ['I excel at managing stress', 'I generally manage well', 'I manage it with some difficulty', 'I find it overwhelming and hard to manage'],
    "How often do you feel confident about your abilities?":
        ['Almost always', 'Often', 'Sometimes', 'Almost never'],
    "Do you feel you have enough emotional support from others?":
        ['I always have the support I need', 'I usually have enough support', 'I have some support, but I need more', 'I don’t have enough support'],
    "How difficult is it for you to make decisions?":
        ['Not difficult at all', 'Not very difficult', 'Somewhat difficult', 'Very difficult'],
    "Do you feel fulfilled in your personal life and career?":
        ['Completely fulfilled', 'Mostly fulfilled', 'Somewhat fulfilled', 'Not fulfilled'],
    "Have you ever received any form of psychological therapy?":
        ['No, never', 'Yes, but it was a long time ago', 'Yes, recently', 'Yes, I am currently in therapy'],
    "How would you describe your usual energy levels?":
        ['High', 'Normal', 'Somewhat low', 'Very low'],
    "How often do you struggle with focus or concentration?":
        ['Never', 'Rarely', 'Occasionally', 'Frequently'],
    "When faced with a challenge, how do you usually react?":
        ['I thrive on challenges', 'I take it as a learning opportunity', 'I feel anxious but manage to cope', 'I feel overwhelmed'],
    "How much control do you feel you have over your life direction?":
        ['Full control', 'Some control', 'Little control', 'No control']
}

category_questions = {
    "Mood": [0, 1, 6, 9, 17],
    "Relationships": [2, 9, 12, 13],
    "Habits": [3, 4, 5, 16],
    "Focus": [10, 11, 13, 18, 19]
}

response_weights = {
    'Rarely or none of the time': 0, 'Some of the time': 1, 'Often': 2, 'Nearly every day': 3,
    'Not at all': 0, 'Several days': 1, 'More than half the days': 2, 'Nearly every day': 3,
    'Very comfortable': 0, 'Somewhat comfortable': 1, 'Uncomfortable': 2, 'Very uncomfortable': 3,
    'Regular and restful': 0, 'Occasionally restless': 1, 'Frequently interrupted': 2, 'Poor and unsatisfying': 3,
    'No symptoms': 0, 'Mild symptoms': 1, 'Moderate symptoms': 2, 'Severe symptoms': 3,
    'Physical activity': 0, 'Talking to friends/family': 0, 'Avoiding the situation': 1, 'Substance use': 3,
    'Mostly positive': 0, 'Mixed feelings': 1, 'Mostly negative': 2, 'I struggle with self-perception': 3,
    'Very satisfied': 0, 'Somewhat satisfied': 1, 'Not very satisfied': 2, 'Not satisfied at all': 3,
    'No change': 0, 'Slight change': 1, 'Noticeable change': 2, 'Significant change': 3,
    'Very healthy': 0, 'Generally healthy': 1, 'Strained': 2, 'Very strained': 3,
    'I excel at managing stress': 0, 'I generally manage well': 1, 'I manage it with some difficulty': 2, 'I find it overwhelming and hard to manage': 3,
    'Almost always': 0, 'Often': 1, 'Sometimes': 2, 'Almost never': 3,
    'I always have the support I need': 0, 'I usually have enough support': 1, 'I have some support, but I need more': 2, 'I don’t have enough support': 3,
    'Not difficult at all': 0, 'Not very difficult': 1, 'Somewhat difficult': 2, 'Very difficult': 3,
    'Completely fulfilled': 0, 'Mostly fulfilled': 1, 'Somewhat fulfilled': 2, 'Not fulfilled': 3,
    'No, never': 0, 'Yes, but it was a long time ago': 1, 'Yes, recently': 2, 'Yes, I am currently in therapy': 3,
    'High': 0, 'Normal': 1, 'Somewhat low': 2, 'Very low': 3,
    'Never': 0, 'Rarely': 1, 'Occasionally': 2, 'Frequently': 3,
    'I thrive on challenges': 0, 'I take it as a learning opportunity': 1, 'I feel anxious but manage to cope': 2, 'I feel overwhelmed': 3,
    'Full control': 0, 'Some control': 1, 'Little control': 2, 'No control': 3
}

def calculate_category_scores(responses):
    category_scores = {}
    max_scores = {}
    for category, question_indices in category_questions.items():
        category_score = sum(
            response_weights.get(responses[i], 0)
            for i in question_indices
            if i < len(responses)
        )
        max_score = len(question_indices) * 3
        category_scores[category] = category_score
        max_scores[category] = max_score

    category_percentages = {
        category: round((max_scores[category] - score) / max_scores[category] * 100, 2)
        for category, score in category_scores.items()
    }
    return category_scores, category_percentages

# -------------------------------------------------
#              Flask Routes (unchanged)
# -------------------------------------------------
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/login")
def login():
    return render_template("log.html")

@app.route("/logout")
def logout():
    return redirect(url_for('login'))
    
@app.route("/quiz")
def quiz_page():
    return render_template("quiz.html")

@app.route("/home")
def home_page():
    return render_template("home.html")

# -------------- QUIZ API Routes --------------
@app.route("/get-question", methods=["GET"])
def get_question():
    index = int(request.args.get('index', 0))
    question_keys = list(questions.keys())
    if index >= len(question_keys):
        return jsonify({"error": "No more questions"}), 400

    question_text = question_keys[index]
    options = questions[question_text]
    return jsonify({
        "question": question_text,
        "options": options
    })

@app.route("/submit-responses", methods=["POST"])
def submit_responses():
    try:
        data = request.json
        responses = data.get('responses', [])
        userid = data.get('userid', None)
        
        if not responses or len(responses) != len(questions):
            return jsonify({"error": "Incomplete or invalid responses"}), 400

        category_scores, category_percentages = calculate_category_scores(responses)
        
        # Save quiz results to MongoDB if userid is provided
        if userid:
            save_quiz_results(userid, responses, category_scores, category_percentages)
        
        return jsonify({
            "mood": category_percentages.get("Mood", 0),
            "relationships": category_percentages.get("Relationships", 0),
            "habits": category_percentages.get("Habits", 0),
            "focus": category_percentages.get("Focus", 0)
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# -------------- User Management Routes --------------
@app.route("/register-user", methods=["POST"])
def register_user():
    try:
        data = request.json
        userid = data.get('userid', None)
        username = data.get('username', '')
        email = data.get('email', None)
        firebase_uid = data.get('firebase_uid', None)
        
        if not userid or not username:
            return jsonify({"error": "UserID and username are required"}), 400
        
        # Save user data to MongoDB
        success = save_user_data(userid, username, email, firebase_uid)
        
        if success:
            return jsonify({
                "message": "User registered successfully",
                "userid": userid,
                "username": username,
                "email": email
            })
        else:
            return jsonify({"error": "Failed to save user data"}), 500
            
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/get-user-data", methods=["GET"])
def get_user_data_route():
    try:
        userid = request.args.get('userid', None)
        
        if not userid:
            return jsonify({"error": "UserID is required"}), 400
        
        user_data = get_user_data(userid)
        
        if user_data:
            # Remove MongoDB _id field for JSON serialization
            user_data.pop('_id', None)
            return jsonify(user_data)
        else:
            return jsonify({"error": "User not found"}), 404
            
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/get-user-by-firebase-uid", methods=["GET"])
def get_user_by_firebase_uid_route():
    try:
        firebase_uid = request.args.get('firebase_uid', None)
        
        if not firebase_uid:
            return jsonify({"error": "Firebase UID is required"}), 400
        
        user_data = get_user_by_firebase_uid(firebase_uid)
        
        if user_data:
            # Remove MongoDB _id field for JSON serialization
            user_data.pop('_id', None)
            return jsonify(user_data)
        else:
            return jsonify({"error": "User not found"}), 404
            
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/get-quiz-history", methods=["GET"])
def get_quiz_history_route():
    try:
        userid = request.args.get('userid', None)
        limit = int(request.args.get('limit', 10))
        
        if not userid:
            return jsonify({"error": "UserID is required"}), 400
        
        quiz_history = get_user_quiz_history(userid, limit)
        
        # Remove MongoDB _id fields for JSON serialization
        for quiz in quiz_history:
            quiz.pop('_id', None)
        
        return jsonify({"quiz_history": quiz_history})
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/get-therapy-insights", methods=["GET"])
def get_therapy_insights_route():
    try:
        userid = request.args.get('userid', None)
        
        if not userid:
            return jsonify({"error": "UserID is required"}), 400
        
        insights = generate_therapy_insights(userid)
        
        if insights:
            return jsonify(insights)
        else:
            return jsonify({"error": "Failed to generate insights"}), 500
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# -------------- Therapy Insights Storage Routes --------------
@app.route("/therapy-insights", methods=["GET"])
def get_stored_therapy_insights():
    try:
        userid = request.args.get('userid', None)
        if not userid:
            return jsonify({"error": "UserID is required"}), 400
        if therapy_insights_col is None:
            return jsonify({"error": "Therapy insights collection unavailable"}), 500
        doc = therapy_insights_col.find_one({"userid": userid})
        if not doc:
            return jsonify({"error": "No stored insights found"}), 404
        doc.pop("_id", None)
        return jsonify(doc.get("insights", {}))
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/therapy-insights/refresh", methods=["POST"])
def refresh_therapy_insights():
    try:
        data = request.json or {}
        userid = data.get("userid") or request.args.get("userid")
        if not userid:
            return jsonify({"error": "UserID is required"}), 400
        refreshed = generate_therapy_insights(userid)
        if refreshed:
            return jsonify(refreshed)
        return jsonify({"error": "Failed to refresh insights"}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/dominant-emotion", methods=["GET"])
def get_dominant_emotion():
    """
    Get today's dominant emotion for a user.
    """
    try:
        userid = request.args.get('userid', None)
        date_str = request.args.get('date', None)  # Optional: YYYY-MM-DD format
        
        if not userid:
            return jsonify({"error": "UserID is required"}), 400
        
        target_date = None
        if date_str:
            try:
                target_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        emotion_data = get_dominant_emotion_for_date(userid, target_date)
        return jsonify(emotion_data)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/daily-emotions", methods=["GET"])
def get_daily_emotions():
    """
    Get daily emotion history for a user (last N days).
    """
    try:
        userid = request.args.get('userid', None)
        days = int(request.args.get('days', 7))
        
        if not userid:
            return jsonify({"error": "UserID is required"}), 400
        
        if days < 1 or days > 30:
            return jsonify({"error": "Days must be between 1 and 30"}), 400
        
        daily_emotions = get_daily_emotion_history(userid, days)
        return jsonify({"daily_emotions": daily_emotions})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/chat-history", methods=["GET"])
def get_chat_history():
    """
    Get chat history grouped by day for a user.
    Returns chats grouped by date, excluding deleted chats.
    """
    try:
        userid = request.args.get('userid', None)
        
        if not userid:
            return jsonify({"error": "UserID is required"}), 400
        
        if user_chats_col is None:
            return jsonify({"error": "Chat collection unavailable"}), 500
        
        # Get all non-deleted chats for the user
        cursor = user_chats_col.find({
            "userid": userid,
            "is_deleted": {"$ne": True}
        }).sort("timestamp", -1)
        
        chats = list(cursor)
        
        # Group chats by date
        chats_by_date = {}
        for chat in chats:
            timestamp = chat.get("timestamp")
            if isinstance(timestamp, datetime):
                date_key = timestamp.strftime("%Y-%m-%d")
                date_display = timestamp.strftime("%B %d, %Y")
            else:
                # Handle string timestamps
                try:
                    dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                    date_key = dt.strftime("%Y-%m-%d")
                    date_display = dt.strftime("%B %d, %Y")
                except:
                    date_key = "unknown"
                    date_display = "Unknown Date"
            
            if date_key not in chats_by_date:
                chats_by_date[date_key] = {
                    "date": date_key,
                    "date_display": date_display,
                    "chats": []
                }
            
            # Format chat data
            chat_id = str(chat.get("_id", ""))
            chat_title = chat.get("chat_title", None)
            message = chat.get("message", "")
            
            # Generate smart title from user chat message
            if not chat_title:
                # Use the actual user message as title, truncate if too long
                chat_title = message.strip()
                if len(chat_title) > 60:
                    # Try to cut at a word boundary
                    truncated = chat_title[:60].rsplit(' ', 1)[0]
                    chat_title = truncated + "..." if truncated else chat_title[:60] + "..."
                if not chat_title.strip():
                    chat_title = "Untitled Chat"
            
            # Format timestamp for display
            if isinstance(timestamp, datetime):
                time_display = timestamp.strftime("%I:%M %p")
            else:
                try:
                    dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                    time_display = dt.strftime("%I:%M %p")
                except:
                    time_display = "Unknown"
            
            chats_by_date[date_key]["chats"].append({
                "id": chat_id,
                "title": chat_title,
                "message": message,
                "response": chat.get("response", ""),
                "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                "time_display": time_display
            })
        
        # Convert to list and sort by date (newest first)
        result = list(chats_by_date.values())
        result.sort(key=lambda x: x["date"], reverse=True)
        
        return jsonify({"chat_history": result})
        
    except Exception as e:
        print(f"[Chat History] Error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/chat/rename", methods=["POST"])
def rename_chat():
    """
    Rename a chat by updating its title.
    """
    try:
        data = request.json or {}
        userid = data.get("userid", None)
        chat_id = data.get("chat_id", None)
        new_title = data.get("title", "").strip()
        
        if not userid:
            return jsonify({"error": "UserID is required"}), 400
        
        if not chat_id:
            return jsonify({"error": "Chat ID is required"}), 400
        
        if not new_title:
            return jsonify({"error": "Title cannot be empty"}), 400
        
        if len(new_title) > 100:
            return jsonify({"error": "Title must be 100 characters or less"}), 400
        
        if user_chats_col is None:
            return jsonify({"error": "Chat collection unavailable"}), 500
        
        # Update the chat title
        result = user_chats_col.update_one(
            {
                "_id": ObjectId(chat_id),
                "userid": userid
            },
            {
                "$set": {
                    "chat_title": new_title,
                    "title_updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.matched_count == 0:
            return jsonify({"error": "Chat not found"}), 404
        
        return jsonify({
            "message": "Chat renamed successfully",
            "chat_id": chat_id,
            "title": new_title
        })
        
    except Exception as e:
        print(f"[Rename Chat] Error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/chat/delete", methods=["POST"])
def delete_chat():
    """
    Hard delete a chat by removing it from MongoDB.
    """
    try:
        data = request.json or {}
        userid = data.get("userid", None)
        chat_id = data.get("chat_id", None)
        
        if not userid:
            return jsonify({"error": "UserID is required"}), 400
        
        if not chat_id:
            return jsonify({"error": "Chat ID is required"}), 400
        
        if user_chats_col is None:
            return jsonify({"error": "Chat collection unavailable"}), 500
        
        # Hard delete the chat from MongoDB
        result = user_chats_col.delete_one(
            {
                "_id": ObjectId(chat_id),
                "userid": userid
            }
        )
        
        if result.deleted_count == 0:
            return jsonify({"error": "Chat not found"}), 404
        
        return jsonify({
            "message": "Chat deleted successfully",
            "chat_id": chat_id
        })
        
    except Exception as e:
        print(f"[Delete Chat] Error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# -------------------------------------------------
# -------------- Daily Dominant Emotion Calculation --------------
# -------------------------------------------------
def get_dominant_emotion_for_date(userid: str, target_date: datetime = None):
    """
    Calculate the dominant emotion for a specific date based on chats from that day.
    Returns the emotion with the highest count for that day, or 'none' if no chats.
    
    Args:
        userid: User ID to filter chats
        target_date: datetime object for the target date (defaults to today in UTC)
    
    Returns:
        dict with 'emotion' (string), 'count' (int), 'flags_count' (dict), 'date' (string)
    """
    if not userid or user_chats_col is None:
        return {"emotion": "none", "count": 0, "flags_count": {}, "date": None}
    
    try:
        # Default to today if no date provided
        if target_date is None:
            target_date = datetime.utcnow()
        
        # Get start and end of the target date (in UTC)
        start_of_day = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
        end_of_day = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59)
        
        # Query chats for that specific day (excluding deleted)
        query = {
            "userid": userid,
            "timestamp": {
                "$gte": start_of_day,
                "$lte": end_of_day
            },
            "is_deleted": {"$ne": True}
        }
        
        cursor = user_chats_col.find(query)
        chat_records = list(cursor)
        
        if not chat_records:
            date_str = target_date.strftime("%Y-%m-%d")
            return {
                "emotion": "none",
                "count": 0,
                "flags_count": {
                    "anger": 0,
                    "sadness": 0,
                    "rumination": 0,
                    "anxiety": 0,
                    "withdrawal": 0,
                    "suicidal_ideation": 0
                },
                "date": date_str
            }
        
        # Count flags for that day
        flags_count = {
            "anger": 0,
            "sadness": 0,
            "rumination": 0,
            "anxiety": 0,
            "withdrawal": 0,
            "suicidal_ideation": 0
        }
        
        for record in chat_records:
            analysis = record.get("analysis", {})
            flags = analysis.get("flags", {})
            for flag in flags_count:
                if flags.get(flag, False):
                    flags_count[flag] += 1
        
        # Determine dominant emotion (highest count)
        if any(flags_count.values()):
            dominant_emotion = max(flags_count, key=flags_count.get)
        else:
            dominant_emotion = "none"
        
        date_str = target_date.strftime("%Y-%m-%d")
        return {
            "emotion": dominant_emotion,
            "count": len(chat_records),
            "flags_count": flags_count,
            "date": date_str
        }
        
    except Exception as e:
        print(f"[Dominant Emotion] Error calculating for date: {e}")
        return {"emotion": "none", "count": 0, "flags_count": {}, "date": None}

def get_daily_emotion_history(userid: str, days: int = 7):
    """
    Get dominant emotion history for the last N days.
    
    Args:
        userid: User ID
        days: Number of days to look back (default 7)
    
    Returns:
        List of dicts with daily dominant emotions, sorted by date (newest first)
    """
    if not userid:
        return []
    
    daily_emotions = []
    today = datetime.utcnow()
    
    for i in range(days):
        target_date = datetime(today.year, today.month, today.day, 0, 0, 0)
        target_date = datetime(target_date.year, target_date.month, target_date.day) - timedelta(days=i)
        emotion_data = get_dominant_emotion_for_date(userid, target_date)
        daily_emotions.append(emotion_data)
    
    # Sort by date (newest first)
    daily_emotions.sort(key=lambda x: x.get("date", ""), reverse=True)
    return daily_emotions

# -------------------------------------------------
# -------------- Therapy Insights Analysis --------------
# -------------------------------------------------
def generate_advice_from_recent_chats(userid: str, days: int = 7):
    """
    Generate therapist-style advice based on user's recent messages.
    Focuses on the last N days of chats, with emphasis on today.
    
    Returns a list of 3-4 actionable advice items.
    """
    if not userid or user_chats_col is None:
        return ["Start a conversation to receive personalized advice"]
    
    try:
        # Get recent chats (last N days, prioritizing today)
        today = datetime.utcnow()
        start_date = today - timedelta(days=days)
        
        # Get chats from the last N days (excluding deleted)
        cursor = user_chats_col.find({
            "userid": userid,
            "timestamp": {"$gte": start_date},
            "is_deleted": {"$ne": True}
        }).sort("timestamp", -1).limit(30)
        
        recent_chats = list(cursor)
        
        if not recent_chats:
            return ["Start a conversation to receive personalized advice"]
        
        # Prepare recent messages for analysis
        recent_messages = []
        emotion_flags = {
            "sadness": 0,
            "anxiety": 0,
            "stress": 0,
            "rumination": 0,
            "anger": 0,
            "withdrawal": 0
        }
        
        for chat in recent_chats:
            message = chat.get("message", "")
            analysis = chat.get("analysis", {})
            flags = analysis.get("flags", {})
            
            if message:
                recent_messages.append(message)
            
            # Count emotion flags
            if flags.get("sadness"):
                emotion_flags["sadness"] += 1
            if flags.get("anxiety"):
                emotion_flags["anxiety"] += 1
            if flags.get("rumination"):
                emotion_flags["rumination"] += 1
            if flags.get("anger"):
                emotion_flags["anger"] += 1
            if flags.get("withdrawal"):
                emotion_flags["withdrawal"] += 1
            # Check for stress indicators
            if "stress" in message.lower() or "stressed" in message.lower() or "overwhelmed" in message.lower():
                emotion_flags["stress"] += 1
        
        # Build prompt for advice generation
        dominant_emotions = [k for k, v in emotion_flags.items() if v > 0]
        dominant_emotions_str = ", ".join(dominant_emotions) if dominant_emotions else "general emotional patterns"
        
        advice_prompt = f"""
You are a warm, experienced Cognitive Behavioral Therapist. Based on the user's recent messages, generate 3-4 actionable pieces of advice.

## User's Recent Emotional Patterns:
{dominant_emotions_str}

## Recent Messages (last {days} days):
{chr(10).join([f"- {msg[:200]}" for msg in recent_messages[:10]])}

## Task:
Based on the user's recent messages, identify the key emotional patterns and generate 3-4 therapist-style, actionable pieces of advice. The advice should:

1. Reflect the user's emotional state (e.g., sadness, anxiety, stress, rumination).
2. Encourage healthy coping behaviors using CBT principles.
3. Be safe, supportive, and empowering.
4. Keep the tone warm, concise, and non-judgmental.
5. Be specific and actionable (what the user can do, not just general statements).

## Output Format:
Return ONLY a JSON array with 3-4 advice strings. No other text. Example:
["Advice item 1", "Advice item 2", "Advice item 3", "Advice item 4"]

## Important:
- Focus on the most recent emotional patterns shown in the messages
- Use CBT techniques like cognitive reframing, behavioral activation, mindfulness
- Make each advice item one clear, actionable sentence
- Be empathetic and validating
- Do NOT include any explanations or meta-commentary, just the advice array
"""
        
        # Get advice from Gemini
        advice_response = query_gemini_api(advice_prompt, max_output_tokens=300, temperature=0.3)
        
        if advice_response:
            try:
                # Try to extract JSON array
                json_match = re.search(r'\[.*\]', advice_response, re.DOTALL)
                if json_match:
                    advice_list = json.loads(json_match.group())
                    if isinstance(advice_list, list) and len(advice_list) > 0:
                        # Ensure we have 3-4 items
                        return advice_list[:4] if len(advice_list) >= 3 else advice_list
            except Exception as e:
                print(f"[Advice] JSON parsing error: {e}")
        
        # Fallback: generate basic advice based on emotion flags
        fallback_advice = []
        if emotion_flags["anxiety"] > 0:
            fallback_advice.append("Practice deep breathing exercises when you notice anxiety building up—try 4-7-8 breathing (inhale 4, hold 7, exhale 8).")
        if emotion_flags["sadness"] > 0:
            fallback_advice.append("Engage in one small, meaningful activity today, even if it's just a 10-minute walk or calling a friend.")
        if emotion_flags["rumination"] > 0:
            fallback_advice.append("When you notice yourself replaying thoughts, gently redirect your attention to the present moment—notice 3 things you can see, hear, and feel.")
        if emotion_flags["stress"] > 0:
            fallback_advice.append("Break overwhelming tasks into smaller, manageable steps and focus on completing just one step at a time.")
        
        if not fallback_advice:
            fallback_advice.append("Continue engaging in regular self-reflection and reach out for support when needed.")
            fallback_advice.append("Practice self-compassion—acknowledge your feelings without judgment.")
            fallback_advice.append("Maintain a consistent routine that includes activities you find meaningful or enjoyable.")
        
        return fallback_advice[:4]
        
    except Exception as e:
        print(f"[Advice] Error generating advice: {e}")
        return ["Continue engaging in self-reflection and reach out for support when needed."]

def generate_therapy_insights(userid: str):
    """
    Generate therapy insights by analyzing user's recent chat history (day-to-day focus).
    Returns structured insights in JSON format.
    """
    if not userid or user_chats_col is None:
        return None
    
    try:
        # Focus on recent chats (last 7 days, with emphasis on today)
        today = datetime.utcnow()
        start_date = today - timedelta(days=7)
        
        # Get recent chat history (prioritizing today and recent days, excluding deleted)
        cursor = user_chats_col.find({
            "userid": userid,
            "timestamp": {"$gte": start_date},
            "is_deleted": {"$ne": True}
        }).sort("timestamp", -1).limit(50)
        chat_records = list(cursor)
        
        # Get today's dominant emotion
        today_emotion = get_dominant_emotion_for_date(userid, datetime.utcnow())
        today_dominant_emotion = today_emotion.get("emotion", "none")
        
        # Get yesterday's dominant emotion for comparison
        yesterday = today - timedelta(days=1)
        yesterday_emotion = get_dominant_emotion_for_date(userid, yesterday)
        yesterday_dominant_emotion = yesterday_emotion.get("emotion", "none")
        
        if not chat_records:
            # Generate advice even if no recent chats (will return default advice)
            advice = generate_advice_from_recent_chats(userid, days=7)
            empty_insights = {
                "overview": {
                    "total_chats": 0,
                    "active_days": 0,
                    "dominant_emotion": today_dominant_emotion,
                    "dominant_emotion_today": today_dominant_emotion,
                    "risk_level": "low"
                },
                "daily_emotions": [today_emotion],
                "flags_distribution": {
                    "anger": 0,
                    "sadness": 0,
                    "rumination": 0,
                    "anxiety": 0,
                    "withdrawal": 0,
                    "suicidal_ideation": 0
                },
                "trends": ["No conversation history available"],
                "top_phrases": ["No data available"],
                "advice": advice
            }
            try:
                if 'therapy_insights_col' in globals() and therapy_insights_col is not None:
                    therapy_insights_col.update_one(
                        {"userid": userid},
                        {"$set": {"userid": userid, "insights": empty_insights, "updated_at": datetime.utcnow()}},
                        upsert=True
                    )
            except Exception as e:
                print(f"[MongoDB] Failed to save empty insights: {e}")
            return empty_insights
        
        # Prepare data for analysis
        analysis_data = []
        for record in chat_records:
            analysis = record.get("analysis", {})
            if analysis:
                analysis_data.append({
                    "message": record.get("message", ""),
                    "analysis": analysis,
                    "timestamp": record.get("timestamp")
                })
        
        # Create analysis prompt focused on day-to-day changes
        analysis_prompt = f"""
You are an assistant that analyzes therapy conversations for day-to-day insights. 
You are NOT a therapist in this step — do not give advice or exercises. 
Your job is to summarize patterns, highlight emotional signals, and 
output structured insights for a dashboard with a focus on RECENT and DAILY changes.

## Context
You will receive a batch of chat records from the last 7 days (with emphasis on today). 
Each record may contain:
- user message
- assistant response
- analysis (flags: anger, sadness, rumination, anxiety, withdrawal, suicidal_ideation)
- short_summary
- timestamp

## Important Focus
- Prioritize TODAY's emotional patterns and chats
- Compare today's patterns with recent days to identify shifts
- Highlight day-to-day changes in emotional state
- Focus on recent trends, not historical patterns

## Task
1. Review all the messages and analyses, with special attention to TODAY's chats.
2. Identify emotional patterns, recurring issues, and shifts in mood DAY BY DAY.
3. Highlight which psychological flags (anger, sadness, rumination, anxiety, withdrawal, suicidal_ideation) are most common RECENTLY.
4. Extract the most frequent short summaries or phrases from RECENT chats.
5. Suggest trends focusing on day-to-day changes (e.g., "anxiety increased today compared to yesterday", "rumination patterns shifted from work stress to relationship concerns").
6. Detect potential risk signals (especially suicidal_ideation) and recommend priority level (low, medium, high).
7. Present insights in a structured JSON format with emphasis on RECENT patterns.

## Output format (strict JSON):
{{
  "overview": {{
    "total_chats": <number>,
    "active_days": <number>,
    "dominant_emotion": "<string> - today's dominant emotion based on today's chats>",
    "dominant_emotion_today": "<string> - same as dominant_emotion>",
    "risk_level": "low|medium|high"
  }},
  "flags_distribution": {{
    "anger": <count>,
    "sadness": <count>,
    "rumination": <count>,
    "anxiety": <count>,
    "withdrawal": <count>,
    "suicidal_ideation": <count>
  }},
  "emotion_analysis": {{
    "primary_emotion": "<string>",
    "intensity_level": "low|moderate|high",
    "trigger_context": "<short phrase>"
  }},
  "cognitive_distortions": [
    {{
      "type": "<string>",
      "example_phrase": "<short phrase>",
      "explanation": "<short explanation>"
    }}
  ],
  "trends": [
    "string insight 1",
    "string insight 2",
    "string insight 3"
  ],
  "top_phrases": [
    "string phrase 1",
    "string phrase 2",
    "string phrase 3"
  ],
  "advice": [
    "Keep monitoring sadness and anxiety",
    "Pay special attention to rumination patterns",
    "Escalate if suicidal_ideation reappears"
  ],
  "cbt_reframe": {{
    "original_thought": "<short phrase>",
    "balanced_thought": "<short phrase>",
    "encouragement": "<short sentence>"
  }},
  "action_advice": [
    "short action 1",
    "short action 2",
    "short action 3"
  ],
  "progress_recommendation": [
    "short recommendation 1",
    "short recommendation 2",
    "short recommendation 3"
  ],
  "risk_alert": "<short sentence>"
}}

## Rules
- Do NOT include raw chat text, only summaries/insights.
- Be concise but informative.
- Always fill every field, even if with zero counts or "none".
- Focus on RECENT patterns and DAY-TO-DAY changes, especially TODAY's patterns.
- Prioritize today's emotional state over historical patterns.

## Data to analyze (recent chats, especially today):
{json.dumps(analysis_data[:30], default=str)}
"""
        
        # Get insights from Gemini
        insights_response = query_gemini_api(analysis_prompt, max_output_tokens=800, temperature=0.1)
        
        if insights_response:
            try:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', insights_response, re.DOTALL)
                if json_match:
                    insights_json = json.loads(json_match.group())
                    # Override dominant_emotion with today's value
                    if "overview" in insights_json:
                        insights_json["overview"]["dominant_emotion"] = today_dominant_emotion
                        insights_json["overview"]["dominant_emotion_today"] = today_dominant_emotion
                    # Add daily emotion history
                    insights_json["daily_emotions"] = get_daily_emotion_history(userid, days=7)
                    # Generate advice based on recent chats
                    insights_json["advice"] = generate_advice_from_recent_chats(userid, days=7)
                    try:
                        if 'therapy_insights_col' in globals() and therapy_insights_col is not None:
                            therapy_insights_col.update_one(
                                {"userid": userid},
                                {"$set": {"userid": userid, "insights": insights_json, "updated_at": datetime.utcnow()}},
                                upsert=True
                            )
                    except Exception as e:
                        print(f"[MongoDB] Failed to save insights: {e}")
                    return insights_json
            except Exception as e:
                print(f"[Insights] JSON parsing error: {e}")
        
        # Fallback: generate basic insights from data
        fallback = generate_fallback_insights(chat_records, userid)
        try:
            if 'therapy_insights_col' in globals() and therapy_insights_col is not None:
                therapy_insights_col.update_one(
                    {"userid": userid},
                    {"$set": {"userid": userid, "insights": fallback, "updated_at": datetime.utcnow()}},
                    upsert=True
                )
        except Exception as e:
            print(f"[MongoDB] Failed to save fallback insights: {e}")
        return fallback
        
    except Exception as e:
        print(f"[Insights] Error generating insights: {e}")
        return None

def generate_fallback_insights(chat_records, userid: str = None):
    """Generate basic insights when AI analysis fails."""
    total_chats = len(chat_records)
    
    # Get today's dominant emotion (based on today's chats only)
    today_emotion = get_dominant_emotion_for_date(userid, datetime.utcnow()) if userid else {"emotion": "none", "flags_count": {}}
    today_dominant_emotion = today_emotion.get("emotion", "none")
    today_flags_count = today_emotion.get("flags_count", {})
    
    # Count flags across recent chats (last 7 days) for distribution
    flags_count = {
        "anger": 0,
        "sadness": 0,
        "rumination": 0,
        "anxiety": 0,
        "withdrawal": 0,
        "suicidal_ideation": 0
    }
    
    for record in chat_records:
        analysis = record.get("analysis", {})
        flags = analysis.get("flags", {})
        for flag in flags_count:
            if flags.get(flag, False):
                flags_count[flag] += 1
    
    # Use today's dominant emotion for overview
    dominant_emotion = today_dominant_emotion
    risk_level = "high" if flags_count["suicidal_ideation"] > 0 else "medium" if any(flags_count[f] > 2 for f in ["anger", "sadness", "anxiety"]) else "low"
    # Intensity heuristic
    total_flags_hits = sum(flags_count.values())
    if total_flags_hits >= 8:
        intensity_level = "high"
    elif total_flags_hits >= 4:
        intensity_level = "moderate"
    else:
        intensity_level = "low"
    # Trigger context heuristic
    trigger_context = "generalized stressors"
    if flags_count["anxiety"] > 0:
        trigger_context = "performance or uncertainty"
    if flags_count["rumination"] > 0:
        trigger_context = "repetitive self-focus"
    if flags_count["anger"] > 0:
        trigger_context = "perceived unfairness or conflict"
    
    # Get daily emotion history
    daily_emotions = get_daily_emotion_history(userid, days=7) if userid else []
    
    # Generate advice based on recent chats
    advice = generate_advice_from_recent_chats(userid, days=7) if userid else ["Start a conversation to receive personalized advice"]
    
    return {
        "overview": {
            "total_chats": total_chats,
            "active_days": min(total_chats, 7),
            "dominant_emotion": dominant_emotion,
            "dominant_emotion_today": dominant_emotion,
            "risk_level": risk_level
        },
        "daily_emotions": daily_emotions,
        "flags_distribution": flags_count,
        "emotion_analysis": {
            "primary_emotion": dominant_emotion,
            "intensity_level": intensity_level,
            "trigger_context": trigger_context
        },
        "cognitive_distortions": [
            {
                "type": "catastrophizing" if flags_count["anxiety"] > 0 else "negative filter" if flags_count["sadness"] > 0 else "generalization",
                "example_phrase": "It will all go wrong" if flags_count["anxiety"] > 0 else "Nothing ever works for me" if flags_count["sadness"] > 0 else "This always happens",
                "explanation": "You might be assuming the worst without full evidence."
            }
        ],
        "trends": [
            f"Most common emotion: {dominant_emotion}",
            f"Total conversations: {total_chats}",
            "Continue monitoring emotional patterns"
        ],
        "top_phrases": [
            "Core emotion identified",
            "Cognitive distortion detected",
            "Reframing suggested",
            "Action step generated"
        ],
        "advice": advice,
        "cbt_reframe": {
            "original_thought": "I can't handle this" if flags_count["anxiety"] > 0 else "I'm not good enough" if flags_count["sadness"] > 0 else "This is unfair",
            "balanced_thought": "I can take this one step at a time and ask for help if needed.",
            "encouragement": "Try focusing on what’s within your control and acknowledge your effort."
        },
        "action_advice": [
            "Take a short grounding break — slow breathing or a 2-min walk.",
            "List 1 thing that went well today.",
            "Focus on small, doable goals instead of perfection.",
            "Note improvement signs before next session."
        ],
        "progress_recommendation": [
            "Schedule next reflection check",
            "Maintain emotional journal entries",
            "Continue CBT dialogue tracking for pattern analysis"
        ],
        "risk_alert": "High distress detected — encourage seeking professional help." if risk_level == "high" else "Stable state — maintain current progress and awareness."
    }

# -------------------------------------------------
# -------------- Chat Endpoint (conversational, listen-first + summarization) --------------
# -------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    """
    Conversational chat flow with adaptive summarization:
    1) Validate message
    2) Run analysis (Gemini JSON analysis attempt -> fallback heuristics)
    3) If suicidal flag -> return crisis guidance immediately
    4) Decide whether to summarize (based on previous messages count)
    5) Build conversational or summarization prompt accordingly
    6) Ask Gemini to generate the reply
    7) Save analysis + assistant reply to DB and return it
    """
    data = request.json or {}
    user_message = data.get("message", "").strip()
    userid = data.get("userid", None)  # optional

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    # 1) Analysis (attempt Gemini-assisted JSON analysis, fallback to heuristics)
    analysis = None
    try:
        analysis = analyze_message_with_gemini(user_message, userid)
    except Exception as e:
        print(f"[Analysis] Gemini analysis error: {e}")
        analysis = None

    if not analysis:
        analysis = local_emotion_heuristics(user_message)

    flags = analysis.get("flags", {})
    confidence = analysis.get("confidence", 0.0)
    short_summary = analysis.get("short_summary", "")

    # 2) Persist the raw message + analysis early (for audit / RAG)
    inserted_id = None
    try:
        if user_chats_col is not None:
            # Generate smart title from user message
            default_title = user_message.strip()
            if len(default_title) > 60:
                # Try to cut at a word boundary
                truncated = default_title[:60].rsplit(' ', 1)[0]
                default_title = truncated + "..." if truncated else default_title[:60] + "..."
            if not default_title.strip():
                default_title = "Untitled Chat"
            
            record = {
                "userid": userid,
                "message": user_message,
                "chat_title": default_title,  # Default title from message
                "analysis": {
                    "flags": flags,
                    "confidence": confidence,
                    "short_summary": short_summary
                },
                "response": None,
                "is_deleted": False,  # Track deletion status
                "timestamp": datetime.utcnow()
            }
            result = user_chats_col.insert_one(record)
            inserted_id = result.inserted_id
    except Exception as e:
        print(f"[MongoDB] Failed to save analysis: {e}")
        inserted_id = None

    # 3) Crisis handling - immediate
    if flags.get("suicidal_ideation"):
        crisis_text = (
            "I'm really sorry you're feeling this way. I can't provide emergency help, but "
            "if you are thinking about harming yourself or feel you might act on these thoughts, "
            "please contact your local emergency services or a crisis hotline right now. "
            "If you're in the US you can dial or text 988 to reach the Suicide & Crisis Lifeline. "
            "If you're elsewhere and safe to do so, please contact local emergency services or a trusted person. "
            "Would you like resources or to talk through how you're feeling right now?"
        )
        try:
            if user_chats_col is not None and inserted_id:
                user_chats_col.update_one({"_id": ObjectId(inserted_id)}, {"$set": {"response": crisis_text}})
        except Exception as e:
            print(f"[MongoDB] Failed to update crisis response: {e}")

        return jsonify({
            "type": "crisis",
            "analysis": analysis,
            "response": crisis_text
        }), 200

    # 4) Decide summarization mode based on prior conversation count
    summarize_mode = False
    # Stress/fear micro-CBT mode detection
    stress_fear_mode = False
    try:
        summarize_mode = should_summarize_for_user(userid, threshold=3)  # switch after 3 saved messages
    except Exception as e:
        print(f"[Decision] Error deciding summarize mode: {e}")
        summarize_mode = False
    try:
        # Trigger micro-CBT when anxiety flag is set or 'stress' mentioned
        text_lower = (user_message or "").lower()
        stress_fear_mode = bool(flags.get("anxiety")) or ("stress" in text_lower or "stressed" in text_lower or "fear" in text_lower or "scared" in text_lower)
    except Exception as e:
        print(f"[Decision] Error deciding stress/fear mode: {e}")
        stress_fear_mode = False

    # 5) Retrieve RAG contexts
    user_chats_snippets = []
    datasets_snippets = []
    therapist_profile = None
    try:
        if userid:
            user_chats_snippets = retrieve_user_chats_context(userid, limit=8)
            therapist_profile = analyze_quiz_responses_for_therapist_profile(userid)
        datasets_snippets = retrieve_datasets_context(user_message, limit=3)
    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")

    # 6) Build conversational prompt (listen-first) or summarization prompt
    conv_prompt = build_conversational_rag_prompt(
        instruction="Conversational CBT therapist (adaptive).",
        user_chats_snippets=user_chats_snippets,
        datasets_snippets=datasets_snippets,
        user_message=user_message,
        analysis=analysis,
        therapist_profile=therapist_profile,
        summarize_mode=summarize_mode,
        stress_fear_mode=stress_fear_mode
    )

    # 7) Ask Gemini to generate the reply
    # Stricter limits for short replies and deterministic output
    max_tokens = 150 if (summarize_mode or stress_fear_mode) else 60
    temperature = 0.0  # deterministic, less verbose / imaginative
    model_response = query_gemini_api(conv_prompt, max_output_tokens=max_tokens, temperature=temperature)
    if model_response is None:
        # fallback: deterministic brief reflection & one question or a short summary if summarize_mode
        if summarize_mode:
            fallback_resp = (
                "I hear how this has been wearing on you. In short: the worry makes it harder to act. "
                "1) Review what you prepared, 2) try a 5-minute grounding before the task, 3) break the next step into 5 minutes. "
                "Would you like to try step 2 now?"
            )
        else:
            fallback_resp = "I hear how overwhelmed you feel. What's the main thought that comes up first?"
        try:
            if user_chats_col is not None and inserted_id:
                user_chats_col.update_one({"_id": ObjectId(inserted_id)}, {"$set": {"response": fallback_resp}})
        except Exception as e:
            print(f"[MongoDB] Failed to update fallback response: {e}")
        return jsonify({
            "type": "normal",
            "analysis": analysis,
            "response": fallback_resp,
            "summarize_mode": summarize_mode
        }), 200

    # enforce short output (safety net)
    if stress_fear_mode:
        # Keep numbered structure; do not trim to sentences
        ai_response = str(model_response).strip()
    else:
        ai_response = trim_to_sentences(str(model_response), max_sentences=2)

    # 8) Save generated reply to DB
    try:
        if user_chats_col is not None and inserted_id:
            user_chats_col.update_one({"_id": ObjectId(inserted_id)}, {"$set": {"response": ai_response}})
    except Exception as e:
        print(f"[MongoDB] Failed to update saved response: {e}")

    # 8b) Auto-update therapy insights for this user based on latest chats
    try:
        if userid:
            generate_therapy_insights(userid)
    except Exception as e:
        print(f"[Insights] Auto-update failed: {e}")

    # 9) Return model reply with analysis metadata and whether summarize mode used
    return jsonify({
        "type": "normal",
        "analysis": analysis,
        "summarize_mode": summarize_mode,
        "response": ai_response
    }), 200

if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    try:
        port = int(os.getenv("FLASK_RUN_PORT", os.getenv("PORT", "5000")))
    except ValueError:
        port = 5000
    try:
        app.run(host=host, port=port)
    except OSError as e:
        if "Address already in use" in str(e):
            alt_port = port + 1
            print(f"[Flask] Port {port} in use. Retrying on {alt_port}...")
            app.run(host=host, port=alt_port)
        else:
            raise
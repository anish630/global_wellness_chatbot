import streamlit as st
import sqlite3, bcrypt, jwt, datetime, joblib, os, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from googletrans import Translator
from langdetect import detect
import spacy
# New imports for analytics
import pandas as pd
import plotly.express as px

# -----------------------------
# Config
# -----------------------------
SECRET_KEY = "wellness_secret_key"
translator = Translator()

# Load spaCy model with error handling
nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Model will be loaded later or user will see error message
    pass

# Admin credentials (hard-coded per requirement)
ADMIN_EMAIL = "anish@gmail.com"
ADMIN_PASSWORD = "anish123"

# Confidence threshold for model predictions (if below -> treat as unknown)
CONF_THRESHOLD = 0.60

# -----------------------------
# Database Setup
# -----------------------------
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

# users table (existing)
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    gender TEXT,
    language TEXT,
    email TEXT UNIQUE,
    password BLOB
)
''')

# knowledge_base table: intent -> JSON list of tips
c.execute('''
CREATE TABLE IF NOT EXISTS knowledge_base (
    intent TEXT PRIMARY KEY,
    title TEXT,
    tips_json TEXT
)
''')

# queries table: log each user query
c.execute('''
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    query_text TEXT,
    detected_intent TEXT,
    timestamp TEXT
)
''')

# feedback table
c.execute('''
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    query_id INTEGER,
    intent TEXT,
    feedback TEXT,
    comment TEXT,
    timestamp TEXT
)
''')
conn.commit()

# -----------------------------
# Default knowledge base (used to populate DB at first run)
# -----------------------------
def default_kb():
    return {
        "headache": {
            "title": "🤕 Headache Relief",
            "tips": [
                "Stay in a quiet, dark room to relax your mind.",
                "Drink enough water, as dehydration often causes headaches.",
                "Apply a cool or warm compress to your forehead or neck.",
                "Avoid looking at bright screens for too long.",
                "Take deep breaths and practice relaxation techniques.",
                "Eat a light snack if you haven't eaten recently.",
                "Sleep for 7–8 hours daily to reduce headache frequency.",
                "Massage your temples gently to relieve tension.",
                "Avoid excessive caffeine or alcohol.",
                "⚠️ Disclaimer: Consult a doctor if headache is severe, sudden, or accompanied by vision problems."
            ]
        },
        "fever": {
            "title": "⚕️ Fever Tips",
            "tips": [
                "Drink plenty of fluids to stay hydrated.",
                "Rest as much as possible to allow your body to recover.",
                "Take paracetamol or ibuprofen if necessary.",
                "Use a damp cloth to sponge your body to reduce temperature.",
                "See a doctor if fever lasts more than 2 days.",
                "Eat light meals to support your immune system.",
                "Monitor your temperature every 4–6 hours.",
                "Keep your room ventilated and cool.",
                "Avoid sudden cold exposure.",
                "⚠️ Disclaimer: High fever (>102°F) or persistent fever needs immediate medical attention."
            ]
        },
        "cut": {
            "title": "🩹 Wound Care",
            "tips": [
                "Rinse the cut with clean running water.",
                "Use an antiseptic cream to prevent infection.",
                "Cover the cut with a sterile bandage.",
                "Keep the area dry and clean.",
                "Seek medical help if cut is deep.",
                "Change dressing daily to avoid infection.",
                "Avoid touching the cut with dirty hands.",
                "Take tetanus vaccine if the cut is severe.",
                "Watch for redness or pus as infection signs.",
                "⚠️ Disclaimer: Severe cuts or heavy bleeding require immediate professional care."
            ]
        },
        "cough": {
            "title": "😷 Cough Relief",
            "tips": [
                "Drink warm water and herbal teas.",
                "Gargle with warm salt water for throat relief.",
                "Use honey and ginger for natural soothing.",
                "Take steam inhalation to ease congestion.",
                "Rest your voice and avoid smoke.",
                "Avoid cold drinks and ice cream.",
                "Eat light, warm meals to soothe throat.",
                "Keep humidifier in room to prevent dryness.",
                "Consult doctor if cough lasts over 10 days.",
                "⚠️ Disclaimer: Persistent cough, especially with blood, needs urgent medical check."
            ]
        },
        "cold": {
            "title": "🤧 Cold Care",
            "tips": [
                "Stay hydrated by drinking warm fluids.",
                "Rest to help your body recover.",
                "Inhale steam to relieve nasal congestion.",
                "Eat vitamin C rich foods like oranges.",
                "Avoid cold exposure.",
                "Use saline nasal drops for blocked nose.",
                "Wash hands frequently to prevent infection.",
                "Avoid dusty or polluted environments.",
                "Light exercise can help if fever-free.",
                "⚠️ Disclaimer: Seek doctor if cold persists more than 2 weeks or worsens."
            ]
        },
        "stomach_pain": {
            "title": "🍲 Stomach Pain Relief",
            "tips": [
                "Drink warm water to relax stomach muscles.",
                "Eat light meals like rice or soup.",
                "Avoid spicy and oily foods.",
                "Use a heating pad for relief.",
                "Seek medical attention if pain is severe.",
                "Eat small meals throughout the day.",
                "Avoid lying down immediately after eating.",
                "Sip ginger or peppermint tea.",
                "Stay hydrated with water and clear soups.",
                "⚠️ Disclaimer: Severe or persistent stomach pain should be checked by a doctor."
            ]
        },
        "chest_pain": {
            "title": "❤️ Chest Pain Guidance",
            "tips": [
                "Stop any activity and rest immediately.",
                "Breathe slowly and deeply to reduce discomfort.",
                "Avoid heavy meals and alcohol.",
                "Call emergency services if pain is sharp or severe.",
                "Consult a doctor even for mild recurring pain.",
                "Sit upright to relieve pressure on chest.",
                "Relax muscles using gentle stretching.",
                "Avoid smoking and caffeine.",
                "Keep a record of pain episodes to share with doctor.",
                "⚠️ Disclaimer: Chest pain can indicate heart issues — seek immediate help if severe."
            ]
        },
        "stress": {
            "title": "🧘 Stress Management",
            "tips": [
                "Practice deep breathing exercises.",
                "Take short breaks during work.",
                "Sleep 7–8 hours to rejuvenate mind.",
                "Avoid excessive caffeine or work stress.",
                "⚠️ Disclaimer: Persistent stress affecting daily life should be discussed with a mental health professional."
            ]
        },
        "wellness": {
            "title": "🌱 Wellness Tips",
            "tips": [
                "Start your day with a glass of water.",
                "Eat more fruits and vegetables daily.",
                "Exercise for at least 30 minutes most days.",
                "Sleep 7–8 hours every night.",
                "Take regular breaks from screens.",
                "Spend time outdoors in sunlight.",
                "Avoid smoking and limit alcohol.",
                "Maintain positive mindset.",
                "Drink sufficient water throughout the day.",
                "⚠️ Disclaimer: General wellness tips are not a substitute for professional medical advice."
            ]
        },
        "greeting": {
            "title": "👋 Greeting",
            "tips": ["Hello! 👋 How can I help you with your health today?"]
        },
        "goodbye": {
            "title": "👋 Goodbye",
            "tips": ["Goodbye! Stay safe and take care 😊"]
        }
    }

# Populate knowledge_base table if empty
c.execute("SELECT COUNT(*) FROM knowledge_base")
if c.fetchone()[0] == 0:
    kb = default_kb()
    for intent, data in kb.items():
        c.execute("INSERT OR REPLACE INTO knowledge_base (intent, title, tips_json) VALUES (?,?,?)",
                  (intent, data['title'], json.dumps(data['tips'], ensure_ascii=False)))
    conn.commit()

# -----------------------------
# Load model (same as your original)
# -----------------------------
if not os.path.exists("intent_model.pkl"):
    texts = [
        "I have a headache", "My head is hurting",
        "I feel feverish", "My temperature is high",
        "I cut my hand", "There is bleeding on my finger",
        "I have a bad cough", "My throat hurts",
        "I have cold and runny nose", "My nose is blocked",
        "My stomach hurts", "I feel chest pain",
        "I am stressed out", "I am anxious",
        "I want wellness tips", "Give me health advice",
        "Hello", "Hi there", "Goodbye", "Bye"
    ]
    labels = [
        "headache", "headache", "fever", "fever",
        "cut", "cut", "cough", "cough",
        "cold", "cold", "stomach_pain", "chest_pain",
        "stress", "stress", "wellness", "wellness",
        "greeting", "greeting", "goodbye", "goodbye"
    ]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression()
    model.fit(X, labels)
    joblib.dump((vectorizer, model), "intent_model.pkl")
else:
    vectorizer, model = joblib.load("intent_model.pkl")

# -----------------------------
# Helpers for DB-backed knowledge base
# -----------------------------
def load_kb_from_db():
    c.execute("SELECT intent, title, tips_json FROM knowledge_base")
    rows = c.fetchall()
    kb = {}
    for intent, title, tips_json in rows:
        kb[intent] = {"title": title, "tips": json.loads(tips_json)}
    return kb

def save_kb_intent(intent, title, tips_list):
    c.execute("INSERT OR REPLACE INTO knowledge_base (intent, title, tips_json) VALUES (?,?,?)",
              (intent, title, json.dumps(tips_list, ensure_ascii=False)))
    conn.commit()

def delete_kb_intent(intent):
    c.execute("DELETE FROM knowledge_base WHERE intent=?", (intent,))
    conn.commit()

# -----------------------------
# Intent detection (improved with confidence)
# -----------------------------
def detect_intents(user_input):
    """
    Returns a list of tuples: (intent, confidence, matched_via_keyword_bool)
    - For keyword matches: confidence = 1.0, matched_via_keyword_bool = True
    - For model predictions: confidence = model_prob, matched_via_keyword_bool = False
    If model confidence < CONF_THRESHOLD, returns intent 'unknown'
    """
    try:
        lang = detect(user_input)
        if lang == 'hi':
            user_input_en = translator.translate(user_input, src='hi', dest='en').text.lower()
        else:
            user_input_en = user_input.lower()
    except:
        user_input_en = user_input.lower()
    
    symptoms_list = [x.strip() for x in user_input_en.replace(',', ' and ').split('and') if x.strip()]
    detected = []
    
    keywords = {
        "fever": ["fever", "temperature", "high fever", "cold body"],
        "headache": ["headache", "head hurts", "migraine"],
        "cut": ["cut", "wound", "bleeding"],
        "cough": ["cough", "throat pain", "dry cough"],
        "cold": ["cold", "sneeze", "runny nose"],
        "stomach_pain": ["stomach pain", "belly ache", "gas"],
        "chest_pain": ["chest pain", "heart pain", "tight chest"],
        "stress": ["stress", "anxiety", "tension"]
    }
    
    for sym in symptoms_list:
        found = False
        for intent, words in keywords.items():
            if any(word in sym for word in words):
                detected.append((intent, 1.0, True))
                found = True
                break
        if not found:
            # model fallback with confidence
            try:
                X = vectorizer.transform([sym])
                probs = model.predict_proba(X)[0]
                classes = model.classes_
                best_idx = int(probs.argmax())
                best_intent = classes[best_idx]
                best_prob = float(probs[best_idx])
                if best_prob >= CONF_THRESHOLD:
                    detected.append((best_intent, best_prob, False))
                else:
                    detected.append(("unknown", best_prob, False))
            except Exception:
                detected.append(("unknown", 0.0, False))
    
    return detected

def extract_symptoms_bodyparts(user_input):
    if nlp is None:
        return [], []
    doc = nlp(user_input)
    symptoms, body_parts = [], []
    for token in doc:
        if token.text.lower() in ["pain", "fever", "cold", "cough", "stress", "headache"]:
            symptoms.append(token.text)
        if token.text.lower() in ["head", "stomach", "chest", "hand", "throat", "nose"]:
            body_parts.append(token.text)
    return symptoms, body_parts

# -----------------------------
# JWT Functions
# -----------------------------
def create_token(email):
    payload = {"email": email, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=12)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload['email']
    except:
        return None

# -----------------------------
# Logging queries & feedback
# -----------------------------
def log_query(email, query_text, detected_intent):
    ts = datetime.datetime.utcnow().isoformat()
    c.execute("INSERT INTO queries (email, query_text, detected_intent, timestamp) VALUES (?,?,?,?)",
              (email, query_text, detected_intent, ts))
    conn.commit()
    return c.lastrowid

def store_feedback(email, query_id, intent, feedback_choice, comment):
    ts = datetime.datetime.utcnow().isoformat()
    c.execute("INSERT INTO feedback (email, query_id, intent, feedback, comment, timestamp) VALUES (?,?,?,?,?,?)",
              (email, query_id, intent, feedback_choice, comment, ts))
    conn.commit()

# -----------------------------
# Streamlit Config & Session state
# -----------------------------
st.set_page_config(page_title="Wellness Chatbot", page_icon="💬", layout="centered")

# Ensure spaCy model is loaded
if nlp is None:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("⚠️ spaCy model 'en_core_web_sm' not found. Please run: `python -m spacy download en_core_web_sm`")
        st.stop()

if "page" not in st.session_state: st.session_state.page = "signup"
if "token" not in st.session_state: st.session_state.token = None
if "history" not in st.session_state: st.session_state.history = []
# NEW: persistent flag to show admin dashboard across reruns
if "show_admin" not in st.session_state: st.session_state.show_admin = False

st.markdown("<h2 style='text-align:center;color:#00bfff;'>💬 Wellness Guide Chatbot</h2>", unsafe_allow_html=True)

# -----------------------------
# Chatbot Response Generator (now DB-backed KB)
# -----------------------------
def get_response(user_input, user_lang="English", user_email=None):
    kb = load_kb_from_db()
    intents_info = detect_intents(user_input)  # list of (intent, conf, matched_keyword_bool)
    symptoms, body_parts = extract_symptoms_bodyparts(user_input)
    responses = []
    primary_intent = intents_info[0][0] if intents_info else 'wellness'
    
    # log query (store primary intent label, even 'unknown')
    query_id = None
    if user_email:
        query_id = log_query(user_email, user_input, primary_intent)
    
    for intent, conf, matched_keyword in intents_info:
        # If intent unknown or not present in KB -> fallback message
        if intent == "unknown" or intent not in kb:
            title = "❓ Unknown Query"
            resp_list = ["The query you are searching is not in our knowledge base. We will update it soon!"]
        else:
            title = kb[intent]['title']
            resp_list = kb[intent]['tips']
        
        responses.append(f"### {title}")
        for tip in resp_list:
            responses.append(tip)
    
    if symptoms:
        responses.append(f"**Detected Symptom(s):** {', '.join(symptoms)}")
    if body_parts:
        responses.append(f"**Affected Body Part(s):** {', '.join(body_parts)}")
    
    final_response = "\n\n".join(responses)
    
    if user_lang == "Hindi":
        try:
            final_response = translator.translate(final_response, src='en', dest='hi').text
        except:
            pass
    
    return final_response, primary_intent, query_id

# -----------------------------
# Sidebar Profile
# -----------------------------
def sidebar_profile(email):
    with st.sidebar:
        st.markdown("## 👤 Profile Settings")
        c.execute("SELECT name, age, gender, language FROM users WHERE email=?", (email,))
        user_data = c.fetchone()
        if user_data:
            new_name = st.text_input("Name", user_data[0])
            new_age = st.number_input("Age", min_value=1, max_value=120, value=user_data[1])
            new_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male","Female","Other"].index(user_data[2]))
            new_language = st.selectbox("Language", ["English", "Hindi"], index=["English","Hindi"].index(user_data[3]))
            if st.button("Update Profile"):
                c.execute("UPDATE users SET name=?, age=?, gender=?, language=? WHERE email=?",
                          (new_name, new_age, new_gender, new_language, email))
                conn.commit()
                st.success("Profile updated!")
        else:
            st.info("No profile data found. Sign up to store your profile.")
        
        st.markdown("---")
        st.markdown("### 🕒 Chat History")
        for chat in st.session_state.history:
            st.markdown(f"**You:** {chat['user']}\n**Bot:** {chat['bot']}")
        if st.button("Clear Chat History"):
            st.session_state.history = []
            st.success("Chat history cleared!")
        
        st.markdown("---")
        if st.button("Logout"):
            st.session_state.page = "login"
            st.session_state.token = None
            st.session_state.history = []
            # Also hide admin when logging out
            st.session_state.show_admin = False

# -----------------------------
# Admin Dashboard
# -----------------------------
def admin_dashboard():
    email = verify_token(st.session_state.token)
    if email != ADMIN_EMAIL:
        st.error("Unauthorized. Admin access only.")
        return
    
    st.title("🛠️ Admin Dashboard ")
    
    kb = load_kb_from_db()
    
    # Overview metrics
    total_users = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_queries = c.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
    positive = c.execute("SELECT COUNT(*) FROM feedback WHERE feedback='up'").fetchone()[0]
    negative = c.execute("SELECT COUNT(*) FROM feedback WHERE feedback='down'").fetchone()[0]
    feedback_pct = f"{(positive/(positive+negative)*100):.1f}%" if (positive+negative)>0 else "N/A"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", total_users)
    col2.metric("Total Queries", total_queries)
    col3.metric("Positive Feedback", feedback_pct)
    
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Knowledge Base", "Feedback", "Queries", "Users"])
    
    with tab1:
        st.header("Knowledge Base Manager")
        intents = list(kb.keys())
        selected = st.selectbox("Select intent to edit or choose <new> to add", ["<new>"] + intents)
        
        if selected == "<new>":
            new_intent = st.text_input("Intent key (no spaces, e.g., new_symptom)")
            new_title = st.text_input("Display title")
            new_tips = st.text_area("Tips (one per line)")
            if st.button("Add Intent"):
                if new_intent and new_title and new_tips:
                    tips_list = [t.strip() for t in new_tips.splitlines() if t.strip()]
                    save_kb_intent(new_intent, new_title, tips_list)
                    st.success(f"Intent '{new_intent}' added.")
                    # refresh local kb var (helps UI reflect changes quickly)
                    kb = load_kb_from_db()
                else:
                    st.error("Please provide intent key, title and tips.")
        else:
            data = kb[selected]
            st.subheader(f"Editing: {selected}")
            title_in = st.text_input("Title", value=data['title'])
            tips_in = st.text_area("Tips (one per line)", value="\n".join(data['tips']))
            if st.button("Save Changes"):
                tips_list = [t.strip() for t in tips_in.splitlines() if t.strip()]
                save_kb_intent(selected, title_in, tips_list)
                st.success("Saved.")
                kb = load_kb_from_db()
            if st.button("Delete Intent"):
                delete_kb_intent(selected)
                st.success("Deleted intent.")
                kb = load_kb_from_db()
    
    with tab2:
        st.header("Feedback")
        # show aggregated feedback counts and a bar chart
        fb_df = pd.read_sql_query("SELECT feedback, COUNT(*) as count FROM feedback GROUP BY feedback", conn)
        if not fb_df.empty:
            st.plotly_chart(px.bar(fb_df, x="feedback", y="count", title="Feedback Counts"))
        
        rows = c.execute("SELECT id, email, query_id, intent, feedback, comment, timestamp FROM feedback ORDER BY timestamp DESC LIMIT 200").fetchall()
        for r in rows:
            fid, email_f, qid, intent_f, fb, comment, ts = r
            st.markdown(f"**{email_f or 'anon'}** — {intent_f} — {ts}")
            st.write(f"Feedback: {fb}")
            if comment:
                st.write(f"Comment: {comment}")
            st.write("---")
        
        # Recent feedback comments table
        fb_comments = pd.read_sql_query("SELECT intent, feedback, comment, timestamp FROM feedback WHERE comment != '' ORDER BY timestamp DESC LIMIT 15", conn)
        if not fb_comments.empty:
            st.subheader("Recent Feedback Comments")
            st.dataframe(fb_comments)
    
    with tab3:
        st.header("Recent Queries & Analytics")
        # Queries over time line chart
        q_over_time = pd.read_sql_query("SELECT date(timestamp) as day, COUNT(*) as queries FROM queries GROUP BY day ORDER BY day", conn)
        if not q_over_time.empty:
            fig = px.line(q_over_time, x="day", y="queries", title="Queries Over Time")
            st.plotly_chart(fig)
        
        # Intent distribution pie chart
        intent_dist = pd.read_sql_query("SELECT detected_intent, COUNT(*) as count FROM queries GROUP BY detected_intent", conn)
        if not intent_dist.empty:
            st.plotly_chart(px.pie(intent_dist, names="detected_intent", values="count", title="Query Intent Distribution"))
        
        # Show recent queries
        qrows = c.execute("SELECT id, email, query_text, detected_intent, timestamp FROM queries ORDER BY timestamp DESC LIMIT 200").fetchall()
        for q in qrows:
            qid, qemail, qtext, dintent, qts = q
            st.markdown(f"**{qemail or 'anon'}** — {dintent} — {qts}")
            st.write(qtext)
            st.write("---")
        
        # Export queries CSV
        if st.button("Export Queries CSV"):
            df_q = pd.read_sql_query("SELECT * FROM queries ORDER BY timestamp DESC", conn)
            csv = df_q.to_csv(index=False)
            st.download_button("Download queries.csv", data=csv, file_name="queries.csv", mime="text/csv")
    
    with tab4:
        st.header("Registered Users")
        users = c.execute("SELECT id, name, email, age, gender, language FROM users ORDER BY id DESC LIMIT 500").fetchall()
        for u in users:
            uid, name, uemail, age, gender, lang = u
            st.write(f"{name} | {uemail} | {age} | {gender} | {lang}")
        
        # users table and basic demographics chart
        user_df = pd.read_sql_query("SELECT age, gender, language FROM users", conn)
        if not user_df.empty:
            try:
                age_hist = px.histogram(user_df, x="age", nbins=10, title="User Age Distribution")
                st.plotly_chart(age_hist)
            except Exception:
                pass
    
    st.markdown("---")
    st.caption("Admin: Use the Knowledge Base tab to edit intents shown to users. Feedback & queries are logged automatically.")

# -----------------------------
# Chatbot Page for normal users
# -----------------------------
def chatbot():
    email = verify_token(st.session_state.token)
    if not email:
        st.error("Session expired. Please login again.")
        st.session_state.page = "login"
        return
    
    # Sidebar: admin dashboard toggle and profile
    # For admin user, show open/close buttons that persist via session_state.show_admin
    if email == ADMIN_EMAIL:
        with st.sidebar:
            if not st.session_state.show_admin:
                if st.button("Open Admin Dashboard"):
                    st.session_state.show_admin = True
            else:
                if st.button("Close Admin Dashboard"):
                    st.session_state.show_admin = False
        
        # if admin requested admin dashboard, render it persistently
        if st.session_state.show_admin:
            admin_dashboard()
            # allow admin to also use chatbot features below (dashboard is shown above)
            # Note: admin_dashboard draws the main page content already; we continue so the chatbot UI and sidebar profile exist
    
    sidebar_profile(email)
    
    c.execute("SELECT language FROM users WHERE email=?", (email,))
    row = c.fetchone()
    user_lang = row[0] if row else "English"
    
    user_input = st.chat_input("Type your health query...")
    if user_input:
        bot_reply, primary_intent, query_id = get_response(user_input, user_lang, user_email=email)
        st.session_state.history.append({"user": user_input, "bot": bot_reply, "intent": primary_intent, "query_id": query_id})
    
    for i, chat in enumerate(st.session_state.history):
        st.markdown(
            f"<div style='text-align:right;'><span style='background-color:#4CAF50;color:white;"
            f"padding:8px 12px;border-radius:15px;display:inline-block;margin:4px;'>{chat['user']}</span></div>",
            unsafe_allow_html=True
        )
        bot_lines = chat['bot'].split("\n\n")
        for line in bot_lines:
            if line.strip().startswith("⚠️"):
                st.markdown(
                    f"<div style='text-align:left;'><span style='background-color:#e74c3c;color:white;"
                    f"padding:8px 12px;border-radius:15px;display:inline-block;margin:4px;font-weight:bold;'>{line}</span></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='text-align:left;'><span style='background-color:#333;color:white;"
                    f"padding:8px 12px;border-radius:15px;display:inline-block;margin:4px;'>{line}</span></div>",
                    unsafe_allow_html=True
                )
        
        # Feedback UI under each bot response
        with st.container():
            st.write("")
            col1, col2, col3 = st.columns([1,1,6])
            up_key = f"up_{i}"
            down_key = f"down_{i}"
            comment_key = f"comment_{i}"
            
            if st.button("👍", key=up_key):
                store_feedback(email, chat.get('query_id'), chat.get('intent'), 'up', '')
                st.success("Thanks for your feedback!")
            
            if st.button("👎", key=down_key):
                # ---------- ensure negative feedback saved even if no comment ----------
                comment = st.text_input("Tell us what went wrong:", key=comment_key)
                # If user enters a comment (non-empty), save it; otherwise save "No comment"
                if comment is not None and comment.strip() != "":
                    store_feedback(email, chat.get('query_id'), chat.get('intent'), 'down', comment.strip())
                else:
                    store_feedback(email, chat.get('query_id'), chat.get('intent'), 'down', "No comment")
                st.success("Thanks — your feedback is recorded!")
                # --------------------------------------------------------------------------------

# -----------------------------
# Signup/Login Routing
# -----------------------------
def signup():
    st.subheader("📝 Sign Up")
    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=1, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    language = st.selectbox("Preferred Language", ["English", "Hindi"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Sign Up"):
        if name and email and password:
            hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
            try:
                c.execute("INSERT INTO users (name, age, gender, language, email, password) VALUES (?,?,?,?,?,?)",
                          (name, age, gender, language, email, hashed))
                conn.commit()
                st.success("Account created! Logging in automatically...")
                st.session_state.token = create_token(email)
                st.session_state.page = "chatbot"
            except Exception as e:
                st.error("Email already registered or DB error.")
        else:
            st.error("Please fill all fields.")
    
    if st.button("Already have an account? Login"):
        st.session_state.page = "login"

def login():
    st.subheader("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # admin special-case
        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            # ensure admin exists in users table for profile features
            try:
                hashed = bcrypt.hashpw(ADMIN_PASSWORD.encode(), bcrypt.gensalt())
                c.execute("INSERT OR IGNORE INTO users (name, age, gender, language, email, password) VALUES (?,?,?,?,?,?)",
                          ("Admin", 30, "Other", "English", ADMIN_EMAIL, hashed))
                conn.commit()
            except:
                pass
            st.session_state.token = create_token(email)
            st.session_state.page = "chatbot"
            st.success("Admin logged in.")
            return
        
        c.execute("SELECT password FROM users WHERE email=?", (email,))
        row = c.fetchone()
        if row and bcrypt.checkpw(password.encode(), row[0]):
            st.session_state.token = create_token(email)
            st.session_state.page = "chatbot"
            st.success("Logged in successfully!")
        else:
            st.error("Invalid email or password.")
    
    if st.button("New user? Sign Up"):
        st.session_state.page = "signup"

# -----------------------------
# Page Routing
# -----------------------------
if st.session_state.page == "signup":
    signup()
elif st.session_state.page == "login":
    login()
elif st.session_state.page == "chatbot":
    chatbot()


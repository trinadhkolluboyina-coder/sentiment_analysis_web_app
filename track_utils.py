import sqlite3

# ================================
# Database Connection
# ================================
conn = sqlite3.connect("emotion_data.db", check_same_thread=False)
c = conn.cursor()

# ================================
# Page Tracking
# ================================
def create_page_visited_table():
    c.execute("""
        CREATE TABLE IF NOT EXISTS page_visited(
            pagename TEXT,
            time_of_visit TEXT
        )
    """)
    conn.commit()

def add_page_visited_details(pagename, time_of_visit):
    c.execute(
        "INSERT INTO page_visited VALUES (?,?)",
        (pagename, str(time_of_visit))
    )
    conn.commit()

def view_all_page_visited_details():
    c.execute("SELECT * FROM page_visited")
    return c.fetchall()

# ================================
# Emotion Classifier Tracking
# ================================
def create_emotionclf_table():
    c.execute("""
        CREATE TABLE IF NOT EXISTS emotion_prediction(
            rawtext TEXT,
            prediction TEXT,
            probability REAL,
            time_of_visit TEXT
        )
    """)
    conn.commit()

def add_prediction_details(rawtext, prediction, probability, time_of_visit):
    c.execute(
        "INSERT INTO emotion_prediction VALUES (?,?,?,?)",
        (rawtext, prediction, float(probability), str(time_of_visit))
    )
    conn.commit()

def view_all_prediction_details():
    c.execute("SELECT * FROM emotion_prediction")
    return c.fetchall()

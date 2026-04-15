import sqlite3
import json
from datetime import datetime
from pathlib import Path
from sagan.config import config

DB_PATH = config.home_dir / "sagan.db"

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_id TEXT,
            tickers TEXT,
            action TEXT,
            confidence REAL,
            conflict BOOLEAN,
            justification TEXT,
            metadata TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_action(model_id, tickers, action, confidence, conflict, justification, extra_meta=None):
    if not isinstance(tickers, str):
        tickers = ",".join(tickers)
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_logs 
        (timestamp, model_id, tickers, action, confidence, conflict, justification, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        model_id,
        tickers,
        action,
        confidence,
        conflict,
        justification,
        json.dumps(extra_meta or {})
    ))
    conn.commit()
    conn.close()

def get_logs(limit=20):
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    logs = cursor.execute("SELECT * FROM user_logs ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(log) for log in logs]

# Initialize on import
init_db()

import sqlite3
import numpy as np
from datetime import datetime

DB_PATH = 'attendance.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        student_id TEXT UNIQUE NOT NULL,
        class_name TEXT,
        embedding BLOB,
        created_at TEXT DEFAULT (datetime('now','localtime'))
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT,
        name TEXT,
        class_name TEXT,
        status TEXT DEFAULT 'đúng giờ',
        timestamp TEXT DEFAULT (datetime('now','localtime')),
        date TEXT
    )''')
    # Migration: thêm cột status nếu chưa có (database cũ)
    try:
        c.execute("ALTER TABLE attendance ADD COLUMN status TEXT DEFAULT 'đúng giờ'")
        conn.commit()
    except Exception:
        pass  # Cột đã tồn tại
    conn.commit()
    conn.close()

def save_student(name, student_id, class_name, embedding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    emb_bytes = embedding.astype(np.float32).tobytes()
    try:
        c.execute('INSERT INTO students (name, student_id, class_name, embedding) VALUES (?,?,?,?)',
                  (name, student_id, class_name, emb_bytes))
        conn.commit()
        return True, "Đăng ký thành công"
    except sqlite3.IntegrityError:
        c.execute('UPDATE students SET name=?, class_name=?, embedding=? WHERE student_id=?',
                  (name, class_name, emb_bytes, student_id))
        conn.commit()
        return True, "Cập nhật thành công"
    finally:
        conn.close()

def get_all_students():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT name, student_id, class_name, embedding FROM students WHERE embedding IS NOT NULL')
    rows = c.fetchall()
    conn.close()
    students = []
    for row in rows:
        emb = np.frombuffer(row[3], dtype=np.float32).copy()
        students.append({'name': row[0], 'student_id': row[1],
                         'class_name': row[2], 'embedding': emb})
    return students

def get_students_list():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT name, student_id, class_name, created_at FROM students ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()
    return [{'name': r[0], 'student_id': r[1], 'class_name': r[2], 'created_at': r[3]} for r in rows]

def save_attendance(student_id, name, class_name, status='đúng giờ'):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    date_str = datetime.now().strftime('%Y-%m-%d')
    c.execute('SELECT id FROM attendance WHERE student_id=? AND date=?', (student_id, date_str))
    if c.fetchone():
        conn.close()
        return False, "Đã điểm danh hôm nay rồi"
    c.execute('INSERT INTO attendance (student_id, name, class_name, status, date) VALUES (?,?,?,?,?)',
              (student_id, name, class_name, status, date_str))
    conn.commit()
    conn.close()
    return True, "Điểm danh thành công"

def get_attendance_history(date=None, student_id=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    query = 'SELECT name, student_id, class_name, status, timestamp, date FROM attendance'
    params, conditions = [], []
    if date:
        conditions.append('date=?'); params.append(date)
    if student_id:
        conditions.append('student_id=?'); params.append(student_id)
    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)
    query += ' ORDER BY timestamp DESC LIMIT 200'
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return [{'name': r[0], 'student_id': r[1], 'class_name': r[2],
             'status': r[3], 'timestamp': r[4], 'date': r[5]} for r in rows]

def delete_student(student_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM students WHERE student_id=?', (student_id,))
    conn.commit()
    conn.close()

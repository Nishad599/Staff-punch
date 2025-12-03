"""
Punch In/Punch Out Attendance System with Face Recognition
For Third-Party Staff Management
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import sqlite3
import os
import base64
import io
import numpy as np
from datetime import datetime, timedelta, timezone, date
from PIL import Image
import cv2
import pickle
import json

# IST Timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def get_ist_now():
    """Get current time in IST"""
    return datetime.now(IST)

# Initialize FastAPI app
app = FastAPI(title="Punch Attendance System", version="2.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
STANDARD_WORKING_HOURS = 9      # Full day = 9 hours
OVERTIME_THRESHOLD = 9.5        # Overtime starts after 9.5 hours
HALF_DAY_MIN_HOURS = 4          # Minimum hours for half day
FULL_DAY_MIN_HOURS = 9          # Minimum hours for full day
DATABASE_PATH = "punch_attendance.db"

# Role-based working hours configuration
ROLE_CONFIG = {
    "housekeeping": {
        "full_day_hours": 9,
        "overtime_threshold": 9.5,
        "half_day_hours": 4.5
    },
    "security": {
        "full_day_hours": 8,
        "overtime_threshold": 8.5,
        "half_day_hours": 4
    }
}

# Designation categories (skill levels)
DESIGNATIONS = ["High-skilled", "Skilled", "Semi-skilled", "Unskilled"]

# ============== Indian National Holidays ==============
INDIAN_HOLIDAYS = {
    # 2024 Holidays
    "2024-01-26": "Republic Day",
    "2024-03-25": "Holi",
    "2024-03-29": "Good Friday",
    "2024-04-11": "Idul Fitr (Eid)",
    "2024-04-14": "Dr. Ambedkar Jayanti",
    "2024-04-17": "Ram Navami",
    "2024-04-21": "Mahavir Jayanti",
    "2024-05-23": "Buddha Purnima",
    "2024-06-17": "Eid ul-Adha (Bakrid)",
    "2024-07-17": "Muharram",
    "2024-08-15": "Independence Day",
    "2024-08-26": "Janmashtami",
    "2024-09-16": "Milad un-Nabi",
    "2024-10-02": "Gandhi Jayanti",
    "2024-10-12": "Dussehra",
    "2024-10-31": "Diwali (Laxmi Puja)",
    "2024-11-01": "Diwali",
    "2024-11-15": "Guru Nanak Jayanti",
    "2024-12-25": "Christmas",
    
    # 2025 Holidays
    "2025-01-26": "Republic Day",
    "2025-03-14": "Holi",
    "2025-03-31": "Idul Fitr (Eid)",
    "2025-04-10": "Mahavir Jayanti",
    "2025-04-14": "Dr. Ambedkar Jayanti",
    "2025-04-18": "Good Friday",
    "2025-05-12": "Buddha Purnima",
    "2025-06-07": "Eid ul-Adha (Bakrid)",
    "2025-07-06": "Muharram",
    "2025-08-15": "Independence Day",
    "2025-08-16": "Janmashtami",
    "2025-09-05": "Milad un-Nabi",
    "2025-10-02": "Gandhi Jayanti",
    "2025-10-21": "Dussehra",
    "2025-10-20": "Diwali",
    "2025-11-05": "Guru Nanak Jayanti",
    "2025-12-25": "Christmas",
    
    # 2026 Holidays
    "2026-01-26": "Republic Day",
    "2026-03-03": "Holi",
    "2026-03-20": "Idul Fitr (Eid)",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Dr. Ambedkar Jayanti",
    "2026-05-01": "Buddha Purnima",
    "2026-05-27": "Eid ul-Adha (Bakrid)",
    "2026-08-15": "Independence Day",
    "2026-10-02": "Gandhi Jayanti",
    "2026-12-25": "Christmas",
}

def is_holiday(date_str: str) -> tuple:
    """Check if a date is a holiday (Sunday or National Holiday)"""
    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
    
    # Check if Sunday (weekday() returns 6 for Sunday)
    if date_obj.weekday() == 6:
        return True, "Sunday"
    
    # Check if National Holiday
    if date_str in INDIAN_HOLIDAYS:
        return True, INDIAN_HOLIDAYS[date_str]
    
    return False, None

def get_working_days_in_month(year: int, month: int) -> dict:
    """Get working days, holidays, and Sundays in a month"""
    import calendar
    
    first_day = date(year, month, 1)
    last_day = date(year, month, calendar.monthrange(year, month)[1])
    
    working_days = 0
    sundays = 0
    national_holidays = 0
    holiday_list = []
    
    current = first_day
    while current <= last_day:
        date_str = current.strftime('%Y-%m-%d')
        is_hol, hol_name = is_holiday(date_str)
        
        if current.weekday() == 6:  # Sunday
            sundays += 1
            holiday_list.append({"date": date_str, "name": "Sunday"})
        elif date_str in INDIAN_HOLIDAYS:
            national_holidays += 1
            holiday_list.append({"date": date_str, "name": INDIAN_HOLIDAYS[date_str]})
        else:
            working_days += 1
        
        current += timedelta(days=1)
    
    return {
        "total_days": (last_day - first_day).days + 1,
        "working_days": working_days,
        "sundays": sundays,
        "national_holidays": national_holidays,
        "holidays": holiday_list
    }

# ============== Face Recognition Setup ==============
FACE_RECOGNITION_AVAILABLE = False
asian_face_recognizer = None

try:
    import insightface
    from insightface.app import FaceAnalysis
    
    class FaceRecognitionSystem:
        def __init__(self):
            self.insight_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.insight_app.prepare(ctx_id=0, det_size=(640, 640))
            self.embedding_dim = 512
            self.known_face_encodings = []
            self.known_face_ids = []
            self.known_face_names = []
            print("✅ Face Recognition System Initialized (buffalo_l 512D)")
            self.load_known_faces()
        
        def detect_and_encode(self, frame):
            """Detect face and return 512D embedding"""
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = frame
            
            results = self.insight_app.get(bgr_frame)
            
            if len(results) == 0:
                return None, None
            
            face = results[0]
            bbox = face.bbox.astype(int)
            location = (bbox[1], bbox[2], bbox[3], bbox[0])
            
            if hasattr(face, 'embedding') and len(face.embedding) == self.embedding_dim:
                return face.embedding.astype(np.float64), location
            
            return None, None
        
        def compare_faces(self, face_embedding, threshold=0.5):
            """Compare face with known faces, return best match"""
            if len(self.known_face_encodings) == 0:
                return None, None, 0.0
            
            face_norm = face_embedding / np.linalg.norm(face_embedding)
            
            best_match_idx = -1
            best_similarity = 0.0
            
            for i, known_encoding in enumerate(self.known_face_encodings):
                known_norm = known_encoding / np.linalg.norm(known_encoding)
                similarity = np.dot(face_norm, known_norm)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = i
            
            if best_similarity > threshold and best_match_idx >= 0:
                return self.known_face_ids[best_match_idx], self.known_face_names[best_match_idx], best_similarity
            
            return None, None, best_similarity
        
        def save_face_encoding(self, emp_id, emp_name, encoding):
            """Save face encoding to file"""
            os.makedirs('face_encodings', exist_ok=True)
            
            face_data = {
                'emp_id': emp_id,
                'emp_name': emp_name,
                'encoding': encoding,
                'timestamp': get_ist_now().isoformat()
            }
            
            filepath = f"face_encodings/face_{emp_id}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(face_data, f)
            
            self.known_face_encodings.append(encoding)
            self.known_face_ids.append(emp_id)
            self.known_face_names.append(emp_name)
            
            print(f"💾 Saved face encoding for {emp_name} ({emp_id})")
        
        def load_known_faces(self):
            """Load all known face encodings"""
            if not os.path.exists('face_encodings'):
                return
            
            for filename in os.listdir('face_encodings'):
                if filename.endswith('.pkl'):
                    filepath = os.path.join('face_encodings', filename)
                    try:
                        with open(filepath, 'rb') as f:
                            face_data = pickle.load(f)
                        self.known_face_encodings.append(face_data['encoding'])
                        self.known_face_ids.append(face_data['emp_id'])
                        self.known_face_names.append(face_data['emp_name'])
                    except Exception as e:
                        print(f"⚠️ Could not load {filename}: {e}")
            
            print(f"📚 Loaded {len(self.known_face_encodings)} known faces")
        
        def delete_face_encoding(self, emp_id):
            """Delete face encoding for an employee"""
            filepath = f"face_encodings/face_{emp_id}.pkl"
            if os.path.exists(filepath):
                os.remove(filepath)
            
            if emp_id in self.known_face_ids:
                idx = self.known_face_ids.index(emp_id)
                self.known_face_encodings.pop(idx)
                self.known_face_ids.pop(idx)
                self.known_face_names.pop(idx)
    
    asian_face_recognizer = FaceRecognitionSystem()
    FACE_RECOGNITION_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️ InsightFace not available: {e}")
    print("Install with: pip install insightface onnxruntime")

# ============== Database Setup ==============
def init_database():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Employees table (with role and designation)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            emp_id TEXT PRIMARY KEY,
            emp_name TEXT NOT NULL,
            phone TEXT,
            role TEXT DEFAULT 'housekeeping',
            designation TEXT DEFAULT 'Unskilled',
            face_registered INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Add role and designation columns if they don't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE employees ADD COLUMN role TEXT DEFAULT 'housekeeping'")
    except:
        pass
    
    try:
        cursor.execute("ALTER TABLE employees ADD COLUMN designation TEXT DEFAULT 'Unskilled'")
    except:
        pass
    
    # Punch records table (updated with day_type and attendance_type)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS punch_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_id TEXT NOT NULL,
            punch_date DATE NOT NULL,
            punch_in_time TIMESTAMP,
            punch_out_time TIMESTAMP,
            total_hours REAL DEFAULT 0,
            regular_hours REAL DEFAULT 0,
            overtime_hours REAL DEFAULT 0,
            day_type TEXT DEFAULT 'working',
            attendance_type TEXT DEFAULT 'absent',
            status TEXT DEFAULT 'incomplete',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (emp_id) REFERENCES employees(emp_id),
            UNIQUE(emp_id, punch_date)
        )
    ''')
    
    # Add new columns if they don't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE punch_records ADD COLUMN day_type TEXT DEFAULT 'working'")
    except:
        pass
    
    try:
        cursor.execute("ALTER TABLE punch_records ADD COLUMN attendance_type TEXT DEFAULT 'absent'")
    except:
        pass
    
    # Holidays table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS holidays (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            holiday_date DATE UNIQUE NOT NULL,
            holiday_name TEXT NOT NULL,
            holiday_type TEXT DEFAULT 'national',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert Indian holidays if not exists
    for date_str, name in INDIAN_HOLIDAYS.items():
        try:
            cursor.execute("INSERT OR IGNORE INTO holidays (holiday_date, holiday_name, holiday_type) VALUES (?, ?, ?)",
                          (date_str, name, 'national'))
        except:
            pass
    
    # Admin users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'admin',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute("SELECT COUNT(*) FROM admin_users WHERE username = 'admin'")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO admin_users (username, password, role) VALUES (?, ?, ?)",
                      ('admin', 'admin123', 'admin'))
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

init_database()

# ============== Pydantic Models ==============
class EmployeeRegistration(BaseModel):
    emp_id: str
    emp_name: str
    phone: str
    role: str = "housekeeping"      # housekeeping or security
    designation: str = "Unskilled"   # High-skilled, Skilled, Semi-skilled, Unskilled
    face_image: str

class PunchRequest(BaseModel):
    image_data: str

class AdminLogin(BaseModel):
    username: str
    password: str

class HolidayRequest(BaseModel):
    holiday_date: str
    holiday_name: str
    holiday_type: str = "national"

# ============== Helper Functions ==============
def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return np.array(image)

def calculate_hours(punch_in: datetime, punch_out: datetime, role: str = "housekeeping") -> dict:
    """Calculate total, regular, overtime hours and attendance type based on role"""
    duration = punch_out - punch_in
    total_hours = duration.total_seconds() / 3600
    
    # Get role-specific config
    config = ROLE_CONFIG.get(role.lower(), ROLE_CONFIG["housekeeping"])
    full_day_hours = config["full_day_hours"]
    overtime_threshold = config["overtime_threshold"]
    half_day_hours = config["half_day_hours"]
    
    # Overtime calculation: starts after threshold
    overtime_hours = max(0, total_hours - overtime_threshold)
    regular_hours = min(total_hours, full_day_hours)
    
    # Determine attendance type based on role hours
    if total_hours >= full_day_hours:
        attendance_type = "full_day"
    elif total_hours >= half_day_hours:
        attendance_type = "half_day"
    else:
        attendance_type = "short_day"
    
    return {
        'total_hours': round(total_hours, 2),
        'regular_hours': round(regular_hours, 2),
        'overtime_hours': round(overtime_hours, 2),
        'attendance_type': attendance_type
    }

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ============== Page Routes ==============
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/punch", response_class=HTMLResponse)
async def punch_page(request: Request):
    return templates.TemplateResponse("punch.html", {"request": request})

@app.get("/employees", response_class=HTMLResponse)
async def employees_page(request: Request):
    return templates.TemplateResponse("employees.html", {"request": request})

@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    return templates.TemplateResponse("reports.html", {"request": request})

@app.get("/calendar", response_class=HTMLResponse)
async def calendar_page(request: Request):
    return templates.TemplateResponse("calendar.html", {"request": request})

@app.get("/monthly-report", response_class=HTMLResponse)
async def monthly_report_page(request: Request):
    return templates.TemplateResponse("monthly_report.html", {"request": request})

# ============== API Routes ==============

@app.post("/api/admin-login")
async def admin_login(data: AdminLogin):
    """Admin login endpoint"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM admin_users WHERE username = ? AND password = ?",
                  (data.username, data.password))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {"success": True, "message": "Login successful", "redirect_url": "/dashboard"}
    else:
        return {"success": False, "message": "Invalid credentials"}

@app.post("/api/register-employee")
async def register_employee(data: EmployeeRegistration):
    """Register new employee with face"""
    if not FACE_RECOGNITION_AVAILABLE:
        return {"success": False, "message": "Face recognition not available"}
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT emp_id FROM employees WHERE emp_id = ?", (data.emp_id,))
        if cursor.fetchone():
            conn.close()
            return {"success": False, "message": f"Employee ID {data.emp_id} already exists"}
        
        image_array = decode_base64_image(data.face_image)
        encoding, location = asian_face_recognizer.detect_and_encode(image_array)
        
        if encoding is None:
            conn.close()
            return {"success": False, "message": "No face detected in the image. Please try again."}
        
        cursor.execute('''
            INSERT INTO employees (emp_id, emp_name, phone, role, designation, face_registered, status)
            VALUES (?, ?, ?, ?, ?, 1, 'active')
        ''', (data.emp_id, data.emp_name, data.phone, data.role.lower(), data.designation))
        
        conn.commit()
        conn.close()
        
        asian_face_recognizer.save_face_encoding(data.emp_id, data.emp_name, encoding)
        
        os.makedirs('employee_photos', exist_ok=True)
        photo_path = f"employee_photos/{data.emp_id}.jpg"
        Image.fromarray(image_array).save(photo_path)
        
        return {
            "success": True,
            "message": f"Employee {data.emp_name} registered successfully!",
            "emp_id": data.emp_id
        }
        
    except Exception as e:
        return {"success": False, "message": f"Registration failed: {str(e)}"}

@app.post("/api/punch")
async def punch_attendance(data: PunchRequest):
    """Handle punch in/out via face recognition"""
    if not FACE_RECOGNITION_AVAILABLE:
        return {"success": False, "message": "Face recognition not available"}
    
    try:
        image_array = decode_base64_image(data.image_data)
        encoding, location = asian_face_recognizer.detect_and_encode(image_array)
        
        if encoding is None:
            return {
                "success": False,
                "message": "No face detected. Please position your face clearly.",
                "face_detected": False
            }
        
        emp_id, emp_name, confidence = asian_face_recognizer.compare_faces(encoding, threshold=0.5)
        
        if emp_id is None:
            return {
                "success": False,
                "message": "Face not recognized. Please register first.",
                "face_detected": True,
                "recognized": False
            }
        
        now = get_ist_now()
        today = now.strftime('%Y-%m-%d')
        
        # Check if today is a holiday
        is_hol, hol_name = is_holiday(today)
        day_type = "holiday" if is_hol else "working"
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get employee role for hours calculation
        cursor.execute("SELECT role FROM employees WHERE emp_id = ?", (emp_id,))
        emp_row = cursor.fetchone()
        role = emp_row['role'] if emp_row and emp_row['role'] else 'housekeeping'
        
        cursor.execute('''
            SELECT * FROM punch_records 
            WHERE emp_id = ? AND punch_date = ?
        ''', (emp_id, today))
        
        record = cursor.fetchone()
        
        if record is None:
            # PUNCH IN
            cursor.execute('''
                INSERT INTO punch_records (emp_id, punch_date, punch_in_time, day_type, status)
                VALUES (?, ?, ?, ?, 'punched_in')
            ''', (emp_id, today, now.isoformat(), day_type))
            
            conn.commit()
            conn.close()
            
            message = f"✅ Punch IN successful!"
            if is_hol:
                message += f" (Working on {hol_name})"
            
            return {
                "success": True,
                "punch_type": "IN",
                "message": message,
                "emp_id": emp_id,
                "emp_name": emp_name,
                "time": now.strftime('%I:%M %p'),
                "confidence": round(confidence * 100, 1),
                "is_holiday": is_hol,
                "holiday_name": hol_name
            }
        
        elif record['status'] == 'punched_in':
            # PUNCH OUT - use role-based calculation
            punch_in_time = datetime.fromisoformat(record['punch_in_time'])
            hours = calculate_hours(punch_in_time, now, role)
            
            cursor.execute('''
                UPDATE punch_records 
                SET punch_out_time = ?, 
                    total_hours = ?, 
                    regular_hours = ?, 
                    overtime_hours = ?,
                    attendance_type = ?,
                    status = 'completed'
                WHERE emp_id = ? AND punch_date = ?
            ''', (now.isoformat(), hours['total_hours'], hours['regular_hours'], 
                  hours['overtime_hours'], hours['attendance_type'], emp_id, today))
            
            conn.commit()
            conn.close()
            
            message = f"✅ Punch OUT successful!"
            if is_hol:
                message += f" (Worked on {hol_name})"
            
            return {
                "success": True,
                "punch_type": "OUT",
                "message": message,
                "emp_id": emp_id,
                "emp_name": emp_name,
                "time": now.strftime('%I:%M %p'),
                "confidence": round(confidence * 100, 1),
                "total_hours": hours['total_hours'],
                "regular_hours": hours['regular_hours'],
                "overtime_hours": hours['overtime_hours'],
                "attendance_type": hours['attendance_type'],
                "punch_in_time": punch_in_time.strftime('%I:%M %p'),
                "is_holiday": is_hol
            }
        
        else:
            conn.close()
            return {
                "success": False,
                "message": f"Already punched in and out today.",
                "emp_id": emp_id,
                "emp_name": emp_name,
                "already_completed": True
            }
        
    except Exception as e:
        print(f"[ERROR] Punch failed: {e}")
        return {"success": False, "message": f"Punch failed: {str(e)}"}

@app.get("/api/employees")
async def get_employees():
    """Get all employees"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT emp_id, emp_name, phone, role, designation, face_registered, status, created_at 
        FROM employees ORDER BY 
            CASE designation 
                WHEN 'High-skilled' THEN 1 
                WHEN 'Skilled' THEN 2 
                WHEN 'Semi-skilled' THEN 3 
                WHEN 'Unskilled' THEN 4 
                ELSE 5 
            END,
            role, emp_name
    ''')
    
    employees = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"success": True, "employees": employees}

@app.get("/api/employee/{emp_id}")
async def get_employee(emp_id: str):
    """Get single employee details"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM employees WHERE emp_id = ?", (emp_id,))
    employee = cursor.fetchone()
    conn.close()
    
    if employee:
        return {"success": True, "employee": dict(employee)}
    return {"success": False, "message": "Employee not found"}

class EmployeeUpdate(BaseModel):
    emp_name: Optional[str] = None
    phone: Optional[str] = None
    role: Optional[str] = None
    designation: Optional[str] = None
    status: Optional[str] = None

@app.put("/api/employee/{emp_id}")
async def update_employee(emp_id: str, data: EmployeeUpdate):
    """Update employee details"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if employee exists
    cursor.execute("SELECT * FROM employees WHERE emp_id = ?", (emp_id,))
    employee = cursor.fetchone()
    
    if not employee:
        conn.close()
        return {"success": False, "message": "Employee not found"}
    
    # Build update query dynamically
    updates = []
    values = []
    
    if data.emp_name:
        updates.append("emp_name = ?")
        values.append(data.emp_name)
    if data.phone:
        updates.append("phone = ?")
        values.append(data.phone)
    if data.role:
        updates.append("role = ?")
        values.append(data.role.lower())
    if data.designation:
        updates.append("designation = ?")
        values.append(data.designation)
    if data.status:
        updates.append("status = ?")
        values.append(data.status)
    
    if not updates:
        conn.close()
        return {"success": False, "message": "No fields to update"}
    
    values.append(emp_id)
    query = f"UPDATE employees SET {', '.join(updates)} WHERE emp_id = ?"
    
    cursor.execute(query, values)
    conn.commit()
    conn.close()
    
    # Update face encoding name if name changed
    if data.emp_name and FACE_RECOGNITION_AVAILABLE:
        if emp_id in asian_face_recognizer.known_face_ids:
            idx = asian_face_recognizer.known_face_ids.index(emp_id)
            asian_face_recognizer.known_face_names[idx] = data.emp_name
    
    return {"success": True, "message": f"Employee {emp_id} updated successfully"}

@app.delete("/api/employee/{emp_id}")
async def delete_employee(emp_id: str):
    """Delete an employee"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM employees WHERE emp_id = ?", (emp_id,))
    cursor.execute("DELETE FROM punch_records WHERE emp_id = ?", (emp_id,))
    
    conn.commit()
    conn.close()
    
    if FACE_RECOGNITION_AVAILABLE:
        asian_face_recognizer.delete_face_encoding(emp_id)
    
    photo_path = f"employee_photos/{emp_id}.jpg"
    if os.path.exists(photo_path):
        os.remove(photo_path)
    
    return {"success": True, "message": f"Employee {emp_id} deleted"}

@app.get("/api/attendance/today")
async def get_today_attendance():
    """Get today's attendance summary"""
    today = get_ist_now().strftime('%Y-%m-%d')
    is_hol, hol_name = is_holiday(today)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM employees WHERE status = 'active'")
    total_employees = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT p.*, e.emp_name 
        FROM punch_records p
        JOIN employees e ON p.emp_id = e.emp_id
        WHERE p.punch_date = ?
        ORDER BY p.punch_in_time DESC
    ''', (today,))
    
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    punched_in = sum(1 for r in records if r['status'] == 'punched_in')
    completed = sum(1 for r in records if r['status'] == 'completed')
    
    return {
        "success": True,
        "date": today,
        "is_holiday": is_hol,
        "holiday_name": hol_name,
        "total_employees": total_employees,
        "punched_in": punched_in,
        "completed": completed,
        "not_punched": total_employees - len(records),
        "records": records
    }

@app.get("/api/attendance/report")
async def get_attendance_report(start_date: str, end_date: str, emp_id: Optional[str] = None):
    """Get attendance report for date range"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if emp_id:
        cursor.execute('''
            SELECT p.*, e.emp_name, e.phone
            FROM punch_records p
            JOIN employees e ON p.emp_id = e.emp_id
            WHERE p.punch_date BETWEEN ? AND ? AND p.emp_id = ?
            ORDER BY p.punch_date DESC
        ''', (start_date, end_date, emp_id))
    else:
        cursor.execute('''
            SELECT p.*, e.emp_name, e.phone
            FROM punch_records p
            JOIN employees e ON p.emp_id = e.emp_id
            WHERE p.punch_date BETWEEN ? AND ?
            ORDER BY p.punch_date DESC, p.emp_id
        ''', (start_date, end_date))
    
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Add holiday info to each record
    for r in records:
        is_hol, hol_name = is_holiday(r['punch_date'])
        r['is_holiday'] = is_hol
        r['holiday_name'] = hol_name
    
    total_regular = sum(r['regular_hours'] or 0 for r in records)
    total_overtime = sum(r['overtime_hours'] or 0 for r in records)
    total_hours = sum(r['total_hours'] or 0 for r in records)
    
    return {
        "success": True,
        "records": records,
        "summary": {
            "total_records": len(records),
            "total_hours": round(total_hours, 2),
            "total_regular_hours": round(total_regular, 2),
            "total_overtime_hours": round(total_overtime, 2)
        }
    }

@app.get("/api/monthly-report/{month}")
async def get_monthly_report(month: str, emp_id: Optional[str] = None, role: Optional[str] = None):
    """Get comprehensive monthly report in muster book format (format: YYYY-MM)"""
    try:
        year, mon = map(int, month.split('-'))
        month_info = get_working_days_in_month(year, mon)
        
        # Get number of days in month
        import calendar
        num_days = calendar.monthrange(year, mon)[1]
    except:
        return {"success": False, "message": "Invalid month format. Use YYYY-MM"}
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all employees or specific employee, optionally filter by role
    if emp_id:
        cursor.execute("SELECT * FROM employees WHERE emp_id = ? AND status = 'active'", (emp_id,))
    elif role:
        cursor.execute("""
            SELECT * FROM employees WHERE status = 'active' AND role = ?
            ORDER BY 
                CASE designation 
                    WHEN 'High-skilled' THEN 1 
                    WHEN 'Skilled' THEN 2 
                    WHEN 'Semi-skilled' THEN 3 
                    WHEN 'Unskilled' THEN 4 
                    ELSE 5 
                END, emp_name
        """, (role.lower(),))
    else:
        cursor.execute("""
            SELECT * FROM employees WHERE status = 'active'
            ORDER BY role,
                CASE designation 
                    WHEN 'High-skilled' THEN 1 
                    WHEN 'Skilled' THEN 2 
                    WHEN 'Semi-skilled' THEN 3 
                    WHEN 'Unskilled' THEN 4 
                    ELSE 5 
                END, emp_name
        """)
    
    employees = [dict(row) for row in cursor.fetchall()]
    
    employee_reports = []
    
    # Group employees by designation
    designation_groups = {
        "High-skilled": [],
        "Skilled": [],
        "Semi-skilled": [],
        "Unskilled": []
    }
    
    for emp in employees:
        # Get role config for this employee
        emp_role = emp.get('role', 'housekeeping') or 'housekeeping'
        role_config = ROLE_CONFIG.get(emp_role, ROLE_CONFIG["housekeeping"])
        full_day_hours = role_config["full_day_hours"]
        half_day_hours = role_config["half_day_hours"]
        
        cursor.execute('''
            SELECT * FROM punch_records
            WHERE emp_id = ? AND strftime('%Y-%m', punch_date) = ?
            ORDER BY punch_date
        ''', (emp['emp_id'], month))
        
        records = [dict(row) for row in cursor.fetchall()]
        
        # Build daily attendance map (P=Present, A=Absent, S=Sunday/Holiday, H=Half day)
        daily_attendance = {}
        for day in range(1, num_days + 1):
            date_str = f"{year}-{mon:02d}-{day:02d}"
            is_hol, hol_name = is_holiday(date_str)
            
            # Check if it's Sunday
            date_obj = date(year, mon, day)
            is_sunday = date_obj.weekday() == 6
            
            if is_sunday or is_hol:
                daily_attendance[day] = {"status": "S", "hours": 0, "is_holiday": True}
            else:
                daily_attendance[day] = {"status": "A", "hours": 0, "is_holiday": False}
        
        # Calculate statistics
        full_days = 0
        half_days = 0
        days_present = 0
        total_hours = 0
        regular_hours = 0
        overtime_hours = 0
        holiday_work_days = 0
        
        for r in records:
            if r['status'] == 'completed':
                # Parse the date
                punch_date = r['punch_date']
                day = int(punch_date.split('-')[2])
                hours_worked = r['total_hours'] or 0
                
                days_present += 1
                total_hours += hours_worked
                regular_hours += r['regular_hours'] or 0
                overtime_hours += r['overtime_hours'] or 0
                
                # Check if worked on holiday/Sunday
                is_hol, _ = is_holiday(punch_date)
                date_obj = date(int(punch_date.split('-')[0]), int(punch_date.split('-')[1]), day)
                is_sunday = date_obj.weekday() == 6
                
                if is_hol or is_sunday:
                    holiday_work_days += 1
                    daily_attendance[day] = {"status": "P", "hours": hours_worked, "is_holiday": True, "worked": True}
                else:
                    # Determine attendance type based on role
                    if hours_worked >= full_day_hours:
                        full_days += 1
                        daily_attendance[day] = {"status": "P", "hours": hours_worked, "is_holiday": False}
                    elif hours_worked >= half_day_hours:
                        half_days += 1
                        daily_attendance[day] = {"status": "H", "hours": hours_worked, "is_holiday": False}
                    else:
                        daily_attendance[day] = {"status": "P", "hours": hours_worked, "is_holiday": False}
        
        # Calculate absent days (only count working days where employee was absent)
        absent_days = 0
        for day, att in daily_attendance.items():
            if not att['is_holiday'] and att['status'] == 'A':
                absent_days += 1
        
        emp_report = {
            "emp_id": emp['emp_id'],
            "emp_name": emp['emp_name'],
            "phone": emp.get('phone', ''),
            "role": emp.get('role', 'housekeeping') or 'housekeeping',
            "designation": emp.get('designation', 'Unskilled') or 'Unskilled',
            "days_present": days_present,
            "full_days": full_days,
            "half_days": half_days,
            "absent_days": absent_days,
            "holiday_work_days": holiday_work_days,
            "total_hours": round(total_hours, 2),
            "regular_hours": round(regular_hours, 2),
            "overtime_hours": round(overtime_hours, 2),
            "avg_hours_per_day": round(total_hours / days_present, 2) if days_present > 0 else 0,
            "daily_attendance": daily_attendance,
            "daily_records": records
        }
        
        employee_reports.append(emp_report)
        
        # Add to designation group
        designation = emp.get('designation', 'Unskilled') or 'Unskilled'
        if designation in designation_groups:
            designation_groups[designation].append(emp_report)
        else:
            designation_groups['Unskilled'].append(emp_report)
    
    conn.close()
    
    # Consolidated summary
    consolidated = {
        "total_employees": len(employee_reports),
        "total_present_days": sum(e['days_present'] for e in employee_reports),
        "total_full_days": sum(e['full_days'] for e in employee_reports),
        "total_half_days": sum(e['half_days'] for e in employee_reports),
        "total_absent_days": sum(e['absent_days'] for e in employee_reports),
        "total_holiday_work": sum(e['holiday_work_days'] for e in employee_reports),
        "total_hours": round(sum(e['total_hours'] for e in employee_reports), 2),
        "total_regular_hours": round(sum(e['regular_hours'] for e in employee_reports), 2),
        "total_overtime_hours": round(sum(e['overtime_hours'] for e in employee_reports), 2)
    }
    
    # Summary by designation
    designation_summary = {}
    for designation, emp_list in designation_groups.items():
        if emp_list:
            designation_summary[designation] = {
                "count": len(emp_list),
                "total_overtime": round(sum(e['overtime_hours'] for e in emp_list), 2),
                "total_absent": sum(e['absent_days'] for e in emp_list),
                "total_present": sum(e['days_present'] for e in emp_list)
            }
    
    return {
        "success": True,
        "month": month,
        "year": year,
        "month_number": mon,
        "num_days": num_days,
        "month_info": month_info,
        "employees": employee_reports,
        "designation_groups": designation_groups,
        "designation_summary": designation_summary,
        "consolidated": consolidated
    }

@app.get("/api/employee/{emp_id}/monthly-summary")
async def get_employee_monthly_summary(emp_id: str, month: str):
    """Get monthly summary for an employee (month format: YYYY-MM)"""
    try:
        year, mon = map(int, month.split('-'))
        month_info = get_working_days_in_month(year, mon)
    except:
        return {"success": False, "message": "Invalid month format"}
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM punch_records
        WHERE emp_id = ? AND strftime('%Y-%m', punch_date) = ?
        ORDER BY punch_date
    ''', (emp_id, month))
    
    daily_records = [dict(row) for row in cursor.fetchall()]
    
    # Add holiday info
    for r in daily_records:
        is_hol, hol_name = is_holiday(r['punch_date'])
        r['is_holiday'] = is_hol
        r['holiday_name'] = hol_name
    
    # Calculate summary
    full_days = 0
    half_days = 0
    total_hours = 0
    regular_hours = 0
    overtime_hours = 0
    holiday_work_days = 0
    
    for r in daily_records:
        if r['status'] == 'completed':
            total_hours += r['total_hours'] or 0
            regular_hours += r['regular_hours'] or 0
            overtime_hours += r['overtime_hours'] or 0
            
            if (r['total_hours'] or 0) >= FULL_DAY_MIN_HOURS:
                full_days += 1
            elif (r['total_hours'] or 0) >= HALF_DAY_MIN_HOURS:
                half_days += 1
            
            if r['is_holiday']:
                holiday_work_days += 1
    
    days_worked = full_days + half_days
    working_days_worked = days_worked - holiday_work_days
    absent_days = max(0, month_info['working_days'] - working_days_worked)
    
    conn.close()
    
    return {
        "success": True,
        "month": month,
        "month_info": month_info,
        "summary": {
            "days_worked": days_worked,
            "full_days": full_days,
            "half_days": half_days,
            "absent_days": absent_days,
            "holiday_work_days": holiday_work_days,
            "total_hours": round(total_hours, 2),
            "regular_hours": round(regular_hours, 2),
            "overtime_hours": round(overtime_hours, 2),
            "avg_hours_per_day": round(total_hours / days_worked, 2) if days_worked > 0 else 0
        },
        "daily_records": daily_records
    }

@app.get("/api/holidays")
async def get_holidays(year: Optional[int] = None):
    """Get all holidays, optionally filtered by year"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if year:
        cursor.execute('''
            SELECT * FROM holidays 
            WHERE strftime('%Y', holiday_date) = ?
            ORDER BY holiday_date
        ''', (str(year),))
    else:
        cursor.execute('SELECT * FROM holidays ORDER BY holiday_date')
    
    holidays = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"success": True, "holidays": holidays}

@app.post("/api/holidays")
async def add_holiday(data: HolidayRequest):
    """Add a new holiday"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO holidays (holiday_date, holiday_name, holiday_type)
            VALUES (?, ?, ?)
        ''', (data.holiday_date, data.holiday_name, data.holiday_type))
        
        # Also add to in-memory dict
        INDIAN_HOLIDAYS[data.holiday_date] = data.holiday_name
        
        conn.commit()
        conn.close()
        
        return {"success": True, "message": f"Holiday '{data.holiday_name}' added"}
    except sqlite3.IntegrityError:
        conn.close()
        return {"success": False, "message": "Holiday already exists for this date"}

@app.delete("/api/holidays/{holiday_date}")
async def delete_holiday(holiday_date: str):
    """Delete a holiday"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM holidays WHERE holiday_date = ?", (holiday_date,))
    
    # Remove from in-memory dict
    if holiday_date in INDIAN_HOLIDAYS:
        del INDIAN_HOLIDAYS[holiday_date]
    
    conn.commit()
    conn.close()
    
    return {"success": True, "message": "Holiday deleted"}

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    today = get_ist_now().strftime('%Y-%m-%d')
    current_month = get_ist_now().strftime('%Y-%m')
    is_hol, hol_name = is_holiday(today)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM employees WHERE status = 'active'")
    total_employees = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'punched_in' THEN 1 ELSE 0 END) as punched_in,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed
        FROM punch_records WHERE punch_date = ?
    ''', (today,))
    today_stats = dict(cursor.fetchone())
    
    cursor.execute('''
        SELECT 
            SUM(total_hours) as total_hours,
            SUM(regular_hours) as regular_hours,
            SUM(overtime_hours) as overtime_hours,
            SUM(CASE WHEN attendance_type = 'full_day' OR total_hours >= ? THEN 1 ELSE 0 END) as full_days,
            SUM(CASE WHEN attendance_type = 'half_day' OR (total_hours >= ? AND total_hours < ?) THEN 1 ELSE 0 END) as half_days
        FROM punch_records 
        WHERE strftime('%Y-%m', punch_date) = ? AND status = 'completed'
    ''', (FULL_DAY_MIN_HOURS, HALF_DAY_MIN_HOURS, FULL_DAY_MIN_HOURS, current_month))
    monthly_stats = dict(cursor.fetchone())
    
    conn.close()
    
    return {
        "success": True,
        "total_employees": total_employees,
        "today": {
            "date": today,
            "is_holiday": is_hol,
            "holiday_name": hol_name,
            "punched_in": today_stats['punched_in'] or 0,
            "completed": today_stats['completed'] or 0,
            "not_punched": total_employees - (today_stats['total'] or 0)
        },
        "monthly": {
            "month": current_month,
            "total_hours": round(monthly_stats['total_hours'] or 0, 2),
            "regular_hours": round(monthly_stats['regular_hours'] or 0, 2),
            "overtime_hours": round(monthly_stats['overtime_hours'] or 0, 2),
            "full_days": monthly_stats['full_days'] or 0,
            "half_days": monthly_stats['half_days'] or 0
        }
    }

@app.get("/api/live-status")
async def get_live_status():
    """Get live punch status (who's currently working)"""
    today = get_ist_now().strftime('%Y-%m-%d')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT p.emp_id, e.emp_name, p.punch_in_time
        FROM punch_records p
        JOIN employees e ON p.emp_id = e.emp_id
        WHERE p.punch_date = ? AND p.status = 'punched_in'
        ORDER BY p.punch_in_time
    ''', (today,))
    
    currently_working = []
    for row in cursor.fetchall():
        punch_in = datetime.fromisoformat(row['punch_in_time'])
        if punch_in.tzinfo is None:
            punch_in = punch_in.replace(tzinfo=IST)
        hours_worked = (get_ist_now() - punch_in).total_seconds() / 3600
        currently_working.append({
            "emp_id": row['emp_id'],
            "emp_name": row['emp_name'],
            "punch_in_time": punch_in.strftime('%I:%M %p'),
            "hours_worked": round(hours_worked, 2)
        })
    
    conn.close()
    
    return {
        "success": True,
        "currently_working": currently_working,
        "count": len(currently_working)
    }

# ============== Run Application ==============
if __name__ == "__main__":
    import uvicorn
    import os
    
    if os.path.exists("key.pem") and os.path.exists("cert.pem"):
        print("🔒 Running with HTTPS")
        uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
    else:
        print("⚠️ Running without HTTPS (camera won't work on remote devices)")
        uvicorn.run(app, host="0.0.0.0", port=8000)

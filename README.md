# 🕐 Punch Attendance System

A **Face Recognition-based Punch In/Punch Out Attendance System** designed for managing third-party staff working hours with automatic overtime calculation.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 👁️ **Face Recognition** | AI-powered face detection using InsightFace (buffalo_l model) |
| ⏰ **Punch In/Out** | Single scan for punch in, second scan for punch out |
| 📊 **Auto Time Calculation** | Automatic calculation of working hours |
| 💼 **9-Hour Standard** | Full day = 9 hours, extra hours = overtime |
| 📈 **Reports** | Daily, weekly, monthly attendance reports |
| 📥 **CSV Export** | Export attendance data for payroll |
| 🔒 **Admin Panel** | Complete employee and attendance management |

## 📋 System Requirements

- Python 3.9+
- Webcam for face capture
- 4GB+ RAM (for face recognition model)

## 🚀 Installation

### 1. Clone/Download the project

```bash
cd punch_attendance_system
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the System

- **Home Page**: http://localhost:8000
- **Punch Terminal**: http://localhost:8000/punch
- **Admin Login**: http://localhost:8000/login

**Default Admin Credentials:**
- Username: `admin`
- Password: `admin123`

## 📁 Project Structure

```
punch_attendance_system/
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── punch_attendance.db     # SQLite database (auto-created)
├── face_encodings/         # Stored face embeddings
├── employee_photos/        # Employee registration photos
├── templates/              # HTML templates
│   ├── index.html          # Home page
│   ├── login.html          # Admin login
│   ├── dashboard.html      # Admin dashboard
│   ├── register.html       # Employee registration
│   ├── employees.html      # Employee management
│   ├── reports.html        # Attendance reports
│   └── punch.html          # Punch terminal
└── static/                 # Static files (CSS, images)
```

## 💡 How It Works

### Employee Registration
1. Admin logs into the system
2. Navigate to "Register New"
3. Enter: Employee ID, Name, Phone
4. Capture face using webcam
5. Submit to register

### Punch In/Out
1. Employee goes to Punch Terminal
2. Position face in camera
3. Click "PUNCH" button
4. **First punch of day** = Punch IN
5. **Second punch of day** = Punch OUT (hours calculated)

### Hour Calculation
- **Standard Day**: 9 hours
- **Total Hours** = Punch OUT - Punch IN
- **Regular Hours** = min(Total, 9)
- **Overtime Hours** = max(0, Total - 9)

**Example:**
```
Punch IN:  7:00 AM
Punch OUT: 7:00 PM
Total:     12 hours
Regular:   9 hours
Overtime:  3 hours
```

## 📊 Database Schema

### employees
| Column | Type | Description |
|--------|------|-------------|
| emp_id | TEXT | Primary key |
| emp_name | TEXT | Employee name |
| phone | TEXT | Phone number |
| face_registered | INT | 1 if face registered |
| status | TEXT | active/inactive |

### punch_records
| Column | Type | Description |
|--------|------|-------------|
| emp_id | TEXT | Employee ID |
| punch_date | DATE | Date of punch |
| punch_in_time | TIMESTAMP | Punch in time |
| punch_out_time | TIMESTAMP | Punch out time |
| total_hours | REAL | Total worked hours |
| regular_hours | REAL | Standard hours (max 9) |
| overtime_hours | REAL | Extra hours |
| status | TEXT | incomplete/punched_in/completed |

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin-login` | POST | Admin authentication |
| `/api/register-employee` | POST | Register new employee |
| `/api/punch` | POST | Punch in/out via face |
| `/api/employees` | GET | List all employees |
| `/api/employee/{id}` | GET | Get employee details |
| `/api/employee/{id}` | DELETE | Delete employee |
| `/api/attendance/today` | GET | Today's attendance |
| `/api/attendance/report` | GET | Date range report |
| `/api/dashboard/stats` | GET | Dashboard statistics |
| `/api/live-status` | GET | Currently working employees |

## ⚙️ Configuration

Edit in `main.py`:

```python
STANDARD_WORKING_HOURS = 9  # Change full day hours
DATABASE_PATH = "punch_attendance.db"  # Database location
```

## 🔒 Security Notes

- Change default admin password after first login
- Use HTTPS in production
- Face encodings are stored locally (consider encryption)
- Add rate limiting for production use

## 🐛 Troubleshooting

### Camera not working
- Check browser permissions
- Try different browser (Chrome recommended)
- Ensure no other app is using camera

### Face not detected
- Improve lighting
- Face the camera directly
- Remove glasses/hat if possible

### InsightFace installation issues
```bash
pip install insightface onnxruntime
```

For GPU support:
```bash
pip install onnxruntime-gpu
```

## 📝 License

MIT License - Free for personal and commercial use.

## 🤝 Support

For issues or feature requests, please create an issue in the repository.

---

**Built with ❤️ using FastAPI + InsightFace**

# app.py
# Enhanced Face Recognition Attendance System with Authentication

import cv2
import face_recognition
import numpy as np
import sqlite3
import bcrypt
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import os
import json
import collections
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, PasswordField, EmailField, SelectField, FileField, SubmitField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import threading
from contextlib import contextmanager
from functools import wraps
import secrets

# User Model for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, username, email, role='employee', active=True, employee_id=None, name=None):
        self.id = user_id
        self.username = username
        self.name = name or username
        self.email = email
        self.role = role
        self._active = active
        self.employee_id = employee_id
    
    def get_id(self):
        return str(self.id)
    
    @property
    def is_active(self):
        return self._active
    
    def is_admin(self):
        return self.role == 'admin'

# Forms for Authentication
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', 
                                   validators=[DataRequired(), EqualTo('password')])
    role = SelectField('Role', choices=[('employee', 'Employee'), ('admin', 'Admin')], 
                      default='employee')
    submit = SubmitField('Sign Up')
    
    def validate_username(self, username):
        with DatabaseManager().get_db_connection() as conn:
            cursor = conn.execute('SELECT id FROM users WHERE username = ?', (username.data,))
            if cursor.fetchone():
                raise ValidationError('Username already exists. Please choose a different one.')
    
    def validate_email(self, email):
        with DatabaseManager().get_db_connection() as conn:
            cursor = conn.execute('SELECT id FROM users WHERE email = ?', (email.data,))
            if cursor.fetchone():
                raise ValidationError('Email already registered. Please use a different email.')

class EmployeeForm(FlaskForm):
    employee_id = StringField('Employee ID', validators=[DataRequired()])
    name = StringField('Full Name', validators=[DataRequired()])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    department = StringField('Department', validators=[DataRequired()])
    photo = FileField('Profile Photo', validators=[DataRequired()])
    submit = SubmitField('Add Employee')

# Authentication decorator
def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin():
            flash('Admin access required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function
class DatabaseManager:
    def __init__(self, db_path="attendance_system.db"):
        self.db_path = db_path
        self.init_database()
        self.migrate_schema()
    
    def get_analytics_data(self, start_date=None, end_date=None):
        """Get analytics data for the dashboard"""
        with self.get_db_connection() as conn:
            if not start_date:
                start_date = datetime.now().date() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now().date()
                
            # Get department-wise attendance
            dept_attendance = conn.execute('''
                SELECT e.department, 
                       COUNT(DISTINCT a.employee_id) as present_count,
                       COUNT(DISTINCT e.employee_id) as total_employees,
                       CAST(COUNT(DISTINCT a.employee_id) AS FLOAT) / 
                       COUNT(DISTINCT e.employee_id) * 100 as attendance_rate
                FROM employees e
                LEFT JOIN attendance a ON e.employee_id = a.employee_id 
                    AND a.date BETWEEN ? AND ?
                WHERE e.is_active = 1
                GROUP BY e.department
            ''', (start_date, end_date)).fetchall()
            
            # Get daily attendance trend
            daily_trend = conn.execute('''
                SELECT date, 
                       COUNT(DISTINCT employee_id) as present_count
                FROM attendance
                WHERE date BETWEEN ? AND ?
                GROUP BY date
                ORDER BY date
            ''', (start_date, end_date)).fetchall()
            
            return {
                'department_stats': [dict(row) for row in dept_attendance],
                'daily_trend': [dict(row) for row in daily_trend]
            }
    
    @contextmanager
    def get_db_connection(self):
        """Thread-safe database connection with optimized settings"""
        conn = sqlite3.connect(
            self.db_path, 
            check_same_thread=False,
            timeout=30.0  # 30 second timeout
        )
        conn.row_factory = sqlite3.Row
        # Optimize SQLite performance
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA synchronous = NORMAL')
        conn.execute('PRAGMA cache_size = 10000')
        conn.execute('PRAGMA temp_store = MEMORY')
        try:
            yield conn
        finally:
            conn.close()
    
    def get_table_columns(self, conn, table_name):
        """Get list of columns for a table"""
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [column[1] for column in cursor.fetchall()]
    
    def migrate_schema(self):
        """Migrate database schema to latest version"""
        with self.get_db_connection() as conn:
            # Check and add missing columns to employees table
            employees_columns = self.get_table_columns(conn, 'employees')
            
            if 'created_by' not in employees_columns:
                conn.execute('ALTER TABLE employees ADD COLUMN created_by INTEGER')
                app.logger.info("Added created_by column to employees table")
            
            if 'is_active' not in employees_columns:
                conn.execute('ALTER TABLE employees ADD COLUMN is_active BOOLEAN DEFAULT 1')
                app.logger.info("Added is_active column to employees table")
            
            if 'department' not in employees_columns:
                conn.execute('ALTER TABLE employees ADD COLUMN department TEXT')
                app.logger.info("Added department column to employees table")
            
            # Check and add missing columns to attendance table
            attendance_columns = self.get_table_columns(conn, 'attendance')
            
            if 'work_hours' not in attendance_columns:
                conn.execute('ALTER TABLE attendance ADD COLUMN work_hours REAL DEFAULT 0')
                app.logger.info("Added work_hours column to attendance table")
            
            if 'break_minutes' not in attendance_columns:
                conn.execute('ALTER TABLE attendance ADD COLUMN break_minutes INTEGER DEFAULT 0')
                app.logger.info("Added break_minutes column to attendance table")
            
            if 'overtime_hours' not in attendance_columns:
                conn.execute('ALTER TABLE attendance ADD COLUMN overtime_hours REAL DEFAULT 0')
                app.logger.info("Added overtime_hours column to attendance table")
            
            if 'location' not in attendance_columns:
                conn.execute('ALTER TABLE attendance ADD COLUMN location TEXT')
                app.logger.info("Added location column to attendance table")
            
            # Create indexes for better performance
            try:
                conn.execute('CREATE INDEX IF NOT EXISTS idx_employees_employee_id ON employees(employee_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_employees_is_active ON employees(is_active)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_attendance_employee_id ON attendance(employee_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_attendance_employee_date ON attendance(employee_id, date)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active)')
                app.logger.info("Database indexes created successfully")
            except sqlite3.Error as e:
                app.logger.warning(f"Index creation warning: {e}")
            
            conn.commit()
    
    def init_database(self):
        """Initialize database with optimized schema"""
        with self.get_db_connection() as conn:
            # Create users table for authentication
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'employee' CHECK (role IN ('employee', 'admin')),
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            
            # Create employees table with all necessary columns
            conn.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT,
                    department TEXT,
                    face_encoding TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by INTEGER,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (created_by) REFERENCES users (id)
                )
            ''')
            
            # Create attendance table with enhanced fields
            conn.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT NOT NULL,
                    check_in_time TIMESTAMP,
                    check_out_time TIMESTAMP,
                    date DATE NOT NULL,
                    status TEXT DEFAULT 'present' CHECK (status IN ('present', 'absent', 'late', 'half_day')),
                    work_hours REAL DEFAULT 0,
                    break_minutes INTEGER DEFAULT 0,
                    overtime_hours REAL DEFAULT 0,
                    location TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
                )
            ''')
            
            # Create admin user if not exists
            cursor = conn.execute('SELECT id FROM users WHERE username = ?', ('admin',))
            if not cursor.fetchone():
                admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
                admin_password_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt())
                conn.execute('''
                    INSERT INTO users (username, email, password_hash, role)
                    VALUES (?, ?, ?, ?)
                ''', ('admin', 'admin@company.com', admin_password_hash.decode('utf-8'), 'admin'))
                app.logger.info("Default admin user created")
            
            conn.commit()
    
    # User authentication methods
    def create_user(self, username, email, password, role='employee'):
        with self.get_db_connection() as conn:
            try:
                password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                conn.execute('''
                    INSERT INTO users (username, email, password_hash, role)
                    VALUES (?, ?, ?, ?)
                ''', (username, email, password_hash.decode('utf-8'), role))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    
    def authenticate_user(self, username, password):
        with self.get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT u.*, e.employee_id, e.name 
                FROM users u 
                LEFT JOIN employees e ON u.email = e.email 
                WHERE u.username = ? AND u.is_active = 1
            ''', (username,))
            user_data = cursor.fetchone()

            if user_data and bcrypt.checkpw(password.encode('utf-8'), user_data['password_hash'].encode('utf-8')):
                # Update last login
                conn.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                           (datetime.now(), user_data['id']))
                conn.commit()
                return User(
                    user_data['id'], 
                    user_data['username'], 
                    user_data['email'], 
                    user_data['role'],
                    employee_id=user_data['employee_id'],
                    name=user_data['name']
                )
            return None
    
    def get_user_by_id(self, user_id):
        with self.get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT u.*, e.employee_id, e.name 
                FROM users u 
                LEFT JOIN employees e ON u.email = e.email 
                WHERE u.id = ? AND u.is_active = 1
            ''', (user_id,))
            user_data = cursor.fetchone()

            if user_data:
                return User(
                    user_data['id'], 
                    user_data['username'], 
                    user_data['email'], 
                    user_data['role'],
                    employee_id=user_data['employee_id'],
                    name=user_data['name']
                )
            return None
    
    def add_employee(self, employee_id, name, email, department, face_encoding, created_by=None):
        """Add a new employee with optimized error handling"""
        with self.get_db_connection() as conn:
            try:
                encoding_str = json.dumps(face_encoding.tolist())
                conn.execute('''
                    INSERT INTO employees (employee_id, name, email, department, face_encoding, created_by)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (employee_id, name, email, department, encoding_str, created_by))
                conn.commit()
                app.logger.info(f"Employee {employee_id} added successfully by user {created_by}")
                return True
            except sqlite3.IntegrityError as e:
                app.logger.error(f"Failed to add employee {employee_id}: {e}")
                return False
            except Exception as e:
                app.logger.error(f"Unexpected error adding employee {employee_id}: {e}")
                return False
    
    def get_all_employees(self, include_inactive=False):
        """Get all employees with optimized query"""
        with self.get_db_connection() as conn:
            query = 'SELECT * FROM employees'
            if not include_inactive:
                query += ' WHERE is_active = 1'
            query += ' ORDER BY created_at DESC'
            
            try:
                cursor = conn.execute(query)
                rows = cursor.fetchall()
                # Convert Row objects to dictionaries for template compatibility
                return [dict(row) for row in rows]
            except sqlite3.Error as e:
                app.logger.error(f"Error fetching employees: {e}")
                return []

    def delete_employee(self, employee_id):
        """Soft delete employee (mark as inactive)"""
        with self.get_db_connection() as conn:
            try:
                cursor = conn.execute('UPDATE employees SET is_active = 0 WHERE employee_id = ?', (employee_id,))
                conn.commit()
                app.logger.info(f"Employee {employee_id} deactivated")
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                app.logger.error(f"Error deactivating employee {employee_id}: {e}")
                return False

    def get_employee_encodings(self, include_inactive=False):
        """Get employee face encodings with optimized processing"""
        with self.get_db_connection() as conn:
            query = 'SELECT employee_id, name, face_encoding FROM employees'
            if not include_inactive:
                query += ' WHERE is_active = 1'
            
            try:
                cursor = conn.execute(query)
                employees = cursor.fetchall()
                
                encodings, names, employee_ids = [], [], []
                
                for emp in employees:
                    try:
                        encoding = np.array(json.loads(emp['face_encoding']))
                        encodings.append(encoding)
                        names.append(emp['name'])
                        employee_ids.append(emp['employee_id'])
                    except (json.JSONDecodeError, ValueError) as e:
                        app.logger.warning(f"Invalid face encoding for employee {emp['employee_id']}: {e}")
                        continue
                
                return encodings, names, employee_ids
            except sqlite3.Error as e:
                app.logger.error(f"Error fetching employee encodings: {e}")
                return [], [], []
    
    def mark_attendance(self, employee_id, attendance_type='check_in'):
        with self.get_db_connection() as conn:
            today = datetime.now().date()
            current_time = datetime.now()
            
            cursor = conn.execute('SELECT * FROM attendance WHERE employee_id = ? AND date = ?', (employee_id, today))
            existing_record = cursor.fetchone()
            
            if existing_record:
                if attendance_type == 'check_out' and existing_record['check_out_time'] is None:
                    # Calculate work hours
                    check_in_str = existing_record['check_in_time']
                    if isinstance(check_in_str, str):
                        try:
                            check_in = datetime.fromisoformat(check_in_str.replace('Z', '+00:00'))
                        except ValueError:
                            check_in = datetime.strptime(check_in_str, '%Y-%m-%d %H:%M:%S.%f')
                    else:
                        check_in = check_in_str
                    
                    work_duration = current_time - check_in
                    work_hours = work_duration.total_seconds() / 3600
                    
                    # Standard work day is 8 hours
                    overtime_hours = max(0, work_hours - 8)
                    
                    conn.execute('''
                        UPDATE attendance 
                        SET check_out_time = ?, work_hours = ?, overtime_hours = ?
                        WHERE id = ?
                    ''', (current_time.isoformat(), work_hours, overtime_hours, existing_record['id']))
                    conn.commit()
                    return {'success': True, 'message': 'Check-out recorded successfully', 'work_hours': work_hours}
                elif attendance_type == 'check_in':
                    return {'success': False, 'message': 'Already checked in for today'}
                else:
                    return {'success': False, 'message': 'Already checked out for today'}
            else:
                if attendance_type == 'check_in':
                    conn.execute('INSERT INTO attendance (employee_id, check_in_time, date) VALUES (?, ?, ?)', 
                               (employee_id, current_time.isoformat(), today))
                    conn.commit()
                    return {'success': True, 'message': 'Check-in recorded successfully'}
                else:
                    return {'success': False, 'message': 'Must check-in first to check-out'}
    
    def get_attendance_records(self, days=30, employee_id=None):
        with self.get_db_connection() as conn:
            start_date = datetime.now().date() - timedelta(days=days)
            query = '''
                SELECT a.*, e.name, e.department
                FROM attendance a
                JOIN employees e ON a.employee_id = e.employee_id
                WHERE a.date >= ? AND e.is_active = 1
            '''
            params = [start_date]
            
            if employee_id:
                query += ' AND a.employee_id = ?'
                params.append(employee_id)
                
            query += ' ORDER BY a.date DESC, a.check_in_time DESC'
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    def get_attendance_summary(self):
        with self.get_db_connection() as conn:
            today = datetime.now().date()
            
            # Present today
            cursor = conn.execute('''
                SELECT COUNT(DISTINCT a.employee_id) as present_today 
                FROM attendance a
                JOIN employees e ON a.employee_id = e.employee_id
                WHERE a.date = ? AND e.is_active = 1
            ''', (today,))
            present_today = cursor.fetchone()['present_today']
            
            # Total active employees
            cursor = conn.execute('SELECT COUNT(*) as total FROM employees WHERE is_active = 1')
            total_employees = cursor.fetchone()['total']
            
            # Late arrivals (after 9 AM)
            cursor = conn.execute('''
                SELECT COUNT(*) as late_today
                FROM attendance a
                JOIN employees e ON a.employee_id = e.employee_id
                WHERE a.date = ? AND TIME(a.check_in_time) > '09:00:00' AND e.is_active = 1
            ''', (today,))
            late_today = cursor.fetchone()['late_today']
            
            # Calculate average attendance for last 30 days
            cursor = conn.execute('''
                SELECT AVG(daily_count) as avg_attendance
                FROM (
                    SELECT COUNT(DISTINCT a.employee_id) as daily_count
                    FROM attendance a
                    JOIN employees e ON a.employee_id = e.employee_id
                    WHERE a.date >= ? AND e.is_active = 1
                    GROUP BY a.date
                )
            ''', (today - timedelta(days=30),))
            avg_result = cursor.fetchone()
            avg_attendance = round((avg_result['avg_attendance'] or 0) / max(total_employees, 1) * 100, 1)
            
            return {
                'present_today': present_today,
                'total_employees': total_employees,
                'absent_today': total_employees - present_today,
                'late_today': late_today,
                'avg_attendance': avg_attendance
            }

from enhanced_recognition import EnhancedFaceRecognitionSystem
import time

class FaceRecognitionSystem(EnhancedFaceRecognitionSystem):
    def __init__(self):
        super().__init__(DatabaseManager())
        self.camera_active = False
    
    def start_camera(self):
        """Initialize and start camera with optimized settings for better face recognition"""
        with self.lock:
            if self.camera is None:
                try:
                    self.camera = cv2.VideoCapture(0)
                    if not self.camera.isOpened():
                        raise Exception("Failed to open camera")
                    
                    # Camera Resolution Settings - 640x480 is optimal for face recognition
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camera.set(cv2.CAP_PROP_FPS, 30)
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Color Space: RGB (Standard) - for skin-tone segmentation
                    # Note: OpenCV captures in BGR, conversion to RGB is done in preprocessing
                    # Ensure camera is in color mode (not grayscale)
                    self.camera.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # Force RGB conversion
                    
                    # Brightness: Balanced (50%) - avoids clipping (losing detail in highlights/shadows)
                    self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
                    
                    # Contrast: Balanced for better feature extraction
                    self.camera.set(cv2.CAP_PROP_CONTRAST, 0.5)
                    
                    # Saturation: Normal level for skin-tone accuracy
                    self.camera.set(cv2.CAP_PROP_SATURATION, 0.65)
                    
                    # Gain: Automatic for consistent color
                    self.camera.set(cv2.CAP_PROP_GAIN, -1)
                    
                    # White Balance: Auto for adaptive lighting
                    self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)
                    
                    # Exposure: Auto exposure for varying lighting conditions
                    self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                    
                    # Disable autofocus for consistent performance
                    self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    
                    self.camera_active = True
                    app.logger.info("Camera started with optimized settings:")
                    app.logger.info("  - Color Space: RGB (Standard) for skin-tone segmentation")
                    app.logger.info("  - Brightness: Balanced (50%) to avoid clipping")
                    app.logger.info("  - Gamma: 2.2 (applied in preprocessing)")
                    app.logger.info("  - Preprocessing: CLAHE for contrast enhancement")
                except Exception as e:
                    app.logger.error(f"Error starting camera: {e}")
                    if self.camera is not None:
                        self.camera.release()
                    self.camera = None
                    raise
    
    def stop_camera(self):
        """Safely stop and release camera"""
        with self.lock:
            self.camera_active = False
            app.logger.info("Camera stop signal sent")
        
        # Allow time for gen_frames loop to detect flag
        time.sleep(0.5)
        self.release_camera()

    def load_known_faces(self):
        """Load face encodings from database with error handling"""
        try:
            with self.lock:
                encodings, names, emp_ids = self.db_manager.get_employee_encodings()
                self.known_face_encodings = encodings
                self.known_face_names = names
                self.known_employee_ids = emp_ids
                app.logger.info(f"Loaded {len(self.known_face_encodings)} face encodings from database")
        except Exception as e:
            app.logger.error(f"Error loading face encodings: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_employee_ids = []
    
    def add_new_employee(self, employee_id, name, email, department, image_path, created_by=None):
        """Add new employee with enhanced face processing"""
        try:
            # Load and validate image
            image = face_recognition.load_image_file(image_path)
            
            # Handle RGBA images
            if len(image.shape) > 2 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

            # Detect faces
            face_locations = face_recognition.face_locations(image, model='hog')  # Use HOG for better accuracy

            if not face_locations:
                return {'success': False, 'message': 'No face detected in the image. Please use a clear photo with a visible face.'}
            if len(face_locations) > 1:
                return {'success': False, 'message': 'Multiple faces detected. Please use an image with only one face.'}
            
            # Generate face encoding with higher precision
            face_encoding = face_recognition.face_encodings(
                image, 
                known_face_locations=face_locations, 
                num_jitters=10,  # More jitters for better accuracy
                model='large'    # Use large model for better accuracy
            )[0]
            
            # Save to database
            success = self.db_manager.add_employee(employee_id, name, email, department, face_encoding, created_by)
            
            if success:
                self.load_known_faces()  # Reload face encodings
                return {'success': True, 'message': 'Employee added successfully'}
            else:
                return {'success': False, 'message': 'Employee ID already exists or database error occurred'}
                
        except IndexError:
            return {'success': False, 'message': 'Could not generate face encoding. Please use a clearer image.'}
        except Exception as e:
            app.logger.error(f"Error adding new employee {employee_id}: {str(e)}")
            return {'success': False, 'message': f'An error occurred: {str(e)}'}

    def recognize_faces_in_frame(self, frame):
        """
        Optimized face recognition with improved preprocessing
        Uses recommended camera settings:
        - Color Space: RGB for skin-tone segmentation
        - Gamma: 2.2 to prevent dark mid-tones
        - Brightness: Balanced (50%) to avoid clipping
        - Preprocessing: CLAHE for contrast enhancement
        """
        try:
            # Preprocess frame with recommended settings (BGR -> RGB, gamma 2.2, brightness, CLAHE)
            rgb_frame = self.face_utils.preprocess_frame(frame)
            
            # Resize frame for faster processing (reduce to 1/2 size for better accuracy)
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
            
            # Detect faces with optimized parameters
            face_locations = face_recognition.face_locations(small_frame, model='hog', number_of_times_to_upsample=0)
            
            if not face_locations:
                return  # No faces detected
            
            face_encodings = face_recognition.face_encodings(small_frame, face_locations, num_jitters=1)

            # Get current known faces with thread safety
            with self.lock:
                known_encodings = self.known_face_encodings.copy()
                known_names = self.known_face_names.copy()
                known_ids = self.known_employee_ids.copy()

            current_time = datetime.now()
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                name = "Unknown"
                employee_id = None
                
                if known_encodings:  # Only proceed if we have known faces
                    # Compare faces with optimized tolerance (0.6 is more permissive)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        best_distance = face_distances[best_match_index]
                        
                        # Use 0.6 tolerance for better recognition (0.6 is more permissive than 0.5)
                        if best_distance < 0.6:
                            name = known_names[best_match_index]
                            employee_id = known_ids[best_match_index]
                            app.logger.info(f"Face matched: {name} with distance {best_distance:.4f}")
                            
                            # Check cooldown to prevent duplicate recognitions
                            last_time = self.last_detection_time.get(employee_id, datetime.min)
                            if (current_time - last_time).total_seconds() > self.detection_cooldown:
                                # Mark attendance
                                result = self.db_manager.mark_attendance(employee_id)
                                if result['success']:
                                    self.last_detection_time[employee_id] = current_time
                                    
                                    detection_event = {
                                        "name": name,
                                        "employee_id": employee_id,
                                        "time": current_time.strftime('%I:%M:%S %p'),
                                        "status": "checked_in"
                                    }
                                    
                                    # Add to recent detections if it's new or different person
                                    if not self.recent_detections or self.recent_detections[0]['employee_id'] != employee_id:
                                        self.recent_detections.appendleft(detection_event)
                                        app.logger.info(f"Attendance marked for {name} ({employee_id})")
                        else:
                            app.logger.debug(f"Face detected but no match (best distance: {best_distance:.4f})")
                
                # Draw rectangle and label on the frame
                # Scale back up face locations since the frame we detected in was scaled to 1/2 size
                top, right, bottom, left = face_location
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                # Draw bounding box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                
                # Draw label
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
                
        except Exception as e:
            app.logger.error(f"Error in face recognition: {e}")

# --- Optimized Flask App Setup ---
app = Flask(__name__)

# Enhanced security configuration
app.secret_key = secrets.token_hex(32)  # More secure 32-byte key
app.config.update(
    # File upload configuration
    UPLOAD_FOLDER='uploads',
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    
    # Security settings
    WTF_CSRF_ENABLED=True,
    WTF_CSRF_TIME_LIMIT=None,  # No time limit for CSRF tokens
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME=timedelta(hours=24),
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    
    # Performance settings
    SEND_FILE_MAX_AGE_DEFAULT=timedelta(hours=1),
    
    # Optimization flags
    THREADED=True,
    USE_RELOADER=False,  # Disable auto-reload in production
)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize extensions
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

csrf = CSRFProtect(app)
face_system = FaceRecognitionSystem()

from flask_moment import Moment
moment = Moment(app)

@login_manager.user_loader
def load_user(user_id):
    return face_system.db_manager.get_user_by_id(int(user_id))

# --- Authentication Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = face_system.db_manager.authenticate_user(form.username.data, form.password.data)
        if user:
            login_user(user, remember=True)
            next_page = request.args.get('next')
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = SignupForm()
    if form.validate_on_submit():
        success = face_system.db_manager.create_user(
            form.username.data,
            form.email.data,
            form.password.data,
            form.role.data
        )
        if success:
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('An error occurred while creating your account.', 'error')
    
    return render_template('signup.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    return render_template('profile.html', user=current_user)

@app.route('/settings')
@login_required
def settings():
    """User settings page"""
    return render_template('settings.html', user=current_user)

# --- Web Routes ---
@app.route('/')
@login_required
def dashboard():
    summary = face_system.db_manager.get_attendance_summary()
    recent_attendance_records = face_system.db_manager.get_attendance_records(days=3)
    
    # Convert check_in_time strings to datetime objects for moment filter
    recent_attendance = []
    for record in recent_attendance_records:
        record_dict = dict(record)
        if record_dict.get('check_in_time') and isinstance(record_dict['check_in_time'], str):
            try:
                record_dict['check_in_time'] = datetime.fromisoformat(record_dict['check_in_time'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        recent_attendance.append(record_dict)
    
    return render_template('dashboard.html', summary=summary, recent_attendance=recent_attendance)

@app.route('/employees')
@login_required
def employees():
    employees_data = face_system.db_manager.get_all_employees()
    
    # Convert created_at strings to datetime objects for template rendering
    for emp in employees_data:
        if emp.get('created_at') and isinstance(emp['created_at'], str):
            try:
                emp['created_at'] = datetime.strptime(emp['created_at'], '%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                app.logger.warning(f"Could not parse created_at for employee {emp.get('employee_id')}")
    
    return render_template('employees.html', employees=employees_data)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/add_employee', methods=['GET', 'POST'])
@login_required
@admin_required
def add_employee():
    form = EmployeeForm()
    if form.validate_on_submit():
        employee_id = form.employee_id.data.strip()
        name = form.name.data.strip()
        email = form.email.data.strip()
        department = form.department.data.strip()
        
        photo = form.photo.data
        if photo and allowed_file(photo.filename):
            filename = secure_filename(photo.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            photo.save(filepath)
            
            result = face_system.add_new_employee(employee_id, name, email, department, filepath, current_user.id)
            
            os.remove(filepath) 
            
            if result['success']:
                flash(result['message'], 'success')
                return redirect(url_for('employees'))
            else:
                flash(result['message'], 'error')
        else:
            flash('Invalid file type. Please use png, jpg, or jpeg.', 'error')

    return render_template('add_employee.html', form=form)

@app.route('/attendance')
@login_required
def attendance():
    """View attendance records"""
    try:
        # Allow employees to see their own attendance, admins can see all
        employee_id = None if current_user.is_admin() else current_user.employee_id
        records_from_db = face_system.db_manager.get_attendance_records(days=30, employee_id=employee_id)

        processed_records = []
        for record in records_from_db:
            record_dict = dict(record)
            duration_str = "-"

            # Handle datetime conversion and duration calculation
            if record_dict.get('check_in_time') and record_dict.get('check_out_time'):
                try:
                    check_in = record_dict['check_in_time']
                    check_out = record_dict['check_out_time']

                    # Convert strings to datetime objects if needed
                    if isinstance(check_in, str):
                        check_in = datetime.fromisoformat(check_in.replace('Z', '+00:00'))
                    if isinstance(check_out, str):
                        check_out = datetime.fromisoformat(check_out.replace('Z', '+00:00'))

                    # Calculate duration
                    duration = check_out - check_in
                    hours, remainder = divmod(duration.total_seconds(), 3600)
                    minutes, _ = divmod(remainder, 60)
                    duration_str = f"{int(hours)}h {int(minutes)}m"
                except (ValueError, TypeError, AttributeError) as e:
                    app.logger.error(f"Error calculating duration: {e}")
                    duration_str = "Error"

            record_dict['duration'] = duration_str

            # Fix date formatting - convert string date to datetime object for moment filter
            if record_dict.get('date'):
                date_value = record_dict['date']
                if isinstance(date_value, str):
                    # Convert string date to datetime object
                    try:
                        record_dict['date'] = datetime.strptime(date_value, '%Y-%m-%d').date()
                        record_dict['date_iso'] = date_value
                    except ValueError:
                        try:
                            parsed_date = datetime.fromisoformat(date_value)
                            record_dict['date'] = parsed_date.date() if hasattr(parsed_date, 'date') else parsed_date
                            record_dict['date_iso'] = str(date_value)
                        except (ValueError, AttributeError):
                            record_dict['date_iso'] = str(date_value)
                else:
                    # If it's already a date/datetime object, convert to ISO format string
                    try:
                        record_dict['date_iso'] = date_value.isoformat()
                    except AttributeError:
                        record_dict['date_iso'] = str(date_value)
            else:
                record_dict['date_iso'] = ''

            record_dict['work_hours_formatted'] = f"{record_dict.get('work_hours', 0):.1f}h" if record_dict.get('work_hours') else '-'
            record_dict['overtime_hours_formatted'] = f"{record_dict.get('overtime_hours', 0):.1f}h" if record_dict.get('overtime_hours', 0) > 0 else '-'
            processed_records.append(record_dict)

        return render_template('attendance.html',
                               records=processed_records,
                               is_admin=current_user.is_admin())
    except Exception as e:
        app.logger.error(f"Error in attendance view: {e}")
        flash('An error occurred while loading attendance records.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/delete_employee/<employee_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_employee(employee_id):
    success = face_system.db_manager.delete_employee(employee_id)
    if success:
        face_system.load_known_faces()
        app.logger.info(f"Employee {employee_id} deleted by user {current_user.username}")
        return jsonify({'success': True, 'message': 'Employee deleted successfully.'})
    else:
        return jsonify({'success': False, 'message': 'Employee not found.'}), 404

@app.route('/mark_attendance', methods=['POST'])
@login_required
def mark_attendance():
    data = request.json
    employee_id = data.get('employee_id')
    attendance_type = data.get('type', 'check_in')
    result = face_system.db_manager.mark_attendance(employee_id, attendance_type)
    return jsonify(result)

# --- Optimized Video Streaming and Camera Control ---
def gen_frames():
    """Optimized video frame generation with better performance control"""
    import time
    
    # Initialize camera with thread safety
    with face_system.lock:
        if face_system.camera is None and not face_system.camera_active:
            try:
                face_system.camera = cv2.VideoCapture(0)
                if not face_system.camera.isOpened():
                    app.logger.error("Could not open camera")
                    return
                
                # Optimize camera settings for better performance
                face_system.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                face_system.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                face_system.camera.set(cv2.CAP_PROP_FPS, 30)  # Higher FPS for better detection
                face_system.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize delay
                face_system.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for performance
                
                # Ensure color mode (not grayscale)
                face_system.camera.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # Force color conversion
                face_system.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
                face_system.camera.set(cv2.CAP_PROP_CONTRAST, 0.5)
                face_system.camera.set(cv2.CAP_PROP_SATURATION, 0.65)
                face_system.camera.set(cv2.CAP_PROP_AUTO_WB, 1)
                face_system.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                
                face_system.camera_active = True
                app.logger.info("Camera initialized with optimized settings")
            except Exception as e:
                app.logger.error(f"Error initializing camera: {e}")
                return
        elif not face_system.camera_active:
            app.logger.info("Camera stop requested, not starting video generation")
            return

    frame_count = 0
    process_every_n_frames = 2  # Process every 2nd frame for performance optimization
    last_frame_time = time.time()
    target_fps = 24  # Streaming at 24 FPS
    frame_delay = 1.0 / target_fps
    
    try:
        while True:
            current_time = time.time()
            
            # Check camera state
            with face_system.lock:
                if not face_system.camera_active or face_system.camera is None:
                    app.logger.info("Camera stop detected, exiting video generation loop")
                    break
                    
                success, frame = face_system.camera.read()
                
            if not success:
                app.logger.warning("Failed to read frame from camera")
                break
                
            if not face_system.camera_active:
                app.logger.info("Camera stop detected after frame read")
                break
            
            # Ensure frame is readable and in color
            if frame is None or frame.size == 0:
                app.logger.warning("Invalid frame received")
                continue
            
            # Ensure frame is in BGR color format (3 channels)
            if len(frame.shape) == 2:
                # Grayscale 2D - convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3:
                if frame.shape[2] == 1:
                    # Single channel 3D - convert to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    # RGBA - convert to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] != 3:
                    # If not 3 channels, convert to BGR
                    app.logger.warning(f"Unexpected frame channels: {frame.shape[2]}")
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # IMPORTANT: Make a copy for display to preserve color during processing
            display_frame = frame.copy()
            
            # Process face recognition only on every nth frame for performance
            frame_count += 1
            if frame_count % process_every_n_frames == 0:
                try:
                    # Pass display_frame for processing
                    face_system.recognize_faces_in_frame(display_frame)
                except Exception as e:
                    app.logger.error(f"Error in face recognition: {e}")
            
            # Encode display_frame for streaming (guaranteed to be BGR 3-channel)
            try:
                # Verify display_frame has 3 channels for JPEG encoding
                if len(display_frame.shape) != 3 or display_frame.shape[2] != 3:
                    app.logger.error(f"Invalid frame shape for encoding: {display_frame.shape}")
                    continue
                
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # Quality: 80/100
                ret, buffer = cv2.imencode('.jpg', display_frame, encode_param)
                if ret:
                    frame_bytes = buffer.tobytes()
                    
                    # Final check before yielding
                    if face_system.camera_active:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        break
                else:
                    app.logger.error("Failed to encode frame to JPEG")
                    continue
                    
            except Exception as e:
                app.logger.error(f"Error encoding frame: {e}")
                break
            
            # Frame rate control
            elapsed = time.time() - last_frame_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
            last_frame_time = time.time()
                
    except Exception as e:
        app.logger.error(f"Unexpected error in gen_frames: {e}")
    finally:
        app.logger.info("Video frame generation ended")

@app.route('/video_feed')
@login_required
def video_feed():
    """Stream processed video feed with face recognition"""
    try:
        face_system.start_camera()
        return Response(
            gen_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        app.logger.error(f"Error in video feed: {e}")
        return Response(
            b'--frame\r\nContent-Type: image/jpeg\r\n\r\n',
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

@app.route('/recent_detections')
@login_required
def recent_detections():
    """Get list of recent face detections"""
    return jsonify(list(face_system.recent_detections))

@app.route('/recognition')
@login_required
def recognition():
    """Render recognition page with enhanced features"""
    return render_template(
        'recognition.html',
        detection_cooldown=face_system.detection_cooldown,
        camera_status=face_system.camera is not None
    )

@app.route('/stop_video_feed', methods=['POST'])
@login_required
def stop_video_feed():
    """Stop video feed and release camera resources"""
    try:
        app.logger.info(f"Camera stop requested by user {current_user.username}")
        face_system.stop_camera()
        response = jsonify({
            'success': True,
            'message': 'Camera stopped successfully'
        })
        response.headers['Content-Type'] = 'application/json'
        return response
    except Exception as e:
        app.logger.error(f"Error stopping camera: {e}")
        response = jsonify({
            'success': False,
            'message': 'Error stopping camera'
        })
        response.headers['Content-Type'] = 'application/json'
        return response, 500

@app.route('/start_video_feed', methods=['POST'])
@login_required
def start_video_feed():
    """Initialize and start the video feed"""
    try:
        app.logger.info(f"Camera start requested by user {current_user.username}")
        face_system.start_camera()
        return jsonify({'success': True, 'message': 'Camera started successfully'})
    except Exception as e:
        app.logger.error(f"Error starting camera: {e}")
        return jsonify({'success': False, 'message': f'Error starting camera: {str(e)}'}), 500

# --- Advanced Features ---
@app.route('/analytics')
@login_required
@admin_required
def analytics():
    """Advanced analytics dashboard for admins"""
    try:
        # Get analytics data with error handling
        with face_system.db_manager.get_db_connection() as conn:
            # Weekly attendance trends
            cursor = conn.execute('''
                SELECT DATE(date) as attendance_date, COUNT(DISTINCT employee_id) as daily_count
                FROM attendance 
                WHERE date >= date('now', '-30 days')
                GROUP BY DATE(date)
                ORDER BY attendance_date DESC
            ''')
            raw_weekly_trends = cursor.fetchall()
            
            # Convert attendance_date strings to datetime objects for moment filter
            weekly_trends = []
            for trend in raw_weekly_trends:
                trend_dict = dict(trend)
                if isinstance(trend_dict['attendance_date'], str):
                    try:
                        trend_dict['attendance_date'] = datetime.strptime(trend_dict['attendance_date'], '%Y-%m-%d').date()
                    except ValueError:
                        pass
                weekly_trends.append(trend_dict)

            # Department-wise attendance (check if department column exists)
            cursor = conn.execute("PRAGMA table_info(employees)")
            columns = [column[1] for column in cursor.fetchall()]

            if 'department' in columns:
                cursor = conn.execute('''
                    SELECT COALESCE(e.department, 'Unknown') as department, 
                           COUNT(DISTINCT a.employee_id) as present_count,
                           COUNT(DISTINCT e.employee_id) as total_count
                    FROM employees e
                    LEFT JOIN attendance a ON e.employee_id = a.employee_id 
                        AND a.date >= date('now', '-7 days')
                    WHERE (e.is_active = 1 OR e.is_active IS NULL)
                    GROUP BY e.department
                ''')
                dept_stats = cursor.fetchall()
            else:
                dept_stats = []

            # Top performers (highest attendance rate)
            cursor = conn.execute('''
                SELECT e.name, e.employee_id, 
                       COALESCE(e.department, 'Unknown') as department,
                       COUNT(a.id) as days_present,
                       ROUND(AVG(COALESCE(a.work_hours, 0)), 1) as avg_hours
                FROM employees e
                LEFT JOIN attendance a ON e.employee_id = a.employee_id 
                    AND a.date >= date('now', '-30 days')
                WHERE (e.is_active = 1 OR e.is_active IS NULL)
                GROUP BY e.employee_id
                ORDER BY days_present DESC
                LIMIT 10
            ''')
            top_performers = cursor.fetchall()

    except sqlite3.OperationalError as e:
        app.logger.error(f"Analytics database error: {e}")
        # Provide empty data if there's a database error
        weekly_trends = []
        dept_stats = []
        top_performers = []
        flash('Some analytics data may not be available due to database schema differences.', 'warning')

    return render_template('analytics.html', 
                         weekly_trends=weekly_trends,
                         dept_stats=dept_stats,
                         top_performers=top_performers)

@app.route('/api/attendance_chart_data')
@login_required
def attendance_chart_data():
    """API endpoint for chart data"""
    if not current_user.is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        with face_system.db_manager.get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT DATE(date) as date, COUNT(DISTINCT employee_id) as count
                FROM attendance 
                WHERE date >= date('now', '-14 days')
                GROUP BY DATE(date)
                ORDER BY date ASC
            ''')
            data = [dict(row) for row in cursor.fetchall()]
        return jsonify(data)
    except Exception as e:
        app.logger.error(f"Chart data error: {e}")
        return jsonify([])

# --- Main Application Runner ---
if __name__ == '__main__':
    # Configure logging
    if not app.debug:
        handler = RotatingFileHandler('attendance_system.log', maxBytes=10000000, backupCount=3)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        handler.setLevel(logging.INFO)
        app.logger.addHandler(handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Attendance System startup')
    
    print("\n" + "="*60)
    print("   FACE RECOGNITION ATTENDANCE SYSTEM")
    print("="*60)
    print("Default admin credentials:")
    print("Username: admin")
    print(f"Password: {os.environ.get('ADMIN_PASSWORD', 'admin123')}")
    if not os.environ.get("ADMIN_PASSWORD"):
        print("\nPlease change the default password after first login!")
    print("="*60 + "\n")
    
    # Use threaded=True to handle multiple requests, like video streaming and API calls
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

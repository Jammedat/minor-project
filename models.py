# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import numpy as np

db = SQLAlchemy()

class Teacher(db.Model):
    __tablename__ = 'teachers'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)  # store hash
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Department(db.Model):
    __tablename__ = 'departments'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    batches = db.relationship('Batch', backref='department', lazy='dynamic')

class Batch(db.Model):
    __tablename__ = 'batches'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)  # e.g., "2024"
    department_id = db.Column(db.Integer, db.ForeignKey('departments.id'), nullable=False)
    students = db.relationship('Student', backref='batch', lazy='dynamic')
    subjects = db.relationship('Subject', backref='batch', lazy='dynamic')

class Subject(db.Model):
    __tablename__ = 'subjects'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    batch_id = db.Column(db.Integer, db.ForeignKey('batches.id'), nullable=False)
    # You might also want to link to teacher if needed
    attendance = db.relationship('Attendance', backref='subject', lazy='dynamic')

class Student(db.Model):
    __tablename__ = 'students'
    roll_no = db.Column(db.String(20), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    department_id = db.Column(db.Integer, db.ForeignKey('departments.id'), nullable=False)
    batch_id = db.Column(db.Integer, db.ForeignKey('batches.id'), nullable=False)
    registered_on = db.Column(db.DateTime, default=datetime.utcnow)
    embeddings = db.relationship('FaceEmbedding', backref='student', lazy='dynamic')
    attendances = db.relationship('Attendance', backref='student', lazy='dynamic')

class FaceEmbedding(db.Model):
    __tablename__ = 'face_embeddings'
    id = db.Column(db.Integer, primary_key=True)
    student_roll_no = db.Column(db.String(20), db.ForeignKey('students.roll_no'), nullable=False)
    embedding_blob = db.Column(db.LargeBinary, nullable=False)  # serialized numpy array
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True)
    student_roll_no = db.Column(db.String(20), db.ForeignKey('students.roll_no'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    method = db.Column(db.String(20), default='face')  # 'face' or 'iris'
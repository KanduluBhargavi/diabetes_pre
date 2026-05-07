
"""
AI Diabetes Risk Analyzer — FastAPI Backend
PostgreSQL + JWT Authentication + Patient Records

Run:
    pip install -r requirements.txt
    python main.py   ← creates tables automatically
    uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import joblib, pandas as pd, os

# ══════════════════════════════════
#  CONFIG
# ══════════════════════════════════
DATABASE_URL = "postgresql://postgres:post%40bha17@localhost:5432/diabetes_db"
SECRET_KEY    = "glycocast_secret_2024"
ALGORITHM     = "HS256"
TOKEN_EXPIRE  = 60 * 24  # 24 hours

# ══════════════════════════════════
#  DATABASE
# ══════════════════════════════════
engine       = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String(100), nullable=False)
    email         = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role          = Column(String(20), nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)
    patients      = relationship("Patient", back_populates="doctor", cascade="all, delete")

class Patient(Base):
    __tablename__ = "patients"
    id         = Column(Integer, primary_key=True, index=True)
    doctor_id  = Column(Integer, ForeignKey("users.id"), nullable=False)
    name       = Column(String(100), nullable=False)
    age        = Column(Integer)
    gender     = Column(String(10))
    visits     = Column(JSON, default=[])
    notes      = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    doctor     = relationship("User", back_populates="patients")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ══════════════════════════════════
#  AUTH HELPERS
# ══════════════════════════════════
pwd_ctx = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
    bcrypt__ident="2b"
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def hash_pw(pw):
    pw = pw[:72] if len(pw.encode()) > 72 else pw
    return pwd_ctx.hash(pw)

def verify_pw(p, h):
    p = p[:72] if len(p.encode()) > 72 else p
    return pwd_ctx.verify(p, h)

def make_token(data):
    d = data.copy()
    d["exp"] = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE)
    return jwt.encode(d, SECRET_KEY, algorithm=ALGORITHM)

def get_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid = payload.get("sub")
        if not uid: raise Exception()
    except:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.query(User).filter(User.id == int(uid)).first()
    if not user: raise HTTPException(status_code=401, detail="User not found")
    return user

def doctor_only(user: User = Depends(get_user)):
    if user.role != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access only")
    return user

# ══════════════════════════════════
#  APP
# ══════════════════════════════════
app = FastAPI(title="AI Diabetes Risk Analyzer v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Load ML model
BASE = os.path.dirname(os.path.abspath(__file__))
try:
    model  = joblib.load(os.path.join(BASE, "models/diabetes_model.pkl"))
    scaler = joblib.load(os.path.join(BASE, "models/scaler.pkl"))
    print("✅ ML models loaded")
except Exception as e:
    print(f"⚠️ {e}")
    model = scaler = None

SMOKING = {"never": 0, "former": 1, "current": 2}

# ══════════════════════════════════
#  SCHEMAS
# ══════════════════════════════════
class RegisterIn(BaseModel):
    name: str
    email: str
    password: str
    role: str

class LoginIn(BaseModel):
    email: str
    password: str

class PredictIn(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    bmi: float
    hba1c: float
    glucose: float
    smoking_status: str
    mode: Optional[str] = "general"

class VisitIn(BaseModel):
    date: str
    pct: float
    bmi: float
    hba1c: float
    glucose: float
    hypertension: int
    heart_disease: int
    smoking: str

class SavePatientIn(BaseModel):
    name: str
    age: int
    gender: str
    visit: VisitIn

class NotesIn(BaseModel):
    notes: str

# ══════════════════════════════════
#  ROUTES
# ══════════════════════════════════
@app.get("/")
def root(): return {"status": "AI Diabetes Risk Analyzer v2 ✅"}

@app.post("/register")
def register(data: RegisterIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(400, "Email already registered")
    if data.role not in ["general", "doctor"]:
        raise HTTPException(400, "Role must be general or doctor")
    user = User(name=data.name, email=data.email, password_hash=hash_pw(data.password), role=data.role)
    db.add(user); db.commit(); db.refresh(user)
    token = make_token({"sub": str(user.id), "role": user.role})
    return {"token": token, "user": {"id": user.id, "name": user.name, "email": user.email, "role": user.role}}

@app.post("/login")
def login(data: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not verify_pw(data.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    token = make_token({"sub": str(user.id), "role": user.role})
    return {"token": token, "user": {"id": user.id, "name": user.name, "email": user.email, "role": user.role}}

@app.get("/me")
def me(user: User = Depends(get_user)):
    return {"id": user.id, "name": user.name, "email": user.email, "role": user.role}

@app.post("/predict")
def predict(data: PredictIn, user: User = Depends(get_user)):
    if not model or not scaler:
        raise HTTPException(503, "ML model not loaded")

    gender  = 1 if data.gender.lower() == "male" else 0
    smoking = SMOKING.get(data.smoking_status.lower(), 0)

    df = pd.DataFrame([[gender, data.age, data.hypertension, data.heart_disease,
                        smoking, data.bmi, data.hba1c, data.glucose]],
                      columns=["gender","age","hypertension","heart_disease",
                               "smoking_history","bmi","HbA1c_level","blood_glucose_level"])

    prob = float(model.predict_proba(scaler.transform(df))[0][1])

    # Clinical penalty adjustment
    # (Framingham Risk Score methodology)
    penalty = 0.0
    if data.hypertension == 1:            penalty += 0.05
    if data.heart_disease == 1:           penalty += 0.05
    if data.smoking_status == "current":  penalty += 0.05
    if data.smoking_status == "former":   penalty += 0.02
    if data.hypertension == 1 and data.heart_disease == 1:            penalty += 0.03
    if data.hypertension == 1 and data.smoking_status == "current":   penalty += 0.03
    if data.heart_disease == 1 and data.smoking_status == "current":  penalty += 0.03

    # KEY FIX — risk can NEVER decrease when comorbidities are added
    adj = max(prob, min(0.99, prob + penalty))
    return {
        "risk": adj,
        "risk_percent": round(adj * 100, 1),
        "risk_level": "High" if adj > 0.30 else "Moderate" if adj > 0.15 else "Low",
        "model_raw": round(prob, 4),
        "clinical_adjustment": round(penalty, 4),
        "mode": data.mode
    }

# ── Patient routes (doctor only) ──
@app.get("/patients")
def get_patients(user: User = Depends(doctor_only), db: Session = Depends(get_db)):
    pts = db.query(Patient).filter(Patient.doctor_id == user.id).all()
    return [{"id":p.id,"name":p.name,"age":p.age,"gender":p.gender,
             "visits":p.visits or [],"notes":p.notes or ""} for p in pts]

@app.post("/patients/save")
def save_patient(data: SavePatientIn, user: User = Depends(doctor_only), db: Session = Depends(get_db)):
    p = db.query(Patient).filter(Patient.doctor_id==user.id, Patient.name==data.name).first()
    visit = data.visit.dict()
    if p:
        visits = list(p.visits or [])
        visits.append(visit)
        p.visits = visits
        p.updated_at = datetime.utcnow()
    else:
        p = Patient(doctor_id=user.id, name=data.name, age=data.age, gender=data.gender, visits=[visit])
        db.add(p)
    db.commit(); db.refresh(p)
    return {"message": f"Saved — {data.name}", "patient_id": p.id, "total_visits": len(p.visits)}

@app.put("/patients/{pid}/notes")
def update_notes(pid: int, data: NotesIn, user: User = Depends(doctor_only), db: Session = Depends(get_db)):
    p = db.query(Patient).filter(Patient.id==pid, Patient.doctor_id==user.id).first()
    if not p: raise HTTPException(404, "Patient not found")
    p.notes = data.notes; db.commit()
    return {"message": "Notes saved"}

@app.delete("/patients/{pid}")
def delete_patient(pid: int, user: User = Depends(doctor_only), db: Session = Depends(get_db)):
    p = db.query(Patient).filter(Patient.id==pid, Patient.doctor_id==user.id).first()
    if not p: raise HTTPException(404, "Patient not found")
    db.delete(p); db.commit()
    return {"message": "Patient deleted"}

# ══════════════════════════════════
#  STARTUP — create tables
# ══════════════════════════════════
if __name__ == "__main__":
    print("Creating tables..."); Base.metadata.create_all(bind=engine); print("✅ Done")
    import uvicorn; uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

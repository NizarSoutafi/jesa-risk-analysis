import asyncpg
import jwt
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
from passlib.context import CryptContext
import logging
from datetime import datetime, timedelta
import os

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = "postgresql://mac@localhost:5432/jesa_risk_analysis"

# JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")  # Replace with a secure key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pydantic models
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    is_admin: Optional[bool] = False

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Database dependency
class Database:
    def __init__(self):
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(DATABASE_URL)
        logger.debug("Database pool created")

    async def disconnect(self):
        await self.pool.close()
        logger.debug("Database pool closed")

db = Database()

@app.on_event("startup")
async def startup():
    await db.connect()

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()

def get_db():
    return db

# JWT Bearer dependency
class JWTBearer(HTTPBearer):
    async def __call__(self, request: Request):
        auth = await super().__call__(request)
        data = jwt.decode(auth.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        request.state.user = data
        async with db.pool.acquire() as conn:
            user = await conn.fetchrow("SELECT * FROM users WHERE email = $1", data["email"])
            if not user:
                raise HTTPException(status_code=403, detail="Invalid token")
        return auth

# Create JWT token
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Endpoints
@app.post("/api/auth/signup")
async def signup(user: UserCreate, request: Request, db: Database = Depends(get_db)):
    logger.debug(f"Received signup request: name='{user.name}' email='{user.email}' password='{user.password}'")
    async with db.pool.acquire() as conn:
        # Check if user already exists
        existing_user = await conn.fetchrow("SELECT * FROM users WHERE email = $1", user.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Check if the request is from an admin (has Authorization header)
        is_admin_request = hasattr(request.state, 'user')
        if is_admin_request:
            user_data = request.state.user
            db_user = await conn.fetchrow("SELECT is_admin FROM users WHERE email = $1", user_data["email"])
            if not db_user or not db_user["is_admin"]:
                raise HTTPException(status_code=403, detail="Admin access required")
            # Admins can set is_admin field
            is_admin = user.is_admin if hasattr(user, 'is_admin') else False
        else:
            # Non-admin signup (public registration)
            if len(user.password) < 8:
                logger.error("Password too short")
                raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")
            is_admin = False

        # Hash the password
        hashed_password = pwd_context.hash(user.password)
        await conn.execute(
            "INSERT INTO users (name, email, password, is_admin) VALUES ($1, $2, $3, $4)",
            user.name,
            user.email,
            hashed_password,
            is_admin
        )
    return {"message": "User registered successfully"}

@app.post("/api/auth/login")
async def login(user: UserLogin, db: Database = Depends(get_db)):
    logger.debug(f"Received login request: email='{user.email}' password='{user.password}'")
    async with db.pool.acquire() as conn:
        db_user = await conn.fetchrow("SELECT * FROM users WHERE email = $1", user.email)
        if not db_user or not pwd_context.verify(user.password, db_user["password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_access_token({"email": user.email})
        return {
            "access_token": token,
            "token_type": "bearer",
            "is_admin": db_user["is_admin"]
        }

@app.get("/api/admin/users", dependencies=[Depends(JWTBearer())])
async def get_users(request: Request, db: Database = Depends(get_db)):
    user = request.state.user
    async with db.pool.acquire() as conn:
        db_user = await conn.fetchrow("SELECT is_admin FROM users WHERE email = $1", user["email"])
        if not db_user or not db_user["is_admin"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        users = await conn.fetch("SELECT id, name, email, is_admin FROM users")
        return [{"id": user["id"], "name": user["name"], "email": user["email"], "is_admin": user["is_admin"]} for user in users]

@app.delete("/api/admin/users/{user_id}", dependencies=[Depends(JWTBearer())])
async def delete_user(user_id: int, request: Request, db: Database = Depends(get_db)):
    user = request.state.user
    async with db.pool.acquire() as conn:
        db_user = await conn.fetchrow("SELECT is_admin FROM users WHERE email = $1", user["email"])
        if not db_user or not db_user["is_admin"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        # Prevent admin from deleting themselves
        target_user = await conn.fetchrow("SELECT email FROM users WHERE id = $1", user_id)
        if target_user["email"] == user["email"]:
            raise HTTPException(status_code=400, detail="Cannot delete yourself")
        result = await conn.execute("DELETE FROM users WHERE id = $1", user_id)
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="User not found")
        return {"message": "User deleted successfully"}

@app.put("/api/admin/users/{user_id}/toggle-admin", dependencies=[Depends(JWTBearer())])
async def toggle_admin(user_id: int, request: Request, db: Database = Depends(get_db)):
    user = request.state.user
    async with db.pool.acquire() as conn:
        db_user = await conn.fetchrow("SELECT is_admin FROM users WHERE email = $1", user["email"])
        if not db_user or not db_user["is_admin"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        # Prevent admin from toggling their own status
        target_user = await conn.fetchrow("SELECT email, is_admin FROM users WHERE id = $1", user_id)
        if not target_user:
            raise HTTPException(status_code=404, detail="User not found")
        if target_user["email"] == user["email"]:
            raise HTTPException(status_code=400, detail="Cannot toggle your own admin status")
        new_status = not target_user["is_admin"]
        await conn.execute("UPDATE users SET is_admin = $1 WHERE id = $2", new_status, user_id)
        return {"message": "Admin status updated"}

@app.get("/api/admin/check", dependencies=[Depends(JWTBearer())])
async def check_admin_status(request: Request, db: Database = Depends(get_db)):
    user = request.state.user
    async with db.pool.acquire() as conn:
        db_user = await conn.fetchrow("SELECT is_admin FROM users WHERE email = $1", user["email"])
        if not db_user or not db_user["is_admin"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        return {"is_admin": db_user["is_admin"]}
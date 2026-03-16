from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# ==========================================
# CẤU HÌNH BẢO MẬT JWT
# ==========================================
SECRET_KEY = "e2e-heart-disease-super-secret-key-2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# OAuth2 scheme: FastAPI tự động đọc token từ Header "Authorization: Bearer <token>"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# ==========================================
# HÀM MÃ HOÁ MẬT KHẨU (dùng bcrypt trực tiếp)
# ==========================================
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# ==========================================
# DATABASE GIẢ LẬP (Demo - Thực tế dùng PostgreSQL)
# ==========================================
# Tạo hash trước để tránh hash lại mỗi lần import
_admin_hash = hash_password("admin123")
_doctor_hash = hash_password("doctor123")

fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "System Administrator",
        "role": "admin",
        "hashed_password": _admin_hash,
    },
    "doctor": {
        "username": "doctor",
        "full_name": "Dr. Nguyen Van A",
        "role": "doctor",
        "hashed_password": _doctor_hash,
    },
}

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserOut(BaseModel):
    username: str
    full_name: str
    role: str

# ==========================================
# CÁC HÀM XỬ LÝ LOGIC
# ==========================================
def authenticate_user(username: str, password: str):
    """Xác thực người dùng: tìm trong DB, kiểm tra mật khẩu"""
    user = fake_users_db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Tạo JWT Token chứa username + thời gian hết hạn"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency Injection: Giải mã token, kiểm tra username.
    Token sai/hết hạn -> trả lỗi 401 Unauthorized.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token không hợp lệ hoặc đã hết hạn. Vui lòng đăng nhập lại.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = fake_users_db.get(username)
    if user is None:
        raise credentials_exception
    return user

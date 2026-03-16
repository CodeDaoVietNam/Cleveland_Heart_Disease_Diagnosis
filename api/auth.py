from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# ==========================================
# CẤU HÌNH BẢO MẬT JWT
# ==========================================
# SECRET_KEY: Chuỗi bí mật dùng để mã hoá token. Trong thực tế nên đặt trong .env
SECRET_KEY = "e2e-heart-disease-super-secret-key-2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Token hết hạn sau 60 phút

# Mã hoá mật khẩu bằng bcrypt (chuẩn công nghiệp)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme: FastAPI tự động đọc token từ Header "Authorization: Bearer <token>"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# ==========================================
# DATABASE GIẢ LẬP (Demo - Thực tế dùng PostgreSQL)
# ==========================================
# Mật khẩu đã được hash bằng bcrypt. Plaintext gốc:
#   admin -> "admin123"
#   doctor -> "doctor123"
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "System Administrator",
        "role": "admin",
        "hashed_password": pwd_context.hash("admin123"),
    },
    "doctor": {
        "username": "doctor",
        "full_name": "Dr. Nguyen Van A",
        "role": "doctor",
        "hashed_password": pwd_context.hash("doctor123"),
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
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """So sánh mật khẩu gốc với mật khẩu đã mã hoá"""
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    """Xác thực người dùng: tìm trong DB, kiểm tra mật khẩu"""
    user = fake_users_db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Tạo JWT Token.
    Token chứa: username + thời gian hết hạn + chữ ký mã hoá.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency Injection: Hàm này tự động chạy TRƯỚC mỗi endpoint được bảo vệ.
    Nó giải mã token, kiểm tra username có hợp lệ không.
    Nếu token sai/hết hạn -> trả lỗi 401 Unauthorized.
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

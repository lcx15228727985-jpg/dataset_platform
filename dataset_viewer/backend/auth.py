"""
用户认证：10 个用户 000-009，默认密码 123456，root 权限。
优先从 MongoDB 读用户（鲁棒性好），无 MongoDB 时回退内存。
"""
import os
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_SECRET = os.environ.get("JWT_SECRET", "dataset-annotation-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 72

# 内存回退：10 个用户 000-009（无 MongoDB 时使用）
USERS = {f"{i:03d}": {"password": "123456", "role": "root"} for i in range(10)}

security = HTTPBearer(auto_error=False)


def _get_user_in_memory(user_id: str) -> Optional[dict]:
    """从内存获取用户"""
    if user_id not in USERS:
        return None
    return {"_id": user_id, **USERS[user_id]}


async def _get_user_from_mongo(user_id: str) -> Optional[dict]:
    """从 MongoDB 获取用户"""
    try:
        from .mongo_db import get_user
        return await get_user(user_id)
    except Exception:
        return None


async def verify_user_async(user_id: str, password: str) -> Optional[dict]:
    """验证用户，返回用户信息或 None"""
    user = await _get_user_from_mongo(user_id)
    if user is None:
        user = _get_user_in_memory(user_id)
    if user is None:
        return None
    if user.get("password") != password:
        return None
    return user


def create_token(user_id: str, role: str = "root") -> str:
    """生成 JWT 令牌"""
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """解码 JWT，失败返回 None"""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception:
        return None


async def _user_exists(user_id: str) -> bool:
    """检查用户是否存在（MongoDB 优先，否则内存）"""
    user = await _get_user_from_mongo(user_id)
    if user is not None:
        return True
    return _get_user_in_memory(user_id) is not None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """
    从 Authorization: Bearer <token> 中解析当前用户。
    未提供或无效时返回 None。
    """
    if not credentials or not credentials.credentials:
        return None
    payload = decode_token(credentials.credentials)
    if not payload or "sub" not in payload:
        return None
    user_id = payload["sub"]
    if not await _user_exists(user_id):
        return None
    return {"user_id": user_id, "role": payload.get("role", "root")}


async def require_user(
    current: Optional[dict] = Depends(get_current_user),
) -> dict:
    """依赖：必须登录，否则 401"""
    if not current:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先登录后再保存标注",
        )
    return current

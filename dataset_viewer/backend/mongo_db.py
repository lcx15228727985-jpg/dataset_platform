"""
MongoDB 异步访问：用户存储（000-009），提升鲁棒性与并发。
当 MONGODB_URI 设置时使用，否则 auth 回退到内存用户。
"""
import os
from typing import Any, Optional

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "annotation"
USERS_COLL = "users"

_client: Any = None


async def get_client():
    """获取 MongoDB 异步客户端（单例）"""
    global _client
    if _client is None:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            _client = AsyncIOMotorClient(MONGODB_URI)
            await _client.admin.command("ping")
        except Exception:
            _client = None
    return _client


def close_client():
    """关闭 MongoDB 连接（同步）"""
    global _client
    if _client:
        _client.close()
        _client = None


async def get_users_collection():
    """获取 users 集合"""
    client = await get_client()
    if not client:
        return None
    return client[DB_NAME][USERS_COLL]


async def ensure_users_seeded() -> bool:
    """
    确保 10 个用户 000-009 已写入 MongoDB。
    若集合为空则写入，否则跳过。
    """
    coll = await get_users_collection()
    if not coll:
        return False
    try:
        n = await coll.count_documents({})
        if n > 0:
            return True
        users = [
            {"_id": f"{i:03d}", "password": "123456", "role": "root"}
            for i in range(10)
        ]
        await coll.insert_many(users)
        return True
    except Exception:
        return False


async def get_user(user_id: str) -> Optional[dict]:
    """从 MongoDB 获取用户"""
    coll = await get_users_collection()
    if not coll:
        return None
    try:
        doc = await coll.find_one({"_id": user_id})
        return dict(doc) if doc else None
    except Exception:
        return None


def get_mongo_uri_default() -> str:
    """本地开发默认 MongoDB 地址"""
    return "mongodb://localhost:27017"

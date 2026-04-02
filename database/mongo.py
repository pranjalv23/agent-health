import logging
import os
import uuid
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket

logger = logging.getLogger("agent_health.mongo")

_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_DB_NAME = os.getenv("MONGO_DB_NAME", "agent_health")


class MongoDB:
    _client: AsyncIOMotorClient | None = None
    _gridfs: AsyncIOMotorGridFSBucket | None = None

    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        if cls._client is None:
            logger.info("Connecting to MongoDB")
            cls._client = AsyncIOMotorClient(_MONGO_URI)
        return cls._client

    @classmethod
    def _db(cls):
        return cls.get_client()[_DB_NAME]

    @classmethod
    def _gridfs_bucket(cls) -> AsyncIOMotorGridFSBucket:
        if cls._gridfs is None:
            cls._gridfs = AsyncIOMotorGridFSBucket(cls._db())
        return cls._gridfs

    @classmethod
    def _conversations(cls):
        return cls._db()["conversations"]

    @classmethod
    def _profiles(cls):
        return cls._db()["health_profiles"]

    @classmethod
    def _files(cls):
        return cls._db()["files"]

    @classmethod
    def generate_session_id(cls) -> str:
        return uuid.uuid4().hex

    # ── Conversation persistence ──

    @classmethod
    async def save_conversation(
        cls,
        session_id: str,
        query: str,
        response: str,
        steps: list[dict] | None = None,
        user_id: str | None = None,
    ) -> str:
        doc = {
            "session_id": session_id,
            "query": query,
            "response": response,
            "steps": steps or [],
            "tools_used": list({s["tool"] for s in (steps or []) if s.get("action") == "tool_call"}),
            "total_tool_calls": sum(1 for s in (steps or []) if s.get("action") == "tool_call"),
            "created_at": datetime.now(timezone.utc),
        }
        if user_id:
            doc["user_id"] = user_id
        result = await cls._conversations().insert_one(doc)
        logger.info(
            "Saved conversation — session='%s', user='%s', doc_id='%s'",
            session_id, user_id or "anonymous", result.inserted_id,
        )
        return str(result.inserted_id)

    @classmethod
    async def get_history(cls, session_id: str) -> list[dict]:
        cursor = cls._conversations().find(
            {"session_id": session_id},
            {"_id": 0, "query": 1, "response": 1, "created_at": 1},
        ).sort("created_at", 1)
        return await cursor.to_list(length=100)

    @classmethod
    async def get_history_by_user(cls, user_id: str) -> list[dict]:
        cursor = cls._conversations().find(
            {"user_id": user_id},
            {"_id": 0, "query": 1, "response": 1, "created_at": 1, "session_id": 1},
        ).sort("created_at", -1)
        return await cursor.to_list(length=200)

    # ── Health profile persistence ──

    @classmethod
    async def save_profile(cls, user_id: str, profile: dict) -> None:
        """Upsert a user's health profile. One profile per user_id."""
        doc = {
            "user_id": user_id,
            "goals": profile.get("goals", ""),
            "fitness_level": profile.get("fitness_level", "beginner"),
            "available_equipment": profile.get("available_equipment", []),
            "dietary_restrictions": profile.get("dietary_restrictions", []),
            "injuries_or_limitations": profile.get("injuries_or_limitations", ""),
            "age": profile.get("age"),
            "weight_kg": profile.get("weight_kg"),
            "height_cm": profile.get("height_cm"),
            "sessions_per_week": profile.get("sessions_per_week"),
            "minutes_per_session": profile.get("minutes_per_session"),
            "updated_at": datetime.now(timezone.utc),
        }
        await cls._profiles().update_one(
            {"user_id": user_id},
            {"$set": doc},
            upsert=True,
        )
        logger.info("Saved health profile for user='%s'", user_id)

    @classmethod
    async def get_profile(cls, user_id: str) -> dict | None:
        """Retrieve a user's health profile by user_id."""
        return await cls._profiles().find_one(
            {"user_id": user_id},
            {"_id": 0},
        )

    # ── File storage (GridFS for plan exports) ──

    @classmethod
    async def store_file(
        cls,
        file_id: str,
        filename: str,
        data: bytes,
        file_type: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Store file content in MongoDB GridFS and save metadata."""
        bucket = cls._gridfs_bucket()
        await bucket.upload_from_stream(
            file_id,
            data,
            metadata={
                "file_id": file_id,
                "original_filename": filename,
                "file_type": file_type,
                "session_id": session_id,
                "user_id": user_id,
            },
        )
        doc = {
            "file_id": file_id,
            "filename": filename,
            "file_type": file_type,
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc),
        }
        await cls._files().insert_one(doc)
        logger.info("Stored file in GridFS — file_id='%s', type='%s', size=%d bytes",
                     file_id, file_type, len(data))

    @classmethod
    async def retrieve_file(cls, file_id: str) -> tuple[bytes, dict] | None:
        """Retrieve file content from GridFS. Returns (data, metadata) or None."""
        bucket = cls._gridfs_bucket()
        try:
            stream = await bucket.open_download_stream_by_name(file_id)
            data = await stream.read()
            meta = await cls._files().find_one({"file_id": file_id}, {"_id": 0})
            return data, meta or {}
        except Exception:
            logger.warning("File not found in GridFS: file_id='%s'", file_id)
            return None

    @classmethod
    async def get_latest_plan(cls, session_id: str) -> dict | None:
        """Retrieve the most recently generated fitness plan file for a session."""
        return await cls._files().find_one(
            {"session_id": session_id, "file_type": "fitness_plan"},
            {"_id": 0},
            sort=[("created_at", -1)],
        )

    @classmethod
    async def ensure_indexes(cls) -> None:
        db = cls._db()
        await db["conversations"].create_index("created_at", expireAfterSeconds=7_776_000)
        await db["health_profiles"].create_index("user_id", unique=True)
        await db["health_profiles"].create_index("updated_at", expireAfterSeconds=31_536_000)
        await db["files"].create_index("created_at", expireAfterSeconds=2_592_000)
        await db["fs.files"].create_index("uploadDate", expireAfterSeconds=2_592_000)
        logger.info("MongoDB TTL indexes ensured")

    @classmethod
    async def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None

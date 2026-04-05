import logging
import os
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from agent_sdk.database.mongo import BaseMongoDatabase

logger = logging.getLogger("agent_health.mongo")

_DB_NAME = os.getenv("MONGO_DB_NAME", "agent_health")


class MongoDB(BaseMongoDatabase):
    _gridfs: AsyncIOMotorGridFSBucket | None = None

    @classmethod
    def db_name(cls) -> str:
        return _DB_NAME

    @classmethod
    def _db(cls):
        return cls.get_client()[cls.db_name()]

    @classmethod
    def _gridfs_bucket(cls) -> AsyncIOMotorGridFSBucket:
        if cls._gridfs is None:
            cls._gridfs = AsyncIOMotorGridFSBucket(cls._db())
        return cls._gridfs

    @classmethod
    def _profiles(cls):
        return cls._db()["health_profiles"]

    @classmethod
    def _files(cls):
        return cls._db()["files"]

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
        await super().ensure_indexes()
        db = cls._db()
        await db["health_profiles"].create_index("user_id", unique=True)
        await db["health_profiles"].create_index("updated_at", expireAfterSeconds=31_536_000)
        await db["files"].create_index("created_at", expireAfterSeconds=2_592_000)
        await db["fs.files"].create_index("uploadDate", expireAfterSeconds=2_592_000)

"""
Twitch API routes for Open-LLM-VTuber
Provides REST API endpoints for Twitch integration management.
"""

from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..service_context import ServiceContext


class TwitchStatusResponse(BaseModel):
    """Response model for Twitch status."""

    enabled: bool
    connected: bool
    channel: str
    recent_messages_count: int
    error: str = None


class TwitchMessageResponse(BaseModel):
    """Response model for Twitch message."""

    user: str
    message: str
    timestamp: str
    is_subscriber: bool
    is_moderator: bool
    is_broadcaster: bool
    channel: str


def init_twitch_routes(default_context_cache: ServiceContext) -> APIRouter:
    """
    Initialize Twitch API routes.

    Args:
        default_context_cache: Service context for accessing Twitch client

    Returns:
        APIRouter: Router with Twitch API endpoints
    """
    router = APIRouter(prefix="/api/twitch", tags=["twitch"])

    @router.get("/status", response_model=TwitchStatusResponse)
    async def get_twitch_status():
        """Get current Twitch connection status."""
        try:
            if not default_context_cache.twitch_client:
                return TwitchStatusResponse(
                    enabled=False,
                    connected=False,
                    channel="",
                    recent_messages_count=0,
                    error="Twitch client not initialized",
                )

            status = default_context_cache.twitch_client.get_connection_status()
            return TwitchStatusResponse(**status)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/messages", response_model=List[TwitchMessageResponse])
    async def get_recent_messages():
        """Get recent Twitch messages."""
        try:
            if not default_context_cache.twitch_client:
                return []

            messages = default_context_cache.twitch_client.get_recent_messages()

            # Convert to response format
            response_messages = []
            for msg in messages:
                response_messages.append(
                    TwitchMessageResponse(
                        user=msg.user,
                        message=msg.message,
                        timestamp=msg.timestamp.isoformat(),
                        is_subscriber=msg.is_subscriber,
                        is_moderator=msg.is_moderator,
                        is_broadcaster=msg.is_broadcaster,
                        channel=msg.channel,
                    )
                )

            return response_messages

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/connect")
    async def connect_twitch():
        """Manually connect to Twitch chat."""
        try:
            if not default_context_cache.twitch_client:
                raise HTTPException(
                    status_code=400, detail="Twitch client not initialized"
                )

            if default_context_cache.twitch_client.is_connected:
                return {"message": "Already connected to Twitch"}

            success = await default_context_cache.twitch_client.connect()
            if success:
                return {"message": "Successfully connected to Twitch"}
            else:
                raise HTTPException(
                    status_code=500, detail="Failed to connect to Twitch"
                )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/disconnect")
    async def disconnect_twitch():
        """Manually disconnect from Twitch chat."""
        try:
            if not default_context_cache.twitch_client:
                return {"message": "Twitch client not initialized"}

            await default_context_cache.twitch_client.disconnect()
            return {"message": "Successfully disconnected from Twitch"}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/reconnect")
    async def reconnect_twitch():
        """Manually reconnect to Twitch chat."""
        try:
            if not default_context_cache.twitch_client:
                raise HTTPException(
                    status_code=400, detail="Twitch client not initialized"
                )

            success = await default_context_cache.twitch_client.reconnect()
            if success:
                return {"message": "Successfully reconnected to Twitch"}
            else:
                raise HTTPException(
                    status_code=500, detail="Failed to reconnect to Twitch"
                )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router

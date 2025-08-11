"""
Twitch integration module for Open-LLM-VTuber
Handles Twitch chat connection, message processing, and integration with the main system.
"""

import asyncio

# // DEBUG: [FIXED] Use loguru instead of stdlib logging | Ref: 3
from loguru import logger
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime


try:
    # v4 docs path
    from twitchAPI.type import AuthScope, ChatEvent
except Exception:  # pragma: no cover
    try:
        from twitchAPI.types import AuthScope  # older
    except Exception:
        try:
            from twitchAPI.oauth import AuthScope  # some versions expose it here
        except Exception:

            class AuthScope:  # minimal fallback
                CHAT_READ = "chat:read"
                CHAT_EDIT = "chat:edit"

    try:
        from twitchAPI.type import ChatEvent  # try again
    except Exception:
        ChatEvent = None  # type: ignore
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
from twitchAPI.chat import Chat, EventData, ChatMessage, ChatSub, ChatCommand

from ..i18n import t


@dataclass
class TwitchMessage:
    """Represents a processed Twitch chat message."""

    user: str
    message: str
    timestamp: datetime
    is_subscriber: bool = False
    is_moderator: bool = False
    is_broadcaster: bool = False
    emotes: Optional[Dict[str, Any]] = None
    bits: int = 0
    channel: str = ""


class TwitchClient:
    """
    Twitch client for handling chat connections and message processing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Twitch client with configuration.

        Args:
            config: Twitch configuration dictionary
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.channel_name = config.get("channel_name", "")
        self.app_id = config.get("app_id", "")
        self.app_secret = config.get("app_secret", "")
        self.max_message_length = config.get("max_message_length", 300)
        self.max_recent_messages = config.get("max_recent_messages", 10)

        # Twitch API objects
        self.twitch: Optional[Twitch] = None
        self.chat: Optional[Chat] = None

        # Message handling
        self.message_callback: Optional[Callable[[TwitchMessage], None]] = None
        self.recent_messages: list[TwitchMessage] = []

        # Connection state
        self.is_connected = False
        self.connection_task: Optional[asyncio.Task] = None

        # Status callback for UI/logic
        self.status_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        logger.bind(component="twitch_api").info(t("twitch.client_initialized"))

    async def initialize(self) -> bool:
        """
        Initialize Twitch API connection.

        Returns:
            bool: True if initialization successful
        """
        if not self.enabled:
            logger.bind(component="twitch_api").info(t("twitch.disabled"))
            return False

        if not self.channel_name or not self.app_id or not self.app_secret:
            logger.bind(component="twitch_api").error(t("twitch.missing_credentials"))
            return False

        try:
            # Initialize Twitch API
            self.twitch = await Twitch(self.app_id, self.app_secret)
            logger.bind(component="twitch_api").info("Twitch API client created")

            # User authentication with required scopes (CHAT_READ, CHAT_EDIT)
            auth_scope = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
            logger.bind(component="twitch_api").info(
                f"Requesting user OAuth with scopes: {auth_scope}"
            )
            authenticator = UserAuthenticator(self.twitch, auth_scope)
            token, refresh_token = await authenticator.authenticate()
            await self.twitch.set_user_authentication(token, auth_scope, refresh_token)
            logger.bind(component="twitch_api").info(
                "User OAuth applied to Twitch client"
            )

            # Initialize chat
            self.chat = await Chat(self.twitch)
            logger.bind(component="twitch_api").info("Chat client initialized")

            return True

        except Exception as e:
            logger.bind(component="twitch_api").error(
                t("twitch.init_error", error=str(e))
            )
            return False

    async def _on_ready(self, ready_event: EventData):
        """Called when chat signals READY (only with ChatEvent API)."""
        try:
            logger.bind(component="twitch_api").info(
                f"Twitch chat READY. Joining channel: {self.channel_name}"
            )
            await ready_event.chat.join_room(self.channel_name)
            logger.bind(component="twitch_api").info(
                f"Twitch chat JOINED channel: {self.channel_name}"
            )
            if self.status_callback:
                try:
                    self.status_callback(self.get_connection_status())
                except Exception:
                    pass
        except Exception as e:
            logger.bind(component="twitch_api").error(f"Error in on_ready handler: {e}")

    async def connect(self) -> bool:
        """
        Connect to Twitch chat.

        Returns:
            bool: True if connection successful
        """
        if not self.chat:
            logger.bind(component="twitch_api").error(t("twitch.not_initialized"))
            return False

        try:
            # Register event handlers (prefer ChatEvent API when available)
            if ChatEvent is not None and hasattr(self.chat, "register_event"):
                logger.bind(component="twitch_api").info(
                    "Registering ChatEvent handlers (READY, MESSAGE, SUB)"
                )
                try:
                    self.chat.register_event(ChatEvent.READY, self._on_ready)
                except Exception:
                    pass
                try:
                    self.chat.register_event(ChatEvent.MESSAGE, self._on_message)
                except Exception:
                    pass
                try:
                    self.chat.register_event(ChatEvent.SUB, self._on_subscription)
                except Exception:
                    pass
            else:
                logger.bind(component="twitch_api").info(
                    "Registering class-based chat callbacks (fallback)"
                )
                self.chat.register_callback(ChatMessage, self._on_message)
                self.chat.register_callback(ChatSub, self._on_subscription)
                self.chat.register_callback(ChatCommand, self._on_command)

            # Start chat loop (v4)
            logger.bind(component="twitch_api").info("Starting Twitch chat loop")
            self.chat.start()

            # Join channel immediately if no READY event available
            if not (ChatEvent and hasattr(self.chat, "register_event")):
                logger.bind(component="twitch_api").info(
                    f"Joining channel: {self.channel_name}"
                )
                await self.chat.join_room(self.channel_name)
                logger.bind(component="twitch_api").info(
                    f"Joined channel: {self.channel_name}"
                )

            self.is_connected = True
            logger.bind(component="twitch_api").info(
                t("twitch.connected", channel=self.channel_name)
            )

            # Notify status
            if self.status_callback:
                try:
                    self.status_callback(self.get_connection_status())
                except Exception:
                    logger.bind(component="twitch_api").debug(
                        "Status callback raised, ignored"
                    )

            # Start connection monitoring
            self.connection_task = asyncio.create_task(self._monitor_connection())

            return True

        except Exception as e:
            logger.bind(component="twitch_api").error(
                t("twitch.connection_error", error=str(e))
            )
            return False

    async def disconnect(self):
        """Disconnect from Twitch chat."""
        if self.connection_task:
            self.connection_task.cancel()

        if self.chat:
            try:
                logger.bind(component="twitch_api").info("Stopping Twitch chat loop")
                self.chat.stop()
            except Exception:
                pass

        if self.twitch:
            await self.twitch.close()

        self.is_connected = False
        logger.bind(component="twitch_api").info(t("twitch.disconnected"))

        # Notify status
        if self.status_callback:
            try:
                self.status_callback(self.get_connection_status())
            except Exception:
                logger.bind(component="twitch_api").debug(
                    "Status callback raised, ignored"
                )

    def set_message_callback(self, callback: Callable[[TwitchMessage], None]):
        """
        Set callback function for processing messages.

        Args:
            callback: Function to handle incoming messages
        """
        self.message_callback = callback
        logger.bind(component="twitch_api").info(t("twitch.callback_set"))

    async def _on_message(self, msg: ChatMessage):
        """
        Handle incoming chat message.

        Args:
            msg: ChatMessage from Twitch API
        """
        try:
            logger.bind(component="twitch_api").info(
                f"[Twitch] MESSAGE in {getattr(msg, 'room', getattr(msg, 'channel', ''))}: {msg.user.name}: {msg.text}"
            )
            # Create TwitchMessage object
            user_obj = getattr(msg, "user", None)
            user_name = getattr(user_obj, "name", getattr(msg, "user_name", "unknown"))
            is_subscriber = bool(getattr(user_obj, "is_subscriber", False))
            is_moderator = bool(getattr(user_obj, "is_moderator", False))
            is_broadcaster = bool(getattr(user_obj, "is_broadcaster", False))
            emotes = getattr(msg, "emotes", None)
            bits_val = getattr(msg, "bits", 0) or 0

            twitch_msg = TwitchMessage(
                user=user_name,
                message=getattr(msg, "text", ""),
                timestamp=datetime.utcnow(),
                is_subscriber=is_subscriber,
                is_moderator=is_moderator,
                is_broadcaster=is_broadcaster,
                emotes=emotes,
                bits=int(bits_val),
                channel=getattr(msg, "room", getattr(msg, "channel", "")),
            )

            # Save recent message and invoke callback
            try:
                self.recent_messages.append(twitch_msg)
                if len(self.recent_messages) > self.max_recent_messages:
                    self.recent_messages.pop(0)
            except Exception:
                pass

            if self.message_callback:
                self.message_callback(twitch_msg)
        except Exception as e:
            logger.bind(component="twitch_api").error(
                t("twitch.message_error", error=str(e))
            )

    async def _on_subscription(self, sub: ChatSub):
        """Handle subscription event."""
        try:
            logger.bind(component="twitch_api").info(
                f"[Twitch] SUB in {getattr(sub, 'room', getattr(sub, 'channel', ''))}: user={sub.user.name}, months={getattr(sub, 'months', '?')}"
            )
            logger.bind(component="twitch_api").info(
                t("twitch.subscription", user=sub.user.name, months=sub.months)
            )
        except Exception as e:
            logger.bind(component="twitch_api").error(
                t("twitch.subscription_error", error=str(e))
            )

    async def _on_command(self, cmd: ChatCommand):
        """Handle command event."""
        try:
            logger.bind(component="twitch_api").info(
                f"[Twitch] COMMAND in {getattr(cmd, 'room', getattr(cmd, 'channel', ''))}: !{cmd.command} {cmd.parameter}"
            )
            logger.bind(component="twitch_api").info(
                t("twitch.command", user=cmd.user.name, command=cmd.command)
            )
        except Exception as e:
            logger.bind(component="twitch_api").error(
                t("twitch.command_error", error=str(e))
            )

    # // DEBUG: [FIXED] Restore monitor/reconnect logic missing after refactor | Ref: 3
    async def _monitor_connection(self):
        """Monitor chat connection and attempt to reconnect on loss."""
        while self.is_connected:
            try:
                await asyncio.sleep(30)
                if not self.chat:
                    continue
                is_ok = True
                try:
                    # twitchAPI Chat may expose is_connected or similar; guard if absent
                    is_ok = bool(getattr(self.chat, "is_connected", lambda: True)())
                except Exception:
                    is_ok = True
                if not is_ok:
                    logger.bind(component="twitch_api").warning(
                        t("twitch.connection_lost")
                    )
                    if self.status_callback:
                        try:
                            self.status_callback(self.get_connection_status())
                        except Exception:
                            logger.bind(component="twitch_api").debug(
                                "Status callback raised, ignored"
                            )
                    await self.reconnect()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.bind(component="twitch_api").error(
                    t("twitch.monitor_error", error=str(e))
                )

    async def reconnect(self) -> bool:
        """Attempt to reconnect to Twitch chat."""
        try:
            logger.bind(component="twitch_api").info(t("twitch.reconnecting"))
            await self.disconnect()
            await asyncio.sleep(5)
            if await self.initialize() and await self.connect():
                logger.bind(component="twitch_api").info(t("twitch.reconnected"))
                # Notify status
                if self.status_callback:
                    try:
                        self.status_callback(self.get_connection_status())
                    except Exception:
                        logger.bind(component="twitch_api").debug(
                            "Status callback raised, ignored"
                        )
                return True
            else:
                logger.bind(component="twitch_api").error(t("twitch.reconnect_failed"))
                return False
        except Exception as e:
            logger.bind(component="twitch_api").error(
                t("twitch.reconnect_error", error=str(e))
            )
            return False

    def get_recent_messages(self) -> list[TwitchMessage]:
        """
        Get list of recent messages.

        Returns:
            List of recent TwitchMessage objects
        """
        return self.recent_messages.copy()

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status.

        Returns:
            Dictionary with connection information
        """
        return {
            "enabled": self.enabled,
            "connected": self.is_connected,
            "channel": self.channel_name,
            "recent_messages_count": len(self.recent_messages),
        }

    def set_status_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback to notify status changes."""
        self.status_callback = callback
        logger.debug("Twitch status callback set")

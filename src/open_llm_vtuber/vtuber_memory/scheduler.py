from __future__ import annotations

import asyncio
from typing import Optional, Callable

from loguru import logger


class ConsolidationScheduler:
    """Async scheduler that periodically triggers memory consolidation.

    Usage:
        scheduler = ConsolidationScheduler(interval_sec=900, trigger=ctx.trigger_memory_consolidation)
        await scheduler.start()
        ...
        await scheduler.stop()

    Optionally a secondary callback can run after consolidation (e.g., TTL prune).
    """

    def __init__(
        self,
        interval_sec: int,
        trigger: Callable[[str], "asyncio.Future"],
        *,
        post_hook: Optional[Callable[[], "asyncio.Future"]] = None,
    ):
        self.interval_sec = max(60, int(interval_sec))
        self._trigger = trigger
        self._post_hook = post_hook
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    async def _loop(self) -> None:
        logger.info(f"ConsolidationScheduler started: every {self.interval_sec}s")
        try:
            while not self._stopping:
                await asyncio.sleep(self.interval_sec)
                if self._stopping:
                    break
                try:
                    await self._trigger("scheduled")
                except Exception as e:
                    logger.debug(f"scheduled consolidation skipped: {e}")
                # Run optional post maintenance task
                if self._post_hook and not self._stopping:
                    try:
                        await self._post_hook()
                    except Exception as e:
                        logger.debug(f"post_hook skipped: {e}")
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("ConsolidationScheduler stopped")

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

from pydantic import Field
from typing import Dict, ClassVar, List
from .i18n import I18nMixin, Description


class BiliBiliLiveConfig(I18nMixin):
    """Configuration for BiliBili Live platform."""

    room_ids: List[int] = Field([], alias="room_ids")
    sessdata: str = Field("", alias="sessdata")

    DESCRIPTIONS = {
        "room_ids": Description(i18n_key="list_of_bilibili_live_room_ids_to_monitor"),
        "sessdata": Description(i18n_key="sessdata_cookie_value_for_authenticated_requests"),
    }


class LiveConfig(I18nMixin):
    """Configuration for live streaming platforms integration."""

    bilibili_live: BiliBiliLiveConfig = Field(
        BiliBiliLiveConfig(), alias="bilibili_live"
    )

    DESCRIPTIONS = {
        "bilibili_live": Description(i18n_key="configuration_for_bilibili_live_streaming"),
    }

import os
import sys
import atexit
import asyncio
import argparse
import subprocess
from pathlib import Path
import tomli
import uvicorn
from loguru import logger
from upgrade_codes.upgrade_manager import UpgradeManager

from src.open_llm_vtuber.server import WebSocketServer
from src.open_llm_vtuber.config_manager import Config, read_yaml, validate_config
# Import simple i18n system
from src.open_llm_vtuber.i18n import set_language, t

os.environ["HF_HOME"] = str(Path(__file__).parent / "models")
os.environ["MODELSCOPE_CACHE"] = str(Path(__file__).parent / "models")

upgrade_manager = UpgradeManager()


def get_version() -> str:
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["project"]["version"]


def init_logger(console_log_level: str = "INFO") -> None:
    logger.remove()
    # Console output
    logger.add(
        sys.stderr,
        level=console_log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
        colorize=True,
    )

    # File output
    logger.add(
        "logs/debug_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
        backtrace=True,
        diagnose=True,
    )


def check_frontend_submodule(lang=None):
    """
    Check if the frontend submodule is initialized. If not, attempt to initialize it.
    If initialization fails, log an error message.
    """
    if lang is None:
        lang = upgrade_manager.lang

    try:
        # Check if the frontend submodule is initialized
        frontend_path = Path(__file__).parent / "frontend" / "index.html"
        if not frontend_path.exists():
            if lang == "zh":
                logger.warning("前端子模块未找到，正在尝试初始化子模块...")
            elif lang == "ru":
                logger.warning("Фронтенд подмодуль не найден, пытаемся инициализировать подмодули...")
            else:
                logger.warning("Frontend submodule not found, attempting to initialize submodules...")

            # Try to initialize submodules
            subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True)

            if frontend_path.exists():
                if lang == "zh":
                    logger.info("👍 前端子模块（和其他子模块）初始化成功。")
                elif lang == "ru":
                    logger.info("👍 Фронтенд подмодуль (и другие подмодули) успешно инициализированы.")
                else:
                    logger.info("👍 Frontend submodule (and other submodules) initialized successfully.")
            else:
                if lang == "zh":
                    logger.critical(
                        '子模块初始化失败。\n你之后可能会在浏览器中看到 {{"detail":"Not Found"}} 的错误提示。请检查我们的快速入门指南和常见问题页面以获取更多信息。'
                    )
                    logger.error(
                        "初始化子模块后，前端文件仍然缺失。\n"
                        + "你是否手动更改或删除了 `frontend` 文件夹？\n"
                        + "它是一个 Git 子模块 - 你不应该直接修改它。\n"
                        + "如果你这样做了，请使用 `git restore frontend` 丢弃你的更改，然后再试一次。\n"
                    )
                elif lang == "ru":
                    logger.critical(
                        'Не удалось инициализировать подмодули.\nВы можете увидеть {{"detail":"Not Found"}} в браузере. Пожалуйста, проверьте наше руководство по быстрому старту и страницу общих проблем для получения дополнительной информации.'
                    )
                    logger.error(
                        "Файлы фронтенда все еще отсутствуют после инициализации подмодулей.\n"
                        + "Вы вручную изменили или удалили папку `frontend`?\n"
                        + "Это Git подмодуль — вы не должны изменять его напрямую.\n"
                        + "Если вы это сделали, отмените изменения с помощью `git restore frontend`, затем попробуйте снова.\n"
                    )
                else:
                    logger.critical(
                        'Failed to initialize submodules. \nYou might see {{"detail":"Not Found"}} in your browser. Please check our quick start guide and common issues page from our documentation.'
                    )
                    logger.error(
                        "Frontend files are still missing after submodule initialization.\n"
                        + "Did you manually change or delete the `frontend` folder?  \n"
                        + "It's a Git submodule — you shouldn't modify it directly.  \n"
                        + "If you did, discard your changes with `git restore frontend`, then try again.\n"
                    )
    except Exception as e:
        if lang == "zh":
            logger.critical(
                f'初始化子模块失败: {e}。\n怀疑你跟 GitHub 之间有网络问题。你之后可能会在浏览器中看到 {{"detail":"Not Found"}} 的错误提示。请检查我们的快速入门指南和常见问题页面以获取更多信息。\n'
            )
        elif lang == "ru":
            logger.critical(
                f'Не удалось инициализировать подмодули: {e}.\nПодозреваем проблемы с сетью между вами и GitHub. Вы можете увидеть {{"detail":"Not Found"}} в браузере. Пожалуйста, проверьте наше руководство по быстрому старту и страницу общих проблем для получения дополнительной информации.\n'
            )
        else:
            logger.critical(
                f'Failed to initialize submodules: {e}. \nYou might see {{"detail":"Not Found"}} in your browser. Please check our quick start guide and common issues page from our documentation.\n'
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Open-LLM-VTuber Server")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--hf_mirror", action="store_true", help="Use Hugging Face mirror"
    )
    return parser.parse_args()


@logger.catch
def run(console_log_level: str):
    init_logger(console_log_level)
    logger.info(f"Open-LLM-VTuber, version v{get_version()}")

    # Check if the frontend submodule is initialized
    lang = upgrade_manager.lang
    check_frontend_submodule(lang)

    # Sync user config with default config
    try:
        upgrade_manager.sync_user_config()
    except Exception as e:
        logger.error(f"Error syncing user config: {e}")

    atexit.register(WebSocketServer.clean_cache)

    # Load configurations from yaml file
    config: Config = validate_config(read_yaml("conf.yaml"))
    server_config = config.system_config
    
    # Get language from config
    lang = getattr(server_config, 'language', 'en')
    
    # Initialize i18n system with the selected language
    if set_language(lang):
        logger.info(f"🌍 Language set to: {lang}")
    else:
        logger.warning(f"⚠️ Language '{lang}' not available, using English")
        set_language("en")
    
    # Test i18n system
    logger.info(t("server.starting"))

    if server_config.enable_proxy:
        logger.info(t("server.proxy_enabled"))

    # Initialize the WebSocket server (synchronous part)
    server = WebSocketServer(config=config)

    # Perform asynchronous initialization (loading context, etc.)
    logger.info(t("server.initializing_context"))
    try:
        asyncio.run(server.initialize())
        logger.info(t("server.context_initialized"))
    except Exception as e:
        logger.error(t("server.context_init_failed", error=str(e)))
        sys.exit(1)  # Exit if initialization fails

    # Run the Uvicorn server
    logger.info(t("server.starting_on", host=server_config.host, port=server_config.port))
    uvicorn.run(
        app=server.app,
        host=server_config.host,
        port=server_config.port,
        log_level=console_log_level.lower(),
    )


if __name__ == "__main__":
    args = parse_args()
    console_log_level = "DEBUG" if args.verbose else "INFO"
    if args.verbose:
        logger.info("Running in verbose mode")
    else:
        logger.info(
            "Running in standard mode. For detailed debug logs, use: uv run run_server.py --verbose"
        )
    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    run(console_log_level=console_log_level)

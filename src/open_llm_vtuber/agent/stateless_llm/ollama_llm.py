import atexit
import requests
from loguru import logger
from .openai_compatible_llm import AsyncLLM

try:
    from openai_harmony import (
        load_harmony_encoding,
        HarmonyEncodingName,
        Role,
        Message,
        Conversation,
        SystemContent,
        TextContent,
    )
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    logger.warning("openai-harmony not available. Harmony mode will be disabled.")


class OllamaLLM(AsyncLLM):
    def __init__(
        self,
        model: str,
        base_url: str,
        llm_api_key: str = "z",
        organization_id: str = "z",
        project_id: str = "z",
        temperature: float = 1.0,
        keep_alive: float = -1,
        unload_at_exit: bool = True,
        use_harmony: bool = False,
    ):
        self.keep_alive = keep_alive
        self.unload_at_exit = unload_at_exit
        self.use_harmony = use_harmony and HARMONY_AVAILABLE
        self.cleaned = False
        
        # Инициализируем Harmony если доступен и включен
        if self.use_harmony:
            try:
                self.harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
                logger.info("Harmony encoding initialized for Ollama")
            except Exception as e:
                logger.error(f"Failed to initialize Harmony encoding: {e}")
                self.use_harmony = False
        
        super().__init__(
            model=model,
            base_url=base_url,
            llm_api_key=llm_api_key,
            organization_id=organization_id,
            project_id=project_id,
            temperature=temperature,
        )
        
        try:
            # preload model
            logger.info("Preloading model for Ollama")
            # Send the POST request to preload model
            logger.debug(
                requests.post(
                    base_url.replace("/v1", "") + "/api/chat",
                    json={
                        "model": model,
                        "keep_alive": keep_alive,
                    },
                )
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to preload model: {e}")
            logger.critical(
                "Fail to connect to Ollama backend. Is Ollama server running? Try running `ollama list` to start the server and try again.\nThe AI will repeat 'Error connecting chat endpoint' until the server is running."
            )
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")
        # If keep_alive is less than 0, register cleanup to unload the model
        if unload_at_exit:
            atexit.register(self.cleanup)

    def __del__(self):
        """Destructor to unload the model"""
        self.cleanup()

    def cleanup(self):
        """Clean up function to unload the model when exitting"""
        if not self.cleaned and self.unload_at_exit:
            logger.info(f"Ollama: Unloading model: {self.model}")
            # Unload the model
            # unloading is just the same as preload, but with keep alive set to 0
            logger.debug(
                requests.post(
                    self.base_url.replace("/v1", "") + "/api/chat",
                    json={
                        "model": self.model,
                        "keep_alive": 0,
                    },
                )
            )
            self.cleaned = True

    def create_harmony_conversation(self, messages):
        """Создает разговор в формате Harmony"""
        if not self.use_harmony:
            return None
            
        harmony_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content_data = msg.get("content", "")
            
            # Обрабатываем content, который может быть строкой или списком словарей
            if isinstance(content_data, list):
                # Извлекаем текст из списка словарей
                extracted_text = ""
                for item in content_data:
                    if isinstance(item, dict) and item.get("type") == "text":
                        extracted_text += item.get("text", "")
                content_to_use = extracted_text
            else:
                content_to_use = content_data
            
            if role == "system":
                content = SystemContent.new()
                content.model_identity = content_to_use
                harmony_msg = Message.from_role_and_content(Role.SYSTEM, content)
            elif role == "user":
                content = TextContent(text=content_to_use)
                harmony_msg = Message.from_role_and_content(Role.USER, content)
            elif role == "assistant":
                content = TextContent(text=content_to_use)
                harmony_msg = Message.from_role_and_content(Role.ASSISTANT, content)
            else:
                # Пропускаем неизвестные роли
                continue
            
            harmony_messages.append(harmony_msg)
        
        return Conversation.from_messages(harmony_messages)

    async def chat_completion(
        self,
        messages,
        system: str = None,
        tools=None,
    ):
        """Переопределяем chat_completion для поддержки Harmony"""
        
        if self.use_harmony:
            # Создаем разговор в формате Harmony
            conversation = self.create_harmony_conversation(messages)
            if conversation:
                # Рендерим в токены
                tokens = self.harmony_enc.render_conversation_for_completion(conversation, Role.ASSISTANT)
                
                # Декодируем для отладки
                decoded = self.harmony_enc.decode_utf8(tokens)
                logger.debug(f"Harmony tokens: {len(tokens)}, decoded: {decoded[:200]}...")
                
                # Здесь можно добавить логику для отправки токенов в Ollama
                # Пока что используем обычный режим как fallback
                logger.info("Harmony mode enabled but not fully implemented yet. Using standard mode.")
        
        # Используем стандартный метод - возвращаем async_generator напрямую
        return super().chat_completion(messages, system, tools)

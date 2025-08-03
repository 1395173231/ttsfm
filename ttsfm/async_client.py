"""
Asynchronous TTS client implementation.

This module provides the AsyncTTSClient class for asynchronous
text-to-speech generation with OpenAI-compatible API.
"""

import json
import uuid
import asyncio
import logging
from typing import Optional, Dict, Any, Union, List
from curl_cffi import requests  # 替换 aiohttp 导入
from curl_cffi.requests import AsyncSession  # 用于异步会话

from .models import (
    TTSRequest, TTSResponse, Voice, AudioFormat,
    get_content_type, get_format_from_content_type
)
from .exceptions import (
    TTSException, APIException, NetworkException, ValidationException,
    create_exception_from_response
)
from .utils import (
    get_realistic_headers, sanitize_text, validate_url, build_url,
    exponential_backoff, estimate_audio_duration, format_file_size,
    validate_text_length, split_text_by_length
)


logger = logging.getLogger(__name__)


class AsyncTTSClient:
    """
    Asynchronous TTS client for text-to-speech generation.
    
    This client provides an async interface for generating speech from text
    using OpenAI-compatible TTS services with support for concurrent requests.
    
    Attributes:
        base_url: Base URL for the TTS service
        api_key: API key for authentication (if required)
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        verify_ssl: Whether to verify SSL certificates
        max_concurrent: Maximum concurrent requests
    """
    
    def __init__(
        self,
        base_url: str = "https://www.openai.fm",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        verify_ssl: bool = True,
        max_concurrent: int = 10,
        **kwargs
    ):
        """
        Initialize the async TTS client.
        
        Args:
            base_url: Base URL for the TTS service
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            verify_ssl: Whether to verify SSL certificates
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional configuration options
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        self.max_concurrent = max_concurrent
        
        # 使用 curl-cffi 的 AsyncSession
        self._session: Optional[AsyncSession] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        if not validate_url(self.base_url):
            raise ValidationException(f"Invalid base URL: {self.base_url}")
        
        logger.info(f"Initialized async TTS client with base URL: {self.base_url}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """确保 HTTP 会话已创建"""
        if self._session is None:
            # 设置请求头
            headers = get_realistic_headers()
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # 创建 curl-cffi 会话
            self._session = AsyncSession(
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
                impersonate="chrome"  # 模拟 Chrome 浏览器
            )
    
    async def generate_speech(
        self,
        text: str,
        voice: Union[Voice, str] = Voice.ALLOY,
        response_format: Union[AudioFormat, str] = AudioFormat.MP3,
        instructions: Optional[str] = None,
        max_length: int = 4096,
        validate_length: bool = True,
        **kwargs
    ) -> TTSResponse:
        """
        Generate speech from text asynchronously.

        Args:
            text: Text to convert to speech
            voice: Voice to use for generation
            response_format: Audio format for output
            instructions: Optional instructions for voice modulation
            max_length: Maximum allowed text length in characters (default: 4096)
            validate_length: Whether to validate text length (default: True)
            **kwargs: Additional parameters

        Returns:
            TTSResponse: Generated audio response

        Raises:
            TTSException: If generation fails
            ValueError: If text exceeds max_length and validate_length is True
        """
        # Create and validate request
        request = TTSRequest(
            input=sanitize_text(text),
            voice=voice,
            response_format=response_format,
            instructions=instructions,
            max_length=max_length,
            validate_length=validate_length,
            **kwargs
        )

        return await self._make_request(request)

    async def generate_speech_long_text(
        self,
        text: str,
        voice: Union[Voice, str] = Voice.ALLOY,
        response_format: Union[AudioFormat, str] = AudioFormat.MP3,
        instructions: Optional[str] = None,
        max_length: int = 4096,
        preserve_words: bool = True,
        **kwargs
    ) -> List[TTSResponse]:
        """
        Generate speech from long text by splitting it into chunks asynchronously.

        This method automatically splits text that exceeds max_length into
        smaller chunks and generates speech for each chunk concurrently.

        Args:
            text: Text to convert to speech
            voice: Voice to use for generation
            response_format: Audio format for output
            instructions: Optional instructions for voice modulation
            max_length: Maximum length per chunk (default: 4096)
            preserve_words: Whether to avoid splitting words (default: True)
            **kwargs: Additional parameters

        Returns:
            List[TTSResponse]: List of generated audio responses

        Raises:
            TTSException: If generation fails for any chunk
        """
        # Sanitize text first
        clean_text = sanitize_text(text)

        # Split text into chunks
        chunks = split_text_by_length(clean_text, max_length, preserve_words)

        if not chunks:
            raise ValueError("No valid text chunks found after processing")

        # Create requests for all chunks
        requests = []
        for chunk in chunks:
            request = TTSRequest(
                input=chunk,
                voice=voice,
                response_format=response_format,
                instructions=instructions,
                max_length=max_length,
                validate_length=False,  # We already split the text
                **kwargs
            )
            requests.append(request)

        # Process all chunks concurrently
        return await self.generate_speech_batch(requests)

    async def generate_speech_from_long_text(
        self,
        text: str,
        voice: Union[Voice, str] = Voice.ALLOY,
        response_format: Union[AudioFormat, str] = AudioFormat.MP3,
        instructions: Optional[str] = None,
        max_length: int = 4096,
        preserve_words: bool = True,
        **kwargs
    ) -> List[TTSResponse]:
        """
        Generate speech from long text by splitting it into chunks asynchronously.

        This is an alias for generate_speech_long_text for consistency.

        Args:
            text: Text to convert to speech
            voice: Voice to use for generation
            response_format: Audio format for output
            instructions: Optional instructions for voice modulation
            max_length: Maximum length per chunk (default: 4096)
            preserve_words: Whether to avoid splitting words (default: True)
            **kwargs: Additional parameters

        Returns:
            List[TTSResponse]: List of generated audio responses

        Raises:
            TTSException: If generation fails for any chunk
        """
        return await self.generate_speech_long_text(
            text=text,
            voice=voice,
            response_format=response_format,
            instructions=instructions,
            max_length=max_length,
            preserve_words=preserve_words,
            **kwargs
        )

    async def generate_speech_batch(
        self,
        requests: List[TTSRequest]
    ) -> List[TTSResponse]:
        """
        Generate speech for multiple requests concurrently.
        
        Args:
            requests: List of TTS requests
            
        Returns:
            List[TTSResponse]: List of generated audio responses
            
        Raises:
            TTSException: If any generation fails
        """
        if not requests:
            return []
        
        # Process requests concurrently with semaphore limiting
        tasks = [self._make_request(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for exceptions and convert them
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                raise TTSException(f"Request {i} failed: {str(response)}")
            results.append(response)
        
        return results
    
    async def generate_speech_from_request(self, request: TTSRequest) -> TTSResponse:
        """
        Generate speech from a TTSRequest object asynchronously.
        
        Args:
            request: TTS request object
            
        Returns:
            TTSResponse: Generated audio response
        """
        return await self._make_request(request)
    
    async def _make_request(self, request: TTSRequest) -> TTSResponse:
        """
        Make the actual HTTP request to the TTS service.
        
        Args:
            request: TTS request object
            
        Returns:
            TTSResponse: Generated audio response
            
        Raises:
            TTSException: If request fails
        """
        await self._ensure_session()
        
        async with self._semaphore:
            url = build_url(self.base_url, "api/generate")
            
            # 准备表单数据
            form_data = {
                'input': request.input,
                'voice': request.voice.value,
                'generation': str(uuid.uuid4()),
                'response_format': request.response_format.value if hasattr(request.response_format, 'value') else str(request.response_format)
            }

            if request.instructions:
                form_data['prompt'] = request.instructions
            else:
                form_data['prompt'] = (
                    "Affect/personality: Natural and clear\n\n"
                    "Tone: Friendly and professional, creating a pleasant listening experience.\n\n"
                    "Pronunciation: Clear, articulate, and steady, ensuring each word is easily understood "
                    "while maintaining a natural, conversational flow.\n\n"
                    "Pause: Brief, purposeful pauses between sentences to allow time for the listener "
                    "to process the information.\n\n"
                    "Emotion: Warm and engaging, conveying the intended message effectively."
                )

            logger.info(f"Generating speech for text: '{request.input[:50]}...' with voice: {request.voice}")

            # 使用 curl-cffi 发送请求
            for attempt in range(self.max_retries + 1):
                try:
                    if attempt > 0:
                        delay = exponential_backoff(attempt - 1)
                        logger.info(f"Retrying request after {delay:.2f}s (attempt {attempt + 1})")
                        await asyncio.sleep(delay)

                    response = await self._session.post(
                        url,
                        data=form_data,
                        timeout=self.timeout
                    )

                    if response.status_code == 200:
                        return await self._process_openai_fm_response(response, request)
                    else:
                        try:
                            error_data = response.json()
                        except (json.JSONDecodeError, ValueError):
                            error_data = {"error": {"message": response.text or "Unknown error"}}
                        
                        exception = create_exception_from_response(
                            response.status_code,
                            error_data,
                            f"TTS request failed with status {response.status_code}"
                        )
                        
                        if response.status_code in [400, 401, 403, 404]:
                            raise exception
                        
                        if attempt == self.max_retries:
                            raise exception
                        
                        logger.warning(f"Request failed with status {response.status_code}, retrying...")
                        continue
                        
                except (requests.RequestsError, asyncio.TimeoutError) as e:
                    if attempt == self.max_retries:
                        raise NetworkException(
                            f"Request error: {str(e)}",
                            retry_count=attempt
                        )
                    logger.warning(f"Request error, retrying...")
                    continue
            
            raise TTSException("Maximum retries exceeded")
    
    async def _process_openai_fm_response(
        self,
        response: requests.Response,
        request: TTSRequest
    ) -> TTSResponse:
        """
        Process a successful response from the openai.fm TTS service.

        Args:
            response: HTTP response object
            request: Original TTS request

        Returns:
            TTSResponse: Processed response object
        """
        # Get content type from response headers
        content_type = response.headers.get("content-type", "audio/mpeg")

        # Get audio data
        audio_data = response.content  # curl-cffi 直接提供字节内容

        if not audio_data:
            raise APIException("Received empty audio data from openai.fm")

        # Determine format from content type
        if "audio/mpeg" in content_type or "audio/mp3" in content_type:
            actual_format = AudioFormat.MP3
        elif "audio/wav" in content_type:
            actual_format = AudioFormat.WAV
        elif "audio/opus" in content_type:
            actual_format = AudioFormat.OPUS
        elif "audio/aac" in content_type:
            actual_format = AudioFormat.AAC
        elif "audio/flac" in content_type:
            actual_format = AudioFormat.FLAC
        else:
            # Default to MP3 for openai.fm
            actual_format = AudioFormat.MP3

        # Estimate duration based on text length
        estimated_duration = estimate_audio_duration(request.input)

        # Check if returned format differs from requested format
        requested_format = request.response_format
        if isinstance(requested_format, str):
            try:
                requested_format = AudioFormat(requested_format.lower())
            except ValueError:
                requested_format = AudioFormat.MP3  # Default fallback

        # Import here to avoid circular imports
        from .models import maps_to_wav

        # Check if format differs from request
        if actual_format != requested_format:
            if maps_to_wav(requested_format.value) and actual_format.value == "wav":
                logger.debug(
                    f"Format '{requested_format.value}' requested, returning WAV format."
                )
            else:
                logger.warning(
                    f"Requested format '{requested_format.value}' but received '{actual_format.value}' "
                    f"from service."
                )

        # Create response object
        tts_response = TTSResponse(
            audio_data=audio_data,
            content_type=content_type,
            format=actual_format,
            size=len(audio_data),
            duration=estimated_duration,
            metadata={
                "response_headers": dict(response.headers),
                "status_code": response.status_code,
                "url": str(response.url),
                "service": "openai.fm",
                "voice": request.voice.value,
                "original_text": request.input[:100] + "..." if len(request.input) > 100 else request.input,
                "requested_format": requested_format.value,
                "actual_format": actual_format.value
            }
        )

        logger.info(
            f"Successfully generated {format_file_size(len(audio_data))} "
            f"of {actual_format.value.upper()} audio from openai.fm using voice '{request.voice.value}'"
        )

        return tts_response
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

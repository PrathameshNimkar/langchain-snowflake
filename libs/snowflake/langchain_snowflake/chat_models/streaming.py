"""Streaming functionality for Snowflake chat models."""

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessageChunk, BaseMessage, ToolCallChunk
from langchain_core.outputs import ChatGenerationChunk

from .._connection.rest_client import RestApiClient, RestApiRequestBuilder
from .._error_handling import SnowflakeErrorHandler

logger = logging.getLogger(__name__)


class SnowflakeStreaming:
    """Mixin class for Snowflake streaming functionality."""

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions from Snowflake Cortex.

        Args:
            messages: List of messages to send to the model
            stop: List of stop sequences (not supported by Cortex)
            run_manager: Callback manager for the run
            **kwargs: Additional keyword arguments

        Yields:
            ChatGenerationChunk: Streaming chunks of the response

        Note: Uses Cortex COMPLETE's native streaming via REST API when tools
        are bound, otherwise falls back to simulated streaming for SQL function.
        """
        try:
            if self._should_use_rest_api():
                # Use native streaming via REST API
                for chunk in self._stream_via_rest_api(messages, run_manager, **kwargs):
                    yield chunk
            else:
                # Simulate streaming by chunking the complete response
                result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

                if result.generations:
                    content = result.generations[0].message.content

                    # Split content into chunks for streaming effect
                    chunk_size = max(1, len(content) // 20)  # Aim for ~20 chunks

                    for i in range(0, len(content), chunk_size):
                        chunk_content = content[i : i + chunk_size]

                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(
                                content=chunk_content,
                                usage_metadata=(result.generations[0].message.usage_metadata if i == 0 else None),
                                response_metadata=(result.generations[0].message.response_metadata if i == 0 else {}),
                            )
                        )

                        yield chunk

                        if run_manager:
                            run_manager.on_llm_new_token(chunk_content)

        except Exception as e:
            # Use centralized error handling to create consistent error response
            error_result = SnowflakeErrorHandler.create_chat_error_result(
                error=e,
                operation="stream chat completions",
                model=self.model,
                input_tokens=self._estimate_tokens(messages),
            )
            # Convert ChatResult to streaming chunk format
            error_content = error_result.generations[0].message.content
            yield ChatGenerationChunk(message=AIMessageChunk(content=error_content))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream response by delegating to sync _stream method.

        This eliminates code duplication by using the sync implementation
        with asyncio.to_thread() for non-blocking execution.
        """
        # Determine streaming method based on tool requirements
        try:
            if self._should_use_rest_api():
                # Use native async REST API streaming with aiohttp
                async for chunk in self._astream_via_rest_api(messages, run_manager, **kwargs):
                    yield chunk
            else:
                # For SQL-based streaming, we currently delegate to sync method since
                # Snowflake doesn't support native SQL streaming, only batch results
                def sync_stream():
                    return list(self._stream(messages, stop, run_manager, **kwargs))

                chunks = await asyncio.to_thread(sync_stream)
                for chunk in chunks:
                    yield chunk

        except Exception as e:
            # Use centralized error handling for consistent async streaming errors
            error_result = SnowflakeErrorHandler.create_chat_error_result(
                error=e,
                operation="async stream chat completions",
                model=self.model,
                input_tokens=self._estimate_tokens(messages),
            )
            # Convert ChatResult to streaming chunk format
            error_content = error_result.generations[0].message.content
            error_chunk = ChatGenerationChunk(message=AIMessageChunk(content=error_content))
            yield error_chunk

    def _stream_via_rest_api(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions via REST API with native streaming support."""
        try:
            # Build REST API payload with streaming enabled
            payload = self._build_rest_api_payload(messages)
            payload["stream"] = True  # Enable native streaming

            # Add generation parameters directly
            payload.update(
                {
                    "temperature": getattr(self, "temperature", 0.7),
                    "max_tokens": getattr(self, "max_tokens", 4096),
                    "top_p": getattr(self, "top_p", 1.0),
                }
            )

            # Get session for centralized REST API client
            session = self._get_session()

            # Use centralized REST API client for streaming
            request_config = RestApiRequestBuilder.cortex_complete_request(
                session=session,
                method="POST",
                payload=payload,
                request_timeout=self.request_timeout,
                verify_ssl=self.verify_ssl,
            )

            tool_call_index = -1
            for chunk_json in RestApiClient.make_sync_streaming_request(request_config, "streaming Cortex Complete"):
                if not chunk_json:
                    continue
                try:
                    chunk_data = json.loads(chunk_json)
                except (json.JSONDecodeError, TypeError):
                    continue

                if not isinstance(chunk_data, dict):
                    continue

                for choice in chunk_data.get("choices", []):
                    delta = choice.get("delta", {})
                    delta_type = delta.get("type")

                    if delta_type == "text":
                        content = delta.get("content", "")
                        if content:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(content=content),
                                generation_info={"stream": True},
                            )
                            if run_manager:
                                run_manager.on_llm_new_token(content)
                            yield chunk

                    elif delta_type == "tool_use":
                        if delta.get("tool_use_id"):
                            tool_call_index += 1
                        tool_call_chunk = ToolCallChunk(
                            name=delta.get("name"),
                            args=delta.get("input", ""),
                            id=delta.get("tool_use_id"),
                            index=max(tool_call_index, 0),
                        )
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content="",
                                tool_call_chunks=[tool_call_chunk],
                            ),
                            generation_info={"stream": True},
                        )

        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, "stream via REST API")

    async def _astream_via_rest_api(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream chat completions via REST API using aiohttp for true async."""
        # Build REST API payload with streaming enabled
        payload = self._build_rest_api_payload(messages)
        payload["stream"] = True  # Enable native streaming

        # Add generation parameters directly
        payload.update(
            {
                "temperature": getattr(self, "temperature", 0.7),
                "max_tokens": getattr(self, "max_tokens", 4096),
                "top_p": getattr(self, "top_p", 1.0),
            }
        )

        session = self._get_session()

        request_config = RestApiRequestBuilder.cortex_complete_request(
            session=session,
            method="POST",
            payload=payload,
            request_timeout=self.request_timeout,
            verify_ssl=self.verify_ssl,
        )

        tool_call_index = -1
        async for chunk_json in RestApiClient.make_async_streaming_request(
            request_config, "async streaming Cortex Complete"
        ):
            if not chunk_json:
                continue
            try:
                chunk_data = json.loads(chunk_json)
            except (json.JSONDecodeError, TypeError):
                continue

            if not isinstance(chunk_data, dict):
                continue

            for choice in chunk_data.get("choices", []):
                delta = choice.get("delta", {})
                delta_type = delta.get("type")

                if delta_type == "text":
                    content = delta.get("content", "")
                    if content:
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=content),
                            generation_info={"stream": True},
                        )
                        if run_manager:
                            await run_manager.on_llm_new_token(content)
                        yield chunk

                elif delta_type == "tool_use":
                    if delta.get("tool_use_id"):
                        tool_call_index += 1
                    tool_call_chunk = ToolCallChunk(
                        name=delta.get("name"),
                        args=delta.get("input", ""),
                        id=delta.get("tool_use_id"),
                        index=max(tool_call_index, 0),
                    )
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            tool_call_chunks=[tool_call_chunk],
                        ),
                        generation_info={"stream": True},
                    )

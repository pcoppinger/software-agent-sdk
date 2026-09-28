import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from openhands.sdk import LLM, Message, TextContent
from openhands.sdk.context.condenser import (
    LLMSummarizingCondenser,
    TeamsCheckpointCondenser,
)
from openhands.sdk.settings import LLMSummarizingCondenserSettings
from openhands.sdk.settings.model import OpenHandsAgentSettings


def test_teams_checkpoint_settings_build_the_checkpoint_condenser() -> None:
    settings = OpenHandsAgentSettings.model_validate(
        {
            "condenser": {
                "condenser_kind": "teams_checkpoint",
                "max_size": 1000,
                "max_tokens": 196608,
                "keep_first": 2,
            }
        }
    )

    agent = settings.create_agent()

    assert type(settings.condenser).__name__ == "TeamsCheckpointCondenserSettings"
    assert isinstance(agent.condenser, TeamsCheckpointCondenser)
    assert agent.condenser.kind == "TeamsCheckpointCondenser"


def test_agent_settings_preserve_tool_call_completion_requirement() -> None:
    settings = OpenHandsAgentSettings.model_validate(
        {"require_tool_call_for_completion": True}
    )

    assert settings.create_agent().require_tool_call_for_completion is True


def test_default_condenser_disables_explicit_provider_thinking() -> None:
    primary = LLM(
        model="ollama_chat/qwen",
        max_output_tokens=24_576,
        reasoning_effort="high",
        stream=True,
        litellm_extra_body={"think": True, "keep_alive": -1},
    )

    condenser = LLMSummarizingCondenserSettings().build_condenser(primary)

    assert isinstance(condenser, LLMSummarizingCondenser)
    assert primary.stream is True
    assert primary.reasoning_effort == "high"
    assert primary.litellm_extra_body["think"] is True
    assert condenser.llm.stream is False
    assert condenser.llm.reasoning_effort == "none"
    assert condenser.llm.max_output_tokens == 4_096
    assert condenser.llm.litellm_extra_body == {
        "think": False,
        "keep_alive": -1,
    }


def test_default_condenser_does_not_add_provider_specific_thinking_option() -> None:
    primary = LLM(
        model="openai/gpt-4o",
        max_output_tokens=8_192,
        reasoning_effort="high",
    )

    condenser = LLMSummarizingCondenserSettings().build_condenser(primary)

    assert isinstance(condenser, LLMSummarizingCondenser)
    assert condenser.llm.litellm_extra_body == {}
    assert condenser.llm.reasoning_effort == "high"
    assert condenser.llm.max_output_tokens == 8_192


def test_explicit_condenser_llm_does_not_inherit_primary_thinking() -> None:
    primary = LLM(
        model="openai/primary",
        base_url="http://127.0.0.1:8800/v1",
        api_key="primary-key",
        max_output_tokens=24_576,
        litellm_extra_body={
            "reasoning_effort": "xhigh",
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        },
    )
    summary_llm = LLM(
        model="openai/summary",
        base_url="http://127.0.0.1:8802/v1",
        api_key="summary-key",
        stream=True,
        max_output_tokens=4_096,
        litellm_extra_body={
            "reasoning_effort": "low",
            "chat_template_kwargs": {
                "enable_thinking": False,
                "preserve_thinking": False,
            },
        },
    )

    condenser = LLMSummarizingCondenserSettings(llm=summary_llm).build_condenser(
        primary
    )

    assert isinstance(condenser, LLMSummarizingCondenser)
    assert condenser.llm is not primary
    assert condenser.llm is not summary_llm
    assert condenser.llm.model == "openai/summary"
    assert condenser.llm.base_url == "http://127.0.0.1:8802/v1"
    assert condenser.llm.stream is False
    assert condenser.llm.usage_id == "condenser"
    assert condenser.llm.max_output_tokens == 4_096
    assert condenser.llm.litellm_extra_body == {
        "reasoning_effort": "low",
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        },
    }


def test_explicit_condenser_llm_sends_non_thinking_options_on_wire() -> None:
    received: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.update(json.loads(body))
            response = json.dumps(
                {
                    "id": "wire-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "summary",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "summary"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    try:
        summary_llm = LLM(
            model="openai/summary",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="summary-key",
            stream=False,
            max_output_tokens=4_096,
            num_retries=0,
            timeout=5,
            litellm_extra_body={
                "reasoning_effort": "low",
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "preserve_thinking": False,
                },
            },
        )

        summary_llm.completion(
            messages=[Message(role="user", content=[TextContent(text="summarize")])]
        )
    finally:
        server.server_close()
        thread.join(timeout=5)

    assert received["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }

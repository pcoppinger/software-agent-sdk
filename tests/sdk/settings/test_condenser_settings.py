from openhands.sdk import LLM
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.settings import LLMSummarizingCondenserSettings


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

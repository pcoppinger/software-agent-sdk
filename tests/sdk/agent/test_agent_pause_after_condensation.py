from unittest.mock import AsyncMock, patch

import pytest

from openhands.sdk.agent import Agent
from openhands.sdk.conversation import Conversation, LocalConversation
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm import LLM


def _conversation(*, pause_after_condensation: bool) -> tuple[Agent, LocalConversation]:
    agent = Agent(llm=LLM(model="test-model"), tools=[])
    tags = (
        {"tekroopauseaftercondensation": "true"} if pause_after_condensation else None
    )
    conversation = Conversation(agent=agent, tags=tags)
    conversation._ensure_agent_ready()
    conversation.state.execution_status = ConversationExecutionStatus.RUNNING
    return agent, conversation


def test_step_pauses_at_opted_in_condensation_boundary() -> None:
    agent, conversation = _conversation(pause_after_condensation=True)
    condensation = Condensation(summary="summary", llm_response_id="response")
    events = []

    with patch(
        "openhands.sdk.agent.agent.prepare_llm_messages",
        return_value=condensation,
    ):
        agent.step(conversation, on_event=events.append)

    assert events == [condensation]
    assert conversation.state.execution_status == ConversationExecutionStatus.PAUSED


def test_step_preserves_default_auto_continuation_after_condensation() -> None:
    agent, conversation = _conversation(pause_after_condensation=False)
    condensation = Condensation(summary="summary", llm_response_id="response")

    with patch(
        "openhands.sdk.agent.agent.prepare_llm_messages",
        return_value=condensation,
    ):
        agent.step(conversation, on_event=lambda _: None)

    assert conversation.state.execution_status == ConversationExecutionStatus.RUNNING


@pytest.mark.asyncio
async def test_astep_pauses_at_opted_in_condensation_boundary() -> None:
    agent, conversation = _conversation(pause_after_condensation=True)
    condensation = Condensation(summary="summary", llm_response_id="response")
    events = []

    with patch(
        "openhands.sdk.agent.agent.aprepare_llm_messages",
        new=AsyncMock(return_value=condensation),
    ):
        await agent.astep(conversation, on_event=events.append)

    assert events == [condensation]
    assert conversation.state.execution_status == ConversationExecutionStatus.PAUSED

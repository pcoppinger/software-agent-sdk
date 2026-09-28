from openhands.sdk.context.condenser import TeamsCheckpointCondenser
from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.event.llm_convertible import MessageEvent
from openhands.sdk.llm import LLM, Message, TextContent


def test_teams_checkpoint_condenser_omits_llm_summary() -> None:
    condenser = TeamsCheckpointCondenser(
        llm=LLM(model="test-model"), max_size=10, keep_first=2
    )
    view = View.from_events(
        [
            MessageEvent(
                llm_message=Message(role="user", content=[TextContent(text="event")]),
                source="user",
            )
            for _ in range(11)
        ]
    )

    condensation = condenser.condense(view)

    assert isinstance(condensation, Condensation)
    assert condensation.summary is None
    assert condensation.summary_offset is None
    assert condensation.llm_response_id == "teams-checkpoint"
    assert condensation.forgotten_event_ids

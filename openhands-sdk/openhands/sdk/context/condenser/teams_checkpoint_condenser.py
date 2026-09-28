from collections.abc import Sequence

from openhands.sdk.context.condenser.base import NoCondensationAvailableException
from openhands.sdk.context.condenser.llm_summarizing_condenser import (
    LLMSummarizingCondenser,
)
from openhands.sdk.context.view import View
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm import LLM


class TeamsCheckpointCondenser(LLMSummarizingCondenser):
    """Condense a Teams conversation without introducing an LLM-written summary.

    Teams owns the restart checkpoint and appends it after this event. Keeping an
    independently generated summary in the model view would make an unverified
    narrative compete with that canonical checkpoint. This condenser preserves the
    normal, bounded event-selection rules but emits no summary.
    """

    def get_condensation(
        self, view: View, agent_llm: LLM | None = None
    ) -> Condensation:
        forgotten_events, _ = self._get_forgotten_events(view, agent_llm=agent_llm)
        if not forgotten_events:
            raise NoCondensationAvailableException("No events available to condense.")
        return Condensation(
            forgotten_event_ids={event.id for event in forgotten_events},
            llm_response_id="teams-checkpoint",
        )

    def _generate_condensation(
        self,
        forgotten_events: Sequence[LLMConvertibleEvent],
        summary_offset: int,
        max_event_str_length: int | None = None,
    ) -> Condensation:
        del summary_offset, max_event_str_length
        if not forgotten_events:
            raise NoCondensationAvailableException("No events available to condense.")
        return Condensation(
            forgotten_event_ids={event.id for event in forgotten_events},
            llm_response_id="teams-checkpoint",
        )

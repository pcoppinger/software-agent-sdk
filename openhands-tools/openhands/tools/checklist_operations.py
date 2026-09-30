"""Separate checklist reading and replacement without a command discriminator."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.sdk.tool.schema import Schema
from openhands.tools.task_tracker.definition import (
    TaskItem,
    TaskTrackerAction,
    TaskTrackerExecutor,
    TaskTrackerObservation,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


class ViewChecklistAction(Action):
    pass


class ReplaceChecklistAction(Action):
    items: list[TaskItem] = Field(
        description="Complete replacement checklist, or [] to clear it."
    )


class _ChecklistExecutor(ToolExecutor[Action, TaskTrackerObservation]):
    def __init__(self, tracker: TaskTrackerExecutor):
        self.tracker = tracker

    def __call__(
        self, action: Action, conversation: "LocalConversation | None" = None
    ) -> TaskTrackerObservation:
        if isinstance(action, ViewChecklistAction):
            return self.tracker(TaskTrackerAction(command="view"), conversation)
        assert isinstance(action, ReplaceChecklistAction)
        return self.tracker(
            TaskTrackerAction(command="plan", task_list=action.items), conversation
        )


class _ChecklistToolBase:
    action_model: ClassVar[type[Action]]
    tool_description: ClassVar[str]

    def _get_tool_schema(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> dict[str, Any]:
        schema = ToolDefinition._get_tool_schema(
            cast(ToolDefinition[Action, TaskTrackerObservation], self),
            add_security_risk_prediction=add_security_risk_prediction,
            action_type=action_type,
        )
        schema["additionalProperties"] = False
        return schema

    @classmethod
    def create(
        cls, conv_state: "ConversationState", tracker: TaskTrackerExecutor | None = None
    ) -> Sequence[ToolDefinition[Action, TaskTrackerObservation]]:
        tracker = tracker or TaskTrackerExecutor(save_dir=conv_state.persistence_dir)
        tool_class = cast(type[ToolDefinition[Action, TaskTrackerObservation]], cls)
        return [
            tool_class(
                action_type=cls.action_model,
                observation_type=TaskTrackerObservation,
                description=cls.tool_description,
                annotations=ToolAnnotations(
                    title=tool_class.name,
                    readOnlyHint=tool_class.name == "view_checklist",
                    destructiveHint=False,
                    idempotentHint=tool_class.name == "view_checklist",
                    openWorldHint=False,
                ),
                executor=_ChecklistExecutor(tracker),
            )
        ]


class ViewChecklistTool(
    _ChecklistToolBase, ToolDefinition[ViewChecklistAction, TaskTrackerObservation]
):
    name = "view_checklist"
    action_model: ClassVar[type[Action]] = ViewChecklistAction
    tool_description: ClassVar[str] = "Show the current checklist without changing it."


class ReplaceChecklistTool(
    _ChecklistToolBase, ToolDefinition[ReplaceChecklistAction, TaskTrackerObservation]
):
    name = "replace_checklist"
    action_model: ClassVar[type[Action]] = ReplaceChecklistAction
    tool_description: ClassVar[str] = (
        "Replace the entire checklist with structured items."
    )


class ChecklistOperations(ToolDefinition[Action, TaskTrackerObservation]):
    name = "checklist_operations"

    @classmethod
    def create(
        cls, conv_state: "ConversationState"
    ) -> list[ToolDefinition[Any, TaskTrackerObservation]]:
        tracker = TaskTrackerExecutor(save_dir=conv_state.persistence_dir)
        tools: list[ToolDefinition[Any, TaskTrackerObservation]] = []
        for tool_class in (ViewChecklistTool, ReplaceChecklistTool):
            tools.extend(tool_class.create(conv_state, tracker=tracker))
        return tools


register_tool(ViewChecklistTool.name, ViewChecklistTool)
register_tool(ReplaceChecklistTool.name, ReplaceChecklistTool)
register_tool(ChecklistOperations.name, ChecklistOperations)

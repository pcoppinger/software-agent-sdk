"""One-operation command tools sharing a conversation-scoped terminal session."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast
from uuid import uuid4

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.sdk.tool.schema import Schema
from openhands.tools.terminal.definition import TerminalAction, TerminalObservation
from openhands.tools.terminal.impl import TerminalExecutor


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


class RunCommandAction(Action):
    command: str = Field(
        min_length=1, description="One shell command to run in the workspace."
    )
    timeout_seconds: float = Field(
        default=30,
        ge=1,
        le=300,
        description=(
            "Return after this many seconds; if still running, use wait_for_command."
        ),
    )


class WaitForCommandAction(Action):
    command_id: str = Field(
        description="Identifier returned by run_command for the still-running command."
    )
    timeout_seconds: float = Field(default=30, ge=1, le=300)


class StopCommandAction(Action):
    command_id: str = Field(
        description="Identifier returned by run_command for the still-running command."
    )


class CommandObservation(TerminalObservation):
    active_command_id: str | None = None


class _CommandSession:
    def __init__(self, working_dir: str, save_dir: str | None):
        # A single subprocess-backed session makes continuation and cancellation
        # unambiguous; tmux pooling may route the next call to another pane.
        self.terminal = TerminalExecutor(
            working_dir=working_dir,
            terminal_type="subprocess",
            full_output_save_dir=save_dir,
        )
        self.active_command_id: str | None = None

    def execute(
        self, action: Action, conversation: "LocalConversation | None"
    ) -> CommandObservation:
        if isinstance(action, RunCommandAction):
            if self.active_command_id is not None:
                return self.error(
                    "A command is already running; wait or stop it first."
                )
            command_id = str(uuid4())
            observation = self.terminal(
                TerminalAction(command=action.command, timeout=action.timeout_seconds),
                conversation,
            )
            if observation.exit_code == -1:
                self.active_command_id = command_id
        elif isinstance(action, (WaitForCommandAction, StopCommandAction)):
            if action.command_id != self.active_command_id:
                return self.error("Unknown or completed command_id.")
            terminal_action = (
                TerminalAction(command="", timeout=action.timeout_seconds)
                if isinstance(action, WaitForCommandAction)
                else TerminalAction(command="C-c", is_input=True, timeout=30)
            )
            observation = self.terminal(terminal_action, conversation)
            if observation.exit_code != -1:
                self.active_command_id = None
        else:
            return self.error("Unsupported command operation.")
        return CommandObservation.from_text(
            text=observation.text,
            is_error=observation.is_error,
            command=observation.command,
            exit_code=observation.exit_code,
            timeout=observation.timeout,
            metadata=observation.metadata,
            full_output_save_dir=observation.full_output_save_dir,
            active_command_id=self.active_command_id,
        )

    def error(self, message: str) -> CommandObservation:
        return CommandObservation.from_text(
            text=message,
            is_error=True,
            command=None,
            exit_code=None,
            active_command_id=self.active_command_id,
        )


class _CommandExecutor(ToolExecutor[Action, CommandObservation]):
    def __init__(self, session: _CommandSession):
        self.session = session

    def __call__(
        self, action: Action, conversation: "LocalConversation | None" = None
    ) -> CommandObservation:
        return self.session.execute(action, conversation)


class _CommandToolBase:
    action_model: ClassVar[type[Action]]
    tool_description: ClassVar[str]

    def _get_tool_schema(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> dict[str, Any]:
        schema = ToolDefinition._get_tool_schema(
            cast(ToolDefinition[Action, CommandObservation], self),
            add_security_risk_prediction=add_security_risk_prediction,
            action_type=action_type,
        )
        schema["additionalProperties"] = False
        return schema

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=("terminal:session",), declared=True)

    @classmethod
    def create(
        cls, conv_state: "ConversationState", session: _CommandSession | None = None
    ) -> Sequence[ToolDefinition[Action, CommandObservation]]:
        session = session or _CommandSession(
            conv_state.workspace.working_dir, conv_state.env_observation_persistence_dir
        )
        tool_class = cast(type[ToolDefinition[Action, CommandObservation]], cls)
        return [
            tool_class(
                action_type=cls.action_model,
                observation_type=CommandObservation,
                description=cls.tool_description,
                annotations=ToolAnnotations(
                    title=tool_class.name,
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=_CommandExecutor(session),
            )
        ]


class RunCommandTool(
    _CommandToolBase, ToolDefinition[RunCommandAction, CommandObservation]
):
    name = "run_command"
    action_model: ClassVar[type[Action]] = RunCommandAction
    tool_description: ClassVar[str] = (
        "Run one shell command in the workspace. If still running, use its "
        "active_command_id with wait_for_command or stop_command."
    )


class WaitForCommandTool(
    _CommandToolBase, ToolDefinition[WaitForCommandAction, CommandObservation]
):
    name = "wait_for_command"
    action_model: ClassVar[type[Action]] = WaitForCommandAction
    tool_description: ClassVar[str] = (
        "Wait for a previously started command and return new output. "
        "Does not start or repeat the command."
    )


class StopCommandTool(
    _CommandToolBase, ToolDefinition[StopCommandAction, CommandObservation]
):
    name = "stop_command"
    action_model: ClassVar[type[Action]] = StopCommandAction
    tool_description: ClassVar[str] = "Interrupt the specified still-running command."


class CommandOperations(ToolDefinition[Action, CommandObservation]):
    name = "command_operations"

    @classmethod
    def create(
        cls, conv_state: "ConversationState"
    ) -> list[ToolDefinition[Any, CommandObservation]]:
        session = _CommandSession(
            conv_state.workspace.working_dir, conv_state.env_observation_persistence_dir
        )
        tools: list[ToolDefinition[Any, CommandObservation]] = []
        for tool_class in (RunCommandTool, WaitForCommandTool, StopCommandTool):
            tools.extend(tool_class.create(conv_state, session=session))
        return tools


register_tool(RunCommandTool.name, RunCommandTool)
register_tool(WaitForCommandTool.name, WaitForCommandTool)
register_tool(StopCommandTool.name, StopCommandTool)
register_tool(CommandOperations.name, CommandOperations)

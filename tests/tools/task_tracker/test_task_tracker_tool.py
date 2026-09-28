from uuid import uuid4

import pytest
from jsonschema import Draft202012Validator
from pydantic import SecretStr

from openhands.sdk.agent import Agent
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.llm import LLM
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.task_tracker.definition import (
    TaskItem,
    TaskTrackerAction,
    TaskTrackerExecutor,
    TaskTrackerTool,
)


def test_plan_without_task_list_does_not_clear_existing_tasks():
    executor = TaskTrackerExecutor()
    executor(
        TaskTrackerAction(
            command="plan",
            task_list=[TaskItem(title="Keep me", notes="", status="todo")],
        )
    )

    result = executor(TaskTrackerAction(command="plan"))

    assert result.is_error
    assert [task.title for task in result.task_list] == ["Keep me"]
    assert [task.title for task in executor(TaskTrackerAction()).task_list] == [
        "Keep me"
    ]


def test_explicit_empty_plan_clears_existing_tasks():
    executor = TaskTrackerExecutor()
    executor(
        TaskTrackerAction(
            command="plan",
            task_list=[TaskItem(title="Remove me", notes="", status="todo")],
        )
    )

    result = executor(TaskTrackerAction(command="plan", task_list=[]))

    assert not result.is_error
    assert result.task_list == []


@pytest.mark.parametrize(
    ("arguments", "valid"),
    [
        ({"command": "view"}, True),
        ({"command": "plan"}, False),
        ({"command": "plan", "task_list": []}, True),
    ],
)
def test_task_tracker_provider_schema_requires_list_for_plan(
    tmp_path, arguments, valid
):
    llm = LLM(model="gpt-4o-mini", api_key=SecretStr("test-key"), usage_id="test")
    state = ConversationState.create(
        id=uuid4(),
        agent=Agent(llm=llm, tools=[]),
        workspace=LocalWorkspace(working_dir=str(tmp_path)),
    )
    function = TaskTrackerTool.create(state)[0].to_openai_tool()["function"]
    assert "parameters" in function
    schema = function["parameters"]
    Draft202012Validator.check_schema(schema)
    assert Draft202012Validator(schema).is_valid(arguments) is valid

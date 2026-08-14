import asyncio
import threading

import pytest

import openhands.sdk.llm.llm as llm_module
from openhands.sdk.llm import LLM


def _local_ollama_llm() -> LLM:
    return LLM(
        model="ollama_chat/qwen3.6:35b-a3b-q4_K_M",
        base_url="http://127.0.0.1:11434",
        api_key="ollama",
        max_input_tokens=32768,
    )


@pytest.mark.asyncio
async def test_local_ollama_capacity_limits_parallel_calls(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "_LOCAL_OLLAMA_CAPACITY",
        threading.BoundedSemaphore(2),
    )
    llm = _local_ollama_llm()
    release = asyncio.Event()
    two_entered = asyncio.Event()
    active = 0
    peak = 0

    async def worker() -> None:
        nonlocal active, peak
        async with llm._alocal_ollama_capacity_ctx():
            active += 1
            peak = max(peak, active)
            if active == 2:
                two_entered.set()
            await release.wait()
            active -= 1

    tasks = [asyncio.create_task(worker()) for _ in range(3)]
    await asyncio.wait_for(two_entered.wait(), timeout=1)
    await asyncio.sleep(0.05)
    assert active == 2
    assert peak == 2

    release.set()
    await asyncio.gather(*tasks)
    assert peak == 2


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leak_local_ollama_capacity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_LOCAL_OLLAMA_CAPACITY",
        threading.BoundedSemaphore(1),
    )
    llm = _local_ollama_llm()
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def hold_capacity() -> None:
        async with llm._alocal_ollama_capacity_ctx():
            holder_entered.set()
            await release_holder.wait()

    async def acquire_capacity() -> None:
        async with llm._alocal_ollama_capacity_ctx():
            return

    holder = asyncio.create_task(hold_capacity())
    await asyncio.wait_for(holder_entered.wait(), timeout=1)

    waiter = asyncio.create_task(acquire_capacity())
    await asyncio.sleep(0.05)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release_holder.set()
    await holder
    await asyncio.wait_for(acquire_capacity(), timeout=1)


def test_local_ollama_detection_includes_openai_compatible_endpoint() -> None:
    llm = LLM(
        model="openai/qwen3.6:35b-a3b-q4_K_M",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        max_input_tokens=32768,
    )

    assert llm._uses_local_ollama()

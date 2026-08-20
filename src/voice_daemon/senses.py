import asyncio
import sys
from typing import Awaitable, Callable, Iterable, Optional, Protocol, Sequence, TextIO, Tuple

from runtime_core.runtime import RuntimeContext, Sense
from voice_daemon.vision_context import maybe_augment_turn_with_vision_context


class ConsoleLineSource:
    def __init__(self, stream: Optional[TextIO] = None):
        self._stream = stream or sys.stdin

    async def read_line(self) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._stream.readline)


class TextDelegate(Protocol):
    def wake_parts(self, text: str) -> Tuple[bool, str, str]:
        ...

    async def handle_console_approval(self, line: str, context: RuntimeContext) -> bool:
        ...


class TextSense(Sense):
    name = "text"

    def __init__(
        self,
        wake_name: str,
        wake_aliases: Iterable[str],
        delegate: TextDelegate,
        *,
        line_source: Optional[ConsoleLineSource] = None,
        vision_query_triggers: Optional[Sequence[str]] = None,
    ):
        self._wake_name = wake_name
        self._wake_aliases = list(wake_aliases)
        self._delegate = delegate
        self._line_source = line_source or ConsoleLineSource()
        self._vision_query_triggers = list(vision_query_triggers or [])

    async def run(self, context: RuntimeContext) -> None:
        print("[voice] Enter text. Use '+ <text>' to patch last turn. Use /approve or /deny for permissions.")
        print(
            "[voice] Wake words: {name} / {aliases}".format(
                name=self._wake_name,
                aliases=", ".join(self._wake_aliases),
            )
        )
        while True:
            line = await self._line_source.read_line()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("/wake"):
                await context.send_wake()
                continue
            if await self._delegate.handle_console_approval(line, context):
                continue
            if line.startswith("+"):
                appended = line[1:].strip()
                if not context.last_turn_id:
                    print("[voice] No turn to patch")
                    continue
                await context.patch_turn(context.last_turn_id, appended)
                continue

            is_wake, remainder, forward_text = self._delegate.wake_parts(line)
            if is_wake:
                await context.send_wake({"wake_word": line})
                if not remainder:
                    print("[voice] Wake detected")
                    continue
                line = forward_text
            line = await maybe_augment_turn_with_vision_context(
                context,
                line,
                triggers=self._vision_query_triggers,
            )
            await context.submit_turn(line)


class ConsoleApprovalSense(Sense):
    name = "console_approval"

    def __init__(
        self,
        delegate: TextDelegate,
        *,
        line_source: Optional[ConsoleLineSource] = None,
    ):
        self._delegate = delegate
        self._line_source = line_source or ConsoleLineSource()

    async def run(self, context: RuntimeContext) -> None:
        print("[voice] Audio mode active. Type /approve or /deny for permissions.")
        while True:
            line = await self._line_source.read_line()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            if line == "/exit":
                break
            await self._delegate.handle_console_approval(line, context)


class AudioSense(Sense):
    name = "audio"

    def __init__(self, run_loop: Callable[[RuntimeContext], Awaitable[None]]):
        self._run_loop = run_loop

    async def run(self, context: RuntimeContext) -> None:
        await self._run_loop(context)

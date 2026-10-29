# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio

from musegadget import voice_call as voice
from test_streaming import chunk, connected
from test_voice_call import Output, TONE, admit, cavd, hangup, line, next_event, source

S = voice.CallState


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(wait(), 2)


def test_working_shows_once_muse_answered_an_open_turn():
    async def scenario():
        async with connected(register_device=False) as (session, vm, unused):
            states = []
            call = voice.VoiceCall(session, device_id="homelink-synthetic", half_duplex=False,
                                   on_state=states.append)
            assert call.state == S.CONNECTING
            task = asyncio.create_task(call.run(source(TONE), Output()))
            media, start = await admit(vm)
            await next_event(vm)

            async def send(*events):
                await chunk(vm, media, b"".join(line(event) for event in events))

            async def expect(state):
                await wait_until(lambda: call.state == state and states and states[-1] == state)

            await expect(S.LISTENING)
            await send(cavd())
            await wait_until(lambda: S.SPEAKING in states)
            await expect(S.LISTENING)
            await send({"type": "input_audio_buffer.speech_started", "item_id": "i1"})
            await send({"type": "hatch.voice.turn_started", "submission_message_id": "s1",
                        "item_id": "i1"})
            await wait_until(lambda: "s1" in call._active_turns)
            await asyncio.sleep(0.12)
            assert call.state == S.LISTENING  # about to answer, not working
            await send({"type": "response.created", "response": {"id": "r1"}},
                       {"type": "response.done", "response": {"id": "r1", "status": "completed"}})
            await expect(S.WORKING)  # answered ("on it"), and the turn is still open
            await send({"type": "hatch.voice.turn_started", "submission_message_id": "s1b",
                        "item_id": "i1"})  # a delayed reply's turn for the same item
            await wait_until(lambda: "s1b" in call._active_turns)
            await asyncio.sleep(0.12)
            assert call.state == S.WORKING
            await send({"type": "input_audio_buffer.speech_started", "item_id": "tv"})
            await expect(S.LISTENING)  # the caller (or the room) talks
            await send({"type": "hatch.voice.input_suppressed", "item_id": "tv"})
            await expect(S.WORKING)  # side speech the server dropped
            await send({"type": "hatch.voice.turn_settled", "response_message_id": "s1b",
                        "async_handoffs_pending": 1})
            await expect(S.LISTENING)
            await send({"type": "input_audio_buffer.speech_started", "item_id": "i3"},
                       {"type": "input_audio_buffer.speech_stopped", "item_id": "i3"},
                       {"type": "hatch.voice.turn_started", "submission_message_id": "s3",
                        "item_id": "i3"},
                       {"type": "response.created", "response": {"id": "r3"}},
                       {"type": "response.done", "response": {"id": "r3", "status": "cancelled"}})
            await expect(S.WORKING)
            # Nothing pending anywhere: every open turn is done, named or not.
            await send({"type": "hatch.voice.turn_settled", "response_message_id": "unnamed",
                        "async_handoffs_pending": 0})
            await expect(S.LISTENING)
            assert not call._active_turns
            await chunk(vm, media, end=True)
            await hangup(vm, media, start, media_ended=True)
            assert await asyncio.wait_for(task, 2) == "remote_end"
            assert call.state == S.ENDED and states[-1] == S.ENDED
            assert states[0] == S.CONNECTING
            assert all(a != b for a, b in zip(states, states[1:]))  # changes only
    asyncio.run(scenario())


def test_a_failing_state_callback_does_not_end_the_call(caplog):
    async def scenario():
        async with connected(register_device=False) as (session, vm, unused):
            def broken(state):
                raise RuntimeError("synthetic")
            call = voice.VoiceCall(session, device_id="homelink-synthetic", on_state=broken)
            task = asyncio.create_task(call.run(source(TONE), Output()))
            media, start = await admit(vm)
            await next_event(vm)
            await wait_until(lambda: call.state == S.LISTENING)
            await asyncio.sleep(0.12)
            assert not task.done()
            await chunk(vm, media, end=True)
            await hangup(vm, media, start, media_ended=True)
            assert await asyncio.wait_for(task, 2) == "remote_end"
    asyncio.run(scenario())
    assert "voice state callback failed" in caplog.text


def test_an_interrupted_reply_settling_late_neither_answers_nor_retires_the_new_turn():
    async def scenario():
        async with connected(register_device=False) as (session, vm, unused):
            call = voice.VoiceCall(session, device_id="homelink-synthetic", half_duplex=False)
            task = asyncio.create_task(call.run(source(TONE), Output()))
            media, start = await admit(vm)
            await next_event(vm)

            async def send(*events):
                await chunk(vm, media, b"".join(line(event) for event in events))
            await wait_until(lambda: call.state == S.LISTENING)
            await send({"type": "hatch.voice.turn_started", "submission_message_id": "s1",
                        "item_id": "i1"},
                       {"type": "response.created", "response": {"id": "r1",
                                                                 "submission_message_id": "s1"}})
            # The caller talks over r1: their new turn starts before r1 finishes.
            await send({"type": "input_audio_buffer.speech_started", "item_id": "i2"},
                       {"type": "input_audio_buffer.speech_stopped", "item_id": "i2"},
                       {"type": "hatch.voice.turn_started", "submission_message_id": "s2",
                        "item_id": "i2"},
                       {"type": "response.done", "response": {"id": "r1", "status": "cancelled"}},
                       {"type": "hatch.voice.turn_settled", "response_message_id": "s1",
                        "async_handoffs_pending": 0})
            await wait_until(lambda: "s1" in call._settled_turns)
            await asyncio.sleep(0.12)
            assert call._active_turns == {"s2": "i2"}  # not retired with the old one
            assert call.state == S.LISTENING  # and not answered by r1
            await send({"type": "response.created", "response": {"id": "rx",
                                                                 "submission_message_id": "s1"}},
                       {"type": "response.done", "response": {"id": "rx", "status": "completed"}})
            await asyncio.sleep(0.12)
            assert call.state == S.LISTENING  # older work speaking first doesn't answer s2
            await send({"type": "response.created", "response": {"id": "r2",
                                                                 "submission_message_id": "s2"}},
                       {"type": "response.done", "response": {"id": "r2", "status": "completed"}})
            await wait_until(lambda: call.state == S.WORKING)
            await send({"type": "hatch.voice.turn_settled", "response_message_id": "s2",
                        "async_handoffs_pending": 0})
            await wait_until(lambda: not call._active_turns and call.state == S.LISTENING)
            await chunk(vm, media, end=True)
            await hangup(vm, media, start, media_ended=True)
            assert await asyncio.wait_for(task, 2) == "remote_end"
    asyncio.run(scenario())


def test_barge_in_and_input_hold_never_show_working_while_the_caller_has_the_floor():
    async def scenario():
        async with connected(register_device=False) as (session, vm, unused):
            call = voice.VoiceCall(session, device_id="homelink-synthetic", half_duplex=False)
            task = asyncio.create_task(call.run(source(TONE), Output()))
            media, start = await admit(vm)
            await next_event(vm)

            async def send(*events):
                await chunk(vm, media, b"".join(line(event) for event in events))
            await send({"type": "hatch.voice.turn_started", "submission_message_id": "s1",
                        "item_id": "i1"},
                       {"type": "response.created", "response": {"id": "r1"}},
                       {"type": "response.done", "response": {"id": "r1", "status": "completed"}})
            await wait_until(lambda: call.state == S.WORKING)
            await call.barge_in()
            assert call.state == S.LISTENING  # the caller is about to talk
            await send({"type": "input_audio_buffer.speech_started", "item_id": "i2"},
                       {"type": "hatch.voice.switch_input_hold", "held": True})
            await wait_until(lambda: call.held)
            assert call.state == S.WORKING  # the hold ended the caller's speech
            await send({"type": "input_audio_buffer.speech_started", "item_id": "i2"})
            await asyncio.sleep(0.12)
            assert call.state == S.WORKING  # a repeated start for an older item is stale
            await chunk(vm, media, end=True)
            await hangup(vm, media, start, media_ended=True)
            assert await asyncio.wait_for(task, 2) == "remote_end"
    asyncio.run(scenario())


def test_ended_is_reported_even_when_the_end_call_is_refused():
    async def scenario():
        async with connected(register_device=False) as (session, vm, unused):
            states = []
            call = voice.VoiceCall(session, device_id="homelink-synthetic", on_state=states.append)
            task = asyncio.create_task(call.run(source(TONE), Output()))
            media, start = await admit(vm)
            await next_event(vm)
            await wait_until(lambda: call.state == S.LISTENING)
            await chunk(vm, media, end=True)
            await hangup(vm, media, start, status=500, media_ended=True)
            try:
                await asyncio.wait_for(task, 3)
            except Exception:
                pass
            assert call.state == S.ENDED and states[-1] == S.ENDED
            assert str(S.ENDED) == "ended"
    asyncio.run(scenario())


def test_early_media_answers_its_turn_and_a_lost_reply_is_kept_once():
    async def scenario():
        async with connected(register_device=False) as (session, vm, unused):
            call = voice.VoiceCall(session, device_id="homelink-synthetic", half_duplex=False)
            task = asyncio.create_task(call.run(source(TONE), Output()))
            media, start = await admit(vm)
            await next_event(vm)

            async def send(*events):
                await chunk(vm, media, b"".join(line(event) for event in events))
            await wait_until(lambda: call.state == S.LISTENING)
            # The reply's early media opens an unmarked response before its turn.
            await send({"type": "input_audio_buffer.speech_started", "item_id": "i1"},
                       {"type": "input_audio_buffer.speech_stopped", "item_id": "i1"},
                       {"type": "response.created", "response": {"id": "r1"}},
                       {"type": "hatch.voice.turn_started", "submission_message_id": "s1",
                        "item_id": "i1"},
                       {"type": "response.done", "response": {"id": "r1", "status": "completed"}})
            await wait_until(lambda: call.state == S.WORKING)
            # s2's reply never comes: one nothing-pending settlement keeps it.
            settle0 = {"type": "hatch.voice.turn_settled", "async_handoffs_pending": 0}
            await send({"type": "input_audio_buffer.speech_started", "item_id": "i2"},
                       {"type": "input_audio_buffer.speech_stopped", "item_id": "i2"},
                       {"type": "hatch.voice.turn_started", "submission_message_id": "s2",
                        "item_id": "i2"},
                       settle0)
            await wait_until(lambda: "s1" in call._settled_turns)
            assert call._active_turns == {"s2": "i2"}
            await send(settle0)
            await wait_until(lambda: not call._active_turns)
            await chunk(vm, media, end=True)
            await hangup(vm, media, start, media_ended=True)
            assert await asyncio.wait_for(task, 2) == "remote_end"
    asyncio.run(scenario())

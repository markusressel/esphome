"""Integration test: addressable light transition with non-uniform color_correct.

Regression test for the domain-mismatch flash bug. When a light has a non-uniform
color_correct setting (e.g. [100%, 100%, 70%]), starting a new transition from a
stable white state must not produce a visible channel dip (flash) on the first frame.

Root cause (before the fix):
  The transition read current LED bytes back through ESPColorView.get() (which
  un-applies color correction) to obtain a "start" color, then interpolated toward
  an uncorrected target color, and finally re-applied color correction when writing
  each frame via ESPColorView.set_rgbw().  On the very first frame, the uncorrect +
  re-correct round-trip could push a channel below its prior hardware value if
  local_brightness_ had changed between the last write and the transition start.

The fix:
  start() pre-computes corrected_target_color_ = correction_.color_correct(target_color_)
  — the raw LED bytes that should be stored at 100% progress.  apply() then reads raw
  LED bytes (get_*_raw(), no un-correction) and writes pre-corrected bytes directly
  (set_rgbw_raw(), no re-correction).  Interpolation stays in hardware-byte domain
  throughout, so the first frame always writes the unmodified start bytes and there is
  no possibility of a double-correction flash.

Test scenario:
  1. Set light to stable full white with color_correct=[100%,100%,70%] and gamma=1.
     Blue hardware byte = round(255 * 70%) = 178.
  2. Issue another "white at 100%" command with a 1-second transition.
  3. During the transition, observe the raw blue hardware byte.
  4. Assert it starts at 178 (no flash) and ends at 178 (no net change).
  5. Also test an interrupted transition: light → white, then mid-transition issue
     another white command.  Blue byte must never dip below the pre-command value.
"""

from __future__ import annotations

import asyncio

from aioesphomeapi import LightInfo, SensorInfo, SensorState
import pytest

from .state_utils import InitialStateHelper, require_entity
from .types import APIClientConnectedFactory, RunCompiledFunction


@pytest.mark.asyncio
async def test_color_correct_no_flash_same_target(
    yaml_config: str,
    run_compiled: RunCompiledFunction,
    api_client_connected: APIClientConnectedFactory,
) -> None:
    """Transition to the same color must not dip any channel (no double-correction flash)."""
    async with run_compiled(yaml_config), api_client_connected() as client:
        entities, _ = await client.list_entities_services()
        light = require_entity(entities, "test_strip", LightInfo)
        sensor = require_entity(entities, "led0_blue_raw", SensorInfo)

        loop = asyncio.get_running_loop()
        samples: list[tuple[float, float]] = []

        def on_state(state: object) -> None:
            if not isinstance(state, SensorState) or state.key != sensor.key:
                return
            samples.append((loop.time(), state.state))

        initial_state_helper = InitialStateHelper(entities)
        client.subscribe_states(initial_state_helper.on_state_wrapper(on_state))
        await initial_state_helper.wait_for_initial_states()

        # Step 1: set light to stable full white (instant, no transition).
        client.light_command(
            key=light.key,
            state=True,
            rgb=(1.0, 1.0, 1.0),
            brightness=1.0,
            transition_length=0,
        )
        await asyncio.sleep(0.2)

        # Record the settled blue byte value (should be ~178 = round(255 * 0.70)).
        stable_samples = [v for (_, v) in samples]
        assert stable_samples, "no sensor samples in stable phase"
        stable_blue = stable_samples[-1]
        # With gamma=1 and color_correct=[100%,100%,70%], the stored byte for full
        # blue (255 uncorrected) is esp_scale8_twice(255,178,255) = ~178.
        assert stable_blue >= 170, (
            f"stable blue byte {stable_blue} is unexpectedly low before transition"
        )

        # Step 2: start a 1-second transition to the same color (white).
        samples.clear()
        transition_s = 1.0
        command_time = loop.time()
        client.light_command(
            key=light.key,
            state=True,
            rgb=(1.0, 1.0, 1.0),
            brightness=1.0,
            transition_length=transition_s,
        )
        await asyncio.sleep(transition_s + 0.2)

        # Rebase to command time.
        post_command = [
            (t - command_time, v) for (t, v) in samples if t >= command_time
        ]
        assert post_command, "no samples received after command"

        # Assertion 1: the blue byte must never dip below the stable value (no flash).
        # Allow a tolerance of 2 to account for integer rounding.
        min_blue = min(v for (_, v) in post_command)
        assert min_blue >= stable_blue - 2, (
            f"blue byte dipped from {stable_blue} to {min_blue} during same-target "
            "transition — double-correction flash detected"
        )

        # Assertion 2: the blue byte must end near its starting value (same target).
        final_samples = [v for (_, v) in post_command[-5:]]
        assert abs(max(final_samples) - stable_blue) <= 2, (
            f"blue byte ended at {max(final_samples)} but started at {stable_blue} "
            "for a same-target transition (expected no net change)"
        )


@pytest.mark.asyncio
async def test_color_correct_interrupted_transition_no_flash(
    yaml_config: str,
    run_compiled: RunCompiledFunction,
    api_client_connected: APIClientConnectedFactory,
) -> None:
    """Interrupting a transition mid-way must not cause a flash on the first frame of the new transition."""
    async with run_compiled(yaml_config), api_client_connected() as client:
        entities, _ = await client.list_entities_services()
        light = require_entity(entities, "test_strip", LightInfo)
        sensor = require_entity(entities, "led0_blue_raw", SensorInfo)

        loop = asyncio.get_running_loop()
        samples: list[tuple[float, float]] = []

        def on_state(state: object) -> None:
            if not isinstance(state, SensorState) or state.key != sensor.key:
                return
            samples.append((loop.time(), state.state))

        initial_state_helper = InitialStateHelper(entities)
        client.subscribe_states(initial_state_helper.on_state_wrapper(on_state))
        await initial_state_helper.wait_for_initial_states()

        # Step 1: set light to stable full white.
        client.light_command(
            key=light.key,
            state=True,
            rgb=(1.0, 1.0, 1.0),
            brightness=1.0,
            transition_length=0,
        )
        await asyncio.sleep(0.2)

        # Step 2: start a 2-second transition to off.
        samples.clear()
        client.light_command(
            key=light.key,
            state=False,
            transition_length=2.0,
        )
        # Let it run for 0.5 s (roughly 25% of the way through).
        await asyncio.sleep(0.5)

        # Step 3: interrupt with a new transition back to white.
        interrupt_time = loop.time()
        blue_before_interrupt = samples[-1][1] if samples else 178
        client.light_command(
            key=light.key,
            state=True,
            rgb=(1.0, 1.0, 1.0),
            brightness=1.0,
            transition_length=1.0,
        )
        await asyncio.sleep(1.2)

        # Collect samples after the interrupt.
        post_interrupt = [
            (t - interrupt_time, v) for (t, v) in samples if t >= interrupt_time
        ]
        assert post_interrupt, "no samples received after interrupt"

        # On the first frame after the interrupt the blue byte must not drop below the
        # pre-interrupt value (no flash at the transition boundary).
        first_frames = [v for (t, v) in post_interrupt if t < 0.1]
        if first_frames:
            assert min(first_frames) >= blue_before_interrupt - 2, (
                f"blue byte dipped from {blue_before_interrupt} to {min(first_frames)} "
                "at transition interrupt boundary — flash detected"
            )

        # The transition should end near the target white value (~178).
        final_samples = [v for (_, v) in post_interrupt[-5:]]
        assert max(final_samples) >= 170, (
            f"blue byte only reached {max(final_samples)} at end of interrupted "
            "transition (expected ~178 for full white)"
        )

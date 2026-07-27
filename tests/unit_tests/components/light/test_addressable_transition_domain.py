"""Unit tests for the addressable-light transition domain-consistency fix.

These tests simulate the C++ transition math in Python to verify that:

1. The pre-computed corrected target (corrected_target_color_) plus raw-byte
   start (uniform_start_raw_) approach never causes a channel dip on the first
   frame of a transition — the "double-correction flash" regression guard.

2. Interrupted transitions also start at the correct value.

3. The final value of every transition equals the corrected target exactly.

The simulation mirrors the C++ implementation in addressable_light.cpp:
  - color_correct_*: applies max_brightness and gamma (no-op gamma=1 for simplicity)
  - subtract_scaled_difference: the in-place lerp helper
  - esp_scale8 / esp_scale8_twice: the integer-scaling helpers from color.h
"""

from __future__ import annotations

import math
import pytest


# ---------------------------------------------------------------------------
# C++ math helpers (mirrors color.h / esp_color_correction.h)
# ---------------------------------------------------------------------------

def esp_scale8(a: int, b: int) -> int:
    """Scale byte a by byte b: (a * b + 127) // 255."""
    return (int(a) * int(b) + 127) // 255


def esp_scale8_twice(a: int, b: int, c: int) -> int:
    """Scale byte a by b then by c using a single intermediate round."""
    # Mirrors: (uint16_t(i) * scale1 * scale2 + (scale1 * scale2 / 2)) / 65025
    intermediate = int(a) * int(b) * int(c)
    # approximate: same formula used in C++ — two sequential scale8s
    return esp_scale8(esp_scale8(a, b), c)


def gamma_correct(value: int, gamma: float = 1.0) -> int:
    """Apply gamma correction to a byte value."""
    if value == 0:
        return 0
    if value == 255:
        return 255
    return round((value / 255.0) ** gamma * 255.0)


def gamma_uncorrect(value: int, gamma: float = 1.0) -> int:
    """Reverse gamma correction."""
    if value == 0:
        return 0
    if value == 255:
        return 255
    return round((value / 255.0) ** (1.0 / gamma) * 255.0)


def color_correct_channel(raw: int, max_brightness: int, local_brightness: int, gamma: float = 1.0) -> int:
    """Mirrors ESPColorCorrection::color_correct_*."""
    res = esp_scale8_twice(raw, max_brightness, local_brightness)
    return gamma_correct(res, gamma)


def color_uncorrect_channel(stored: int, max_brightness: int, local_brightness: int, gamma: float = 1.0) -> int:
    """Mirrors ESPColorCorrection::color_uncorrect_channel_."""
    if max_brightness == 0 or local_brightness == 0:
        return 0
    uncorrected = gamma_uncorrect(stored, gamma) * 255
    res = (uncorrected // max_brightness) * 255 // local_brightness
    return min(res, 255)


def subtract_scaled_difference(a: int, b: int, scale: int) -> int:
    """Mirrors the C++ subtract_scaled_difference helper."""
    result = int(a) - (((int(a) - int(b)) * scale) // 256)
    return max(0, min(255, result))


def smoothed_progress(x: float) -> float:
    """Matches LightTransformer::smoothed_progress: 6x^5 - 15x^4 + 10x^3."""
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeLED:
    """Minimal simulation of a single LED in the hardware buffer."""
    def __init__(self, max_r: int = 255, max_g: int = 255, max_b: int = 255,
                 max_w: int = 255, gamma: float = 1.0) -> None:
        self._max = (max_r, max_g, max_b, max_w)
        self._gamma = gamma
        self._local_brightness = 255
        # Raw stored bytes (post-correction).
        self.raw_r: int = 0
        self.raw_g: int = 0
        self.raw_b: int = 0
        self.raw_w: int = 0

    def set_local_brightness(self, lb: int) -> None:
        self._local_brightness = lb

    # --- corrected write (like ESPColorView::set_*) ---
    def set_r(self, raw_input: int) -> None:
        self.raw_r = color_correct_channel(raw_input, self._max[0], self._local_brightness, self._gamma)

    def set_g(self, raw_input: int) -> None:
        self.raw_g = color_correct_channel(raw_input, self._max[1], self._local_brightness, self._gamma)

    def set_b(self, raw_input: int) -> None:
        self.raw_b = color_correct_channel(raw_input, self._max[2], self._local_brightness, self._gamma)

    def set_rgb(self, r: int, g: int, b: int) -> None:
        self.set_r(r); self.set_g(g); self.set_b(b)

    # --- uncorrected read (like ESPColorView::get_*) ---
    def get_r(self) -> int:
        return color_uncorrect_channel(self.raw_r, self._max[0], self._local_brightness, self._gamma)

    def get_g(self) -> int:
        return color_uncorrect_channel(self.raw_g, self._max[1], self._local_brightness, self._gamma)

    def get_b(self) -> int:
        return color_uncorrect_channel(self.raw_b, self._max[2], self._local_brightness, self._gamma)

    # --- raw read (like ESPColorView::get_*_raw()) ---
    def get_r_raw(self) -> int:
        return self.raw_r

    def get_g_raw(self) -> int:
        return self.raw_g

    def get_b_raw(self) -> int:
        return self.raw_b

    # --- raw write (like ESPColorView::set_*_raw() — the new methods) ---
    def set_r_raw(self, v: int) -> None:
        self.raw_r = v

    def set_g_raw(self, v: int) -> None:
        self.raw_g = v

    def set_b_raw(self, v: int) -> None:
        self.raw_b = v

    def set_rgb_raw(self, r: int, g: int, b: int) -> None:
        self.raw_r = r; self.raw_g = g; self.raw_b = b


def simulate_stable_state(led: FakeLED, color_b: int = 255, brightness: float = 1.0) -> None:
    """Write the stable state using the same path as AddressableLight::update_state()."""
    local_brightness = round(brightness * 255)
    led.set_local_brightness(local_brightness)
    led.set_b(color_b)


def corrected_target_for_transition(
    target_b: int,
    end_brightness: float,
    max_b: int,
    gamma: float = 1.0,
) -> int:
    """Compute corrected_target_color_.blue the same way start() does."""
    # target_color_.blue = target_b (from color_from_light_color_values)
    # target_color_ *= to_uint8_scale(brightness * 1.0)
    local_brightness = 255  # set_local_brightness(255)
    brightness_scale = round(end_brightness * 255)
    target_scaled = esp_scale8(target_b, brightness_scale)
    # corrected_target = color_correct(target_scaled) with local_brightness=255
    return color_correct_channel(target_scaled, max_b, local_brightness, gamma)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma", [1.0, 2.8])
def test_no_flash_same_target_full_brightness(gamma: float) -> None:
    """
    White → White (same target), full brightness.
    The very first frame of the transition must not change any raw byte.
    """
    max_b = 178  # 70% of 255

    led = FakeLED(max_b=max_b, gamma=gamma)
    simulate_stable_state(led, color_b=255, brightness=1.0)
    stable_raw_b = led.raw_b

    # --- Transition start ---
    led.set_local_brightness(255)
    corrected_target_b = corrected_target_for_transition(255, 1.0, max_b, gamma)

    # Raw start (uniform_start_raw_) — read before any frame is written.
    uniform_start_raw_b = led.get_b_raw()

    # First frame (progress just above 0): remaining ≈ 256 → should write start unchanged.
    sp = smoothed_progress(0.001)
    remaining = int(256.0 * (1.0 - sp))
    first_frame_b = subtract_scaled_difference(corrected_target_b, uniform_start_raw_b, remaining)
    led.set_b_raw(first_frame_b)

    # Key assertion: first frame must not dip below the stable value.
    assert led.raw_b >= stable_raw_b - 1, (
        f"First frame dipped from {stable_raw_b} to {led.raw_b} "
        f"(gamma={gamma}, max_b={max_b}) — double-correction flash!"
    )


@pytest.mark.parametrize("gamma", [1.0, 2.8])
def test_no_flash_same_target_half_brightness(gamma: float) -> None:
    """
    White → White at 50% brightness.
    The raw byte at the start of the transition must equal the stable value.
    """
    max_b = 178  # 70% of 255

    led = FakeLED(max_b=max_b, gamma=gamma)
    simulate_stable_state(led, color_b=255, brightness=0.5)
    stable_raw_b = led.raw_b

    # --- Transition start ---
    led.set_local_brightness(255)
    corrected_target_b = corrected_target_for_transition(255, 0.5, max_b, gamma)

    uniform_start_raw_b = led.get_b_raw()

    sp = smoothed_progress(0.001)
    remaining = int(256.0 * (1.0 - sp))
    first_frame_b = subtract_scaled_difference(corrected_target_b, uniform_start_raw_b, remaining)

    # First frame should stay near the stable value.
    assert abs(first_frame_b - stable_raw_b) <= 1, (
        f"First frame changed from {stable_raw_b} to {first_frame_b} "
        f"at 50% brightness (gamma={gamma}) — unexpected jump!"
    )


@pytest.mark.parametrize("gamma", [1.0, 2.8])
def test_transition_reaches_correct_target(gamma: float) -> None:
    """
    The final frame of the transition must reach corrected_target_color_ exactly.
    """
    max_b = 178

    led = FakeLED(max_b=max_b, gamma=gamma)
    simulate_stable_state(led, color_b=255, brightness=1.0)  # stable white

    # Transition to half-brightness blue.
    led.set_local_brightness(255)
    corrected_target_b = corrected_target_for_transition(255, 0.5, max_b, gamma)

    # Simulate full transition in 50 frames.
    last_sp = 0.0
    for frame in range(1, 51):
        t = frame / 50.0
        sp = smoothed_progress(t)
        if sp > last_sp and last_sp < 1.0:
            remaining = int(256.0 * (1.0 - sp))
            led.raw_b = subtract_scaled_difference(corrected_target_b, led.raw_b, remaining)
            last_sp = sp

    assert led.raw_b == corrected_target_b, (
        f"Final blue byte {led.raw_b} != target {corrected_target_b} "
        f"(gamma={gamma}) — transition didn't reach target!"
    )


def test_no_flash_interrupted_transition() -> None:
    """
    Interrupted transition: start white→off at 30% progress, then interrupt with white→white.
    The first frame of the second transition must not dip below the current hardware value.
    """
    max_b = 178  # 70%
    gamma = 1.0

    led = FakeLED(max_b=max_b, gamma=gamma)
    simulate_stable_state(led, color_b=255, brightness=1.0)

    # First transition: white → off (target brightness=0, so corrected_target_b=0).
    led.set_local_brightness(255)
    corrected_target_off_b = corrected_target_for_transition(255, 0.0, max_b, gamma)  # = 0
    assert corrected_target_off_b == 0

    last_sp = 0.0
    # Run to ~30% progress.
    for frame in range(1, 31):
        t = frame / 100.0
        sp = smoothed_progress(t)
        if sp > last_sp and last_sp < 1.0:
            remaining = int(256.0 * (1.0 - sp))
            led.raw_b = subtract_scaled_difference(corrected_target_off_b, led.raw_b, remaining)
            last_sp = sp

    hardware_b_at_interrupt = led.raw_b
    assert hardware_b_at_interrupt < 178, "test setup: should be mid-transition"

    # Interrupt with second transition: → white at full brightness.
    # start() is called again: local_brightness stays 255.
    corrected_target_white_b = corrected_target_for_transition(255, 1.0, max_b, gamma)
    assert corrected_target_white_b == 178

    uniform_start_raw_b = led.raw_b  # raw read, no uncorrect roundtrip

    # First frame of the second transition.
    sp2 = smoothed_progress(0.001)
    remaining2 = int(256.0 * (1.0 - sp2))
    first_frame_b = subtract_scaled_difference(corrected_target_white_b, uniform_start_raw_b, remaining2)

    assert first_frame_b >= hardware_b_at_interrupt - 1, (
        f"First frame of interrupted transition dipped from {hardware_b_at_interrupt} "
        f"to {first_frame_b} — flash at interrupt boundary!"
    )


def test_non_uniform_path_no_flash() -> None:
    """
    Non-uniform path (per-LED delta): the first frame must not dip below prior hardware bytes.
    """
    max_b = 178
    gamma = 1.0

    led = FakeLED(max_b=max_b, gamma=gamma)
    simulate_stable_state(led, color_b=255, brightness=1.0)
    stable_raw_b = led.raw_b

    led.set_local_brightness(255)
    corrected_target_b = corrected_target_for_transition(255, 1.0, max_b, gamma)

    # First frame using non-uniform (per-LED delta) formula.
    last_progress = 0.0
    first_progress = smoothed_progress(0.02)
    scale = int(256.0 * max((1.0 - first_progress) / max(1.0 - last_progress, 1e-9), 0.0))

    first_frame_b = subtract_scaled_difference(corrected_target_b, led.get_b_raw(), scale)

    assert first_frame_b >= stable_raw_b - 1, (
        f"Non-uniform path: first frame dipped from {stable_raw_b} to {first_frame_b}"
    )


@pytest.mark.parametrize("max_b,end_brightness", [
    (255, 1.0),   # uniform correction, full brightness
    (178, 1.0),   # non-uniform (70%) correction, full brightness
    (178, 0.5),   # non-uniform correction, half brightness
    (128, 0.75),  # 50% correction, 75% brightness
])
def test_first_frame_does_not_exceed_start(max_b: int, end_brightness: float) -> None:
    """
    For a same-target transition (start state == end state), the first frame raw
    byte must equal the stable raw byte (within ±1 for integer rounding).
    """
    gamma = 1.0
    led = FakeLED(max_b=max_b, gamma=gamma)
    simulate_stable_state(led, color_b=255, brightness=end_brightness)
    stable_raw_b = led.raw_b

    led.set_local_brightness(255)
    corrected_target_b = corrected_target_for_transition(255, end_brightness, max_b, gamma)

    # Uniform path first frame.
    sp = smoothed_progress(0.001)
    remaining = int(256.0 * (1.0 - sp))
    first_b = subtract_scaled_difference(corrected_target_b, led.get_b_raw(), remaining)

    assert abs(first_b - stable_raw_b) <= 1, (
        f"First frame {first_b} differs from stable {stable_raw_b} by more than 1 "
        f"(max_b={max_b}, end_brightness={end_brightness})"
    )

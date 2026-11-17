"""Tajima .DST binary embroidery file writer."""

from typing import List, Tuple
from core.geometry import mm_to_units

Point2D = Tuple[float, float]


def _encode_dst_record(dx: int, dy: int, jump: bool = False) -> bytes:
    """
    Encode a single relative stitch movement (-121..+121 in 0.1mm units)
    into a 3-byte Tajima DST record.
    Note: DST uses a balanced ternary encoding per coordinate (-81, -27, -9, -3, -1, +1, +3, +9, +27, +81).
    """
    # Invert Y for machine coordinate convention if needed; keep standard cartesian mapping here
    b0 = 0
    b1 = 0
    b2 = 0b00000011  # End bits always set on byte 2

    if jump:
        b2 |= 0b10000000

    # Encode X
    if dx >= 41:
        b2 |= 0b00000100
        dx -= 81
    if dx <= -41:
        b2 |= 0b00001000
        dx += 81
    if dx >= 14:
        b1 |= 0b00000100
        dx -= 27
    if dx <= -14:
        b1 |= 0b00001000
        dx += 27
    if dx >= 5:
        b0 |= 0b00000100
        dx -= 9
    if dx <= -5:
        b0 |= 0b00001000
        dx += 9
    if dx >= 2:
        b1 |= 0b00000001
        dx -= 3
    if dx <= -2:
        b1 |= 0b00000010
        dx += 3
    if dx >= 1:
        b0 |= 0b00000001
        dx -= 1
    if dx <= -1:
        b0 |= 0b00000010
        dx += 1

    # Encode Y
    if dy >= 41:
        b2 |= 0b00100000
        dy -= 81
    if dy <= -41:
        b2 |= 0b00010000
        dy += 81
    if dy >= 14:
        b1 |= 0b00100000
        dy -= 27
    if dy <= -14:
        b1 |= 0b00010000
        dy += 27
    if dy >= 5:
        b0 |= 0b00100000
        dy -= 9
    if dy <= -5:
        b0 |= 0b00010000
        dy += 9
    if dy >= 2:
        b1 |= 0b10000000
        dy -= 3
    if dy <= -2:
        b1 |= 0b01000000
        dy += 3
    if dy >= 1:
        b0 |= 0b10000000
        dy -= 1
    if dy <= -1:
        b0 |= 0b01000000
        dy += 1

    return bytes([b0, b1, b2])


def write_dst(filepath: str, stitches_mm: List[Point2D], label: str = "DESIGN") -> None:
    """Write stitch coordinates (in millimeters) to a valid Tajima .DST binary file."""
    header_text = (
        f"LA:{label[:16]:<16}\r"
        f"ST:{len(stitches_mm):7d}\r"
        f"CO:  1\r"
        f"+X:    500\r"
        f"-X:    500\r"
        f"+Y:    500\r"
        f"-Y:    500\r"
        f"AX:+    0\r"
        f"AY:+    0\r"
    ).encode("ascii")

    header = bytearray(512)
    header[: len(header_text)] = header_text
    for i in range(len(header_text), 511):
        header[i] = 0x20  # Space padding
    header[511] = 0x1A  # EOF marker for DST header

    body = bytearray()
    prev_x = 0
    prev_y = 0

    for x_mm, y_mm in stitches_mm:
        curr_x = mm_to_units(x_mm)
        curr_y = mm_to_units(y_mm)
        dx = curr_x - prev_x
        dy = curr_y - prev_y

        # Break long jumps into max +/-121 unit steps
        while abs(dx) > 121 or abs(dy) > 121:
            step_x = max(-121, min(121, dx))
            step_y = max(-121, min(121, dy))
            body.extend(_encode_dst_record(step_x, step_y, jump=True))
            dx -= step_x
            dy -= step_y

        body.extend(_encode_dst_record(dx, dy, jump=False))
        prev_x = curr_x
        prev_y = curr_y

    # End of pattern record
    body.extend(bytes([0x00, 0x00, 0xF3]))

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(body)

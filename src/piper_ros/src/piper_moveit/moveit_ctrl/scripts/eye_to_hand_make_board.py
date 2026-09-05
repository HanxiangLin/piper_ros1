#!/usr/bin/env python3
"""Generate a metric A4 ArUco GridBoard PDF, PNG preview, and board.json."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

# Also usable directly from an unbuilt source checkout, without ROS.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from piper_eye_to_hand.board import Board


POINTS_PER_METRE = 72.0 / 0.0254
PAGE_WIDTH_M = 0.210
PAGE_HEIGHT_M = 0.297
BOARD_TOP_M = 0.050


def make_pdf(board):
    """Create an uncompressed vector PDF; all geometry is specified in metres."""
    if board.width > 0.190 + 1e-9 or board.height > 0.200 + 1e-9:
        raise ValueError("Board does not fit the A4 print area (190 x 200 mm)")
    scale = POINTS_PER_METRE
    x_origin = (PAGE_WIDTH_M - board.width) / 2.0
    y_top = PAGE_HEIGHT_M - BOARD_TOP_M
    commands = ["0 g", "0 G"]

    def text_at(x, y, size, message):
        message = message.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands.append("BT /F1 {} Tf {:.6f} {:.6f} Td ({}) Tj ET".format(
            size, x * scale, y * scale, message))

    text_at(0.015, 0.280, 14, "Piper eye-to-hand calibration / ArUco GridBoard")
    text_at(0.015, 0.271, 10, "Print at 100% / Actual size. Disable Fit to page.")
    text_at(0.015, 0.263, 9, "{}; IDs {}..{}; {} columns x {} rows".format(
        board.config["dictionary"], board.first_id, board.first_id + board.count - 1,
        board.config["markers_x"], board.config["markers_y"]))
    text_at(0.015, 0.256, 9, "Marker {:.3f} mm; gap {:.3f} mm; board {:.3f} x {:.3f} mm".format(
        board.config["marker_length"] * 1000, board.config["separation"] * 1000,
        board.width * 1000, board.height * 1000))
    cells = board.dictionary.markerSize + 2
    cell_length = board.config["marker_length"] / cells
    for marker_id in range(board.first_id, board.first_id + board.count):
        marker = board.marker_image(marker_id, cells)
        corner = board.marker_corners(marker_id)[0]
        for row in range(cells):
            for column in range(cells):
                if marker[row, column] != 0:
                    continue
                x = x_origin + corner[0] + column * cell_length
                y = y_top - corner[1] - (row + 1) * cell_length
                commands.append("{:.6f} {:.6f} {:.6f} {:.6f} re f".format(
                    x * scale, y * scale, cell_length * scale, cell_length * scale))
    # A labelled, physically 100 mm ruler independent of raster preview scaling.
    ruler_x, ruler_y = 0.055, 0.029
    commands.extend(["0.5 w", "{:.6f} {:.6f} m {:.6f} {:.6f} l S".format(
        ruler_x * scale, ruler_y * scale, (ruler_x + 0.100) * scale, ruler_y * scale)])
    for tick in range(11):
        x = (ruler_x + tick * 0.010) * scale
        tick_height = 0.003 if tick in (0, 5, 10) else 0.002
        commands.append("{:.6f} {:.6f} m {:.6f} {:.6f} l S".format(
            x, ruler_y * scale, x, (ruler_y + tick_height) * scale))
    text_at(0.054, 0.023, 9, "0")
    text_at(0.082, 0.023, 9, "Verify this ruler = 100 mm")
    text_at(0.152, 0.023, 9, "100")
    text_at(0.015, 0.013, 9, "Mount flat on a rigid backing. Measure marker size after printing.")
    content = ("\n".join(commands) + "\n").encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {:.6f} {:.6f}] "
         "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".format(
             PAGE_WIDTH_M * scale, PAGE_HEIGHT_M * scale)).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"endstream",
    ]
    result = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend("{} 0 obj\n".format(number).encode("ascii") + obj + b"\nendobj\n")
    start_xref = len(result)
    result.extend("xref\n0 {}\n0000000000 65535 f \n".format(len(offsets)).encode("ascii"))
    for offset in offsets[1:]:
        result.extend("{:010d} 00000 n \n".format(offset).encode("ascii"))
    result.extend(("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n".format(
        len(offsets), start_xref)).encode("ascii"))
    return bytes(result)


def make_preview(board):
    ppm = 150.0 / 0.0254
    page = np.full((int(round(PAGE_HEIGHT_M * ppm)), int(round(PAGE_WIDTH_M * ppm))),
                   255, dtype=np.uint8)
    board_image = board.render(pixels_per_metre=ppm, margin=0)
    x = (page.shape[1] - board_image.shape[1]) // 2
    y = int(round(BOARD_TOP_M * ppm))
    page[y:y + board_image.shape[0], x:x + board_image.shape[1]] = board_image
    cv2.putText(page, "ArUco GridBoard - PREVIEW ONLY", (50, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.putText(page, "Print board.pdf at 100% / Actual size", (50, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, 0, 2)
    ruler_y = int(round((PAGE_HEIGHT_M - 0.029) * ppm))
    cv2.line(page, (int(round(0.055 * ppm)), ruler_y),
             (int(round(0.155 * ppm)), ruler_y), 0, 2)
    cv2.putText(page, "PDF ruler: 100 mm", (int(round(0.055 * ppm)), ruler_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, 0, 2)
    success, encoded = cv2.imencode(".png", page)
    if not success:
        raise RuntimeError("Could not encode board PNG preview")
    return encoded.tobytes()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--markers-x", type=int, default=4)
    parser.add_argument("--markers-y", type=int, default=3)
    parser.add_argument("--marker-length", type=float, default=0.035, help="Black marker outer side, metres")
    parser.add_argument("--separation", type=float, default=0.008, help="White gap between markers, metres")
    parser.add_argument("--first-id", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        board = Board({"dictionary": args.dictionary, "markers_x": args.markers_x,
                       "markers_y": args.markers_y, "marker_length": args.marker_length,
                       "separation": args.separation, "first_id": args.first_id})
        destination = args.output_dir.expanduser().resolve()
        paths = [destination / name for name in ("board.pdf", "board.png", "board.json")]
        for path in paths:
            if path.exists() or path.is_symlink():
                raise ValueError("Refusing to overwrite {}".format(path))
        # Build all outputs before creating files so invalid dimensions leave no partial board.
        contents = [make_pdf(board), make_preview(board),
                    (json.dumps(board.config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")]
        destination.mkdir(parents=True, exist_ok=True)
        for path, content in zip(paths, contents):
            with path.open("xb") as stream:
                stream.write(content)
        for path in paths:
            print(path)
        print("Print the PDF at 100%, then measure the 100 mm ruler and {:.3f} mm markers.".format(
            board.config["marker_length"] * 1000))
        return 0
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        print("Board generation failed: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

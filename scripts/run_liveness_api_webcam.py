import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


API_BASE_URL = os.getenv(
    "FACE_API_BASE_URL",
    "http://localhost:8080",
)

API_KEY = os.getenv("FACE_API_KEY")

FRAME_INTERVAL_SECONDS = 0.15
REQUEST_TIMEOUT_SECONDS = 15


def draw_text(
    image,
    text: str,
    line_number: int,
) -> None:
    y_position = 35 + line_number * 32

    cv2.putText(
        image,
        text,
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def require_api_key() -> str:
    if not API_KEY:
        raise RuntimeError(
            "FACE_API_KEY is not set. "
            "Export it before running this script."
        )

    return API_KEY


def create_session(
    headers: dict[str, str],
) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/liveness/sessions",
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    return response.json()


def send_frame(
    *,
    session_id: str,
    frame,
    headers: dict[str, str],
) -> dict[str, Any]:
    encoded, buffer = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            90,
        ],
    )

    if not encoded:
        raise RuntimeError(
            "Could not encode webcam frame."
        )

    files = {
        "image": (
            "frame.jpg",
            buffer.tobytes(),
            "image/jpeg",
        )
    }

    data = {
        "threshold": "80.0",
    }

    response = requests.post(
        (
            f"{API_BASE_URL}/liveness/sessions/"
            f"{session_id}/frames"
        ),
        headers=headers,
        files=files,
        data=data,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"Frame request failed "
            f"({response.status_code}): {detail}"
        )

    return response.json()


def get_results(
    *,
    session_id: str,
    headers: dict[str, str],
) -> dict[str, Any]:
    response = requests.get(
        (
            f"{API_BASE_URL}/liveness/sessions/"
            f"{session_id}/results"
        ),
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    return response.json()


def main() -> None:
    api_key = require_api_key()

    headers = {
        "x-api-key": api_key,
    }

    session = create_session(headers)

    session_id = session["sessionId"]

    print("Session:", session_id)
    print(
        "Initial instruction:",
        session.get(
            "instruction",
            "Look at the camera with your eyes open",
        ),
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the Mac camera."
        )

    latest_result = session
    last_sent_at = 0.0
    terminal_at: float | None = None

    try:
        while True:
            success, frame = camera.read()

            if not success:
                raise RuntimeError(
                    "Could not read webcam frame."
                )

            display_frame = frame.copy()

            now = time.monotonic()

            should_send = (
                terminal_at is None
                and now - last_sent_at
                >= FRAME_INTERVAL_SECONDS
            )

            if should_send:
                last_sent_at = now

                try:
                    latest_result = send_frame(
                        session_id=session_id,
                        frame=frame,
                        headers=headers,
                    )

                    print(
                        latest_result["status"],
                        latest_result["challengeStage"],
                        latest_result["challengeProgress"],
                        latest_result["instruction"],
                    )

                except RuntimeError as error:
                    print(error)

            instruction = latest_result.get(
                "instruction",
                "Preparing challenge",
            )

            progress = latest_result.get(
                "challengeProgress",
                0,
            )

            status = latest_result.get(
                "status",
                "CREATED",
            )

            stage = latest_result.get(
                "challengeStage",
                "WAITING",
            )

            draw_text(
                display_frame,
                instruction,
                0,
            )

            draw_text(
                display_frame,
                f"Progress: {progress}%",
                1,
            )

            draw_text(
                display_frame,
                f"Status: {status}",
                2,
            )

            draw_text(
                display_frame,
                f"Stage: {stage}",
                3,
            )

            if status in {
                "SUCCEEDED",
                "FAILED",
                "EXPIRED",
            }:
                if terminal_at is None:
                    terminal_at = time.monotonic()

                if status == "SUCCEEDED":
                    draw_text(
                        display_frame,
                        "ACTIVE LIVENESS PASSED",
                        5,
                    )
                else:
                    draw_text(
                        display_frame,
                        f"ACTIVE LIVENESS {status}",
                        5,
                    )

                if (
                    time.monotonic() - terminal_at
                    >= 2.0
                ):
                    break

            cv2.imshow(
                "Liveness API Challenge",
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in {
                ord("q"),
                ord("Q"),
            }:
                print("Cancelled.")
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    final_result = get_results(
        session_id=session_id,
        headers=headers,
    )

    print("\nFinal result:")

    for key, value in final_result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
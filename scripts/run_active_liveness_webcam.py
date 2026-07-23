import time

import cv2

from app.active_liveness_challenge import (
    ActiveLivenessChallenge,
)
from app.active_liveness_engine import (
    ActiveLivenessEngine,
)


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
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the Mac camera."
        )

    active_engine = ActiveLivenessEngine()

    challenge = ActiveLivenessChallenge(
        timeout_seconds=20.0,
    )

    passed_at: float | None = None
    failed_at: float | None = None

    print("Complete the instructions shown on screen.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                raise RuntimeError(
                    "Could not read a camera frame."
                )

            display_frame = frame.copy()

            try:
                signal = active_engine.analyze_image(
                    frame
                )

                challenge_result = challenge.update(
                    signal
                )

                draw_text(
                    display_frame,
                    challenge_result["instruction"],
                    0,
                )

                draw_text(
                    display_frame,
                    (
                        "Progress: "
                        f"{challenge_result['progress']}%"
                    ),
                    1,
                )

                draw_text(
                    display_frame,
                    (
                        "Blink score: "
                        f"{signal['blinkScore']:.4f}"
                    ),
                    2,
                )

                draw_text(
                    display_frame,
                    (
                        "Yaw: "
                        f"{signal['yawDegrees']:.2f}"
                    ),
                    3,
                )

                draw_text(
                    display_frame,
                    (
                        "Stage: "
                        f"{challenge_result['status']}"
                    ),
                    4,
                )

                if challenge_result["passed"]:
                    if passed_at is None:
                        passed_at = time.monotonic()

                    draw_text(
                        display_frame,
                        "ACTIVE LIVENESS PASSED",
                        6,
                    )

                    if (
                        time.monotonic() - passed_at
                        >= 2.0
                    ):
                        break

                elif challenge_result["failed"]:
                    if failed_at is None:
                        failed_at = time.monotonic()

                    draw_text(
                        display_frame,
                        "ACTIVE LIVENESS FAILED",
                        6,
                    )

                    if (
                        time.monotonic() - failed_at
                        >= 2.0
                    ):
                        break

            except ValueError as error:
                draw_text(
                    display_frame,
                    str(error),
                    0,
                )

                draw_text(
                    display_frame,
                    "Keep exactly one face visible",
                    1,
                )

            cv2.imshow(
                "Active Liveness Challenge",
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Cancelled.")
                break

    finally:
        active_engine.close()
        camera.release()
        cv2.destroyAllWindows()

    if challenge.stage == "PASSED":
        print("Active liveness challenge passed.")
    elif challenge.stage == "FAILED":
        print("Active liveness challenge failed.")
    else:
        print(
            "Active liveness challenge stopped before completion."
        )


if __name__ == "__main__":
    main()
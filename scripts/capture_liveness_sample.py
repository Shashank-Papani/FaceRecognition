import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a private liveness sample "
            "from the Mac webcam."
        )
    )

    parser.add_argument(
        "output_path",
        help="Where the captured JPG should be saved.",
    )

    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the Mac camera."
        )

    print("Press SPACE to capture.")
    print("Press Q to cancel.")
    print("Output:", output_path)

    try:
        while True:
            success, frame = camera.read()

            if not success:
                raise RuntimeError(
                    "Could not read a camera frame."
                )

            cv2.imshow(
                "Liveness sample capture",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                saved = cv2.imwrite(
                    str(output_path),
                    frame,
                )

                if not saved:
                    raise RuntimeError(
                        "The image could not be saved."
                    )

                print("Saved:", output_path)
                break

            if key == ord("q"):
                print("Cancelled.")
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
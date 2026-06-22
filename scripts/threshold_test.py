import itertools
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.face_engine import FaceEngine


engine = FaceEngine()

PEOPLE_DIR = PROJECT_ROOT / "threshold_tests" / "people"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_image_paths(person_folder: Path) -> list[Path]:
    return sorted(
        path
        for path in person_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    )


def load_people_images() -> dict[str, list[Path]]:
    if not PEOPLE_DIR.exists():
        raise FileNotFoundError(
            "Missing threshold_tests/people folder. "
            "Create threshold_tests/people/<person_name>/ "
            "and add images."
        )

    people = {}

    for person_folder in PEOPLE_DIR.iterdir():
        if not person_folder.is_dir():
            continue

        images = get_image_paths(person_folder)

        if images:
            people[person_folder.name] = images

    if len(people) < 2:
        raise ValueError(
            "Add images for at least two different people inside "
            "threshold_tests/people/."
        )

    return people


def get_embedding(image_path: Path) -> np.ndarray:
    embedding = engine.get_embedding(str(image_path))

    return np.asarray(embedding, dtype=np.float32).reshape(-1)


def cosine_similarity(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray,
) -> float:
    denominator = (
        np.linalg.norm(embedding_1)
        * np.linalg.norm(embedding_2)
    )

    if denominator == 0:
        raise ValueError("Cannot compare a zero-length embedding.")

    return float(
        np.dot(embedding_1, embedding_2) / denominator
    )


def compute_embeddings(
    people: dict[str, list[Path]],
) -> dict[str, list[dict]]:
    embeddings = {}

    print("\nLOADING EMBEDDINGS")
    print("--------------------")

    for person_name, image_paths in people.items():
        embeddings[person_name] = []

        for image_path in image_paths:
            try:
                embedding = get_embedding(image_path)
            except Exception as error:
                print(
                    f"Skipped {person_name}/{image_path.name}: "
                    f"{error}"
                )
                continue

            embeddings[person_name].append(
                {
                    "image_path": image_path,
                    "embedding": embedding,
                }
            )

            print(
                f"Loaded {person_name}: {image_path.name}"
            )

    return embeddings


def evaluate_same_person(
    embeddings: dict[str, list[dict]],
) -> list[float]:
    print("\nSAME-PERSON TESTS")
    print("--------------------")

    scores = []

    for person_name, records in embeddings.items():
        if len(records) < 2:
            print(
                f"Skipping {person_name}: "
                "at least two valid images are required."
            )
            continue

        for record_1, record_2 in itertools.combinations(
            records,
            2,
        ):
            score = cosine_similarity(
                record_1["embedding"],
                record_2["embedding"],
            )

            scores.append(score)

            print(
                f"{person_name}: "
                f"{record_1['image_path'].name} vs "
                f"{record_2['image_path'].name}: "
                f"{score:.4f}"
            )

    return scores


def evaluate_different_people(
    embeddings: dict[str, list[dict]],
) -> list[float]:
    print("\nDIFFERENT-PEOPLE TESTS")
    print("--------------------")

    scores = []
    person_names = list(embeddings.keys())

    for person_1, person_2 in itertools.combinations(
        person_names,
        2,
    ):
        for record_1 in embeddings[person_1]:
            for record_2 in embeddings[person_2]:
                score = cosine_similarity(
                    record_1["embedding"],
                    record_2["embedding"],
                )

                scores.append(score)

                print(
                    f"{person_1}/{record_1['image_path'].name} vs "
                    f"{person_2}/{record_2['image_path'].name}: "
                    f"{score:.4f}"
                )

    return scores


def summarize_scores(
    label: str,
    scores: list[float],
) -> None:
    print(f"\n{label} SUMMARY")
    print("--------------------")

    if not scores:
        print("No scores found.")
        return

    print(f"count: {len(scores)}")
    print(f"min:   {min(scores):.4f}")
    print(f"max:   {max(scores):.4f}")
    print(f"avg:   {sum(scores) / len(scores):.4f}")


def evaluate_thresholds(
    same_scores: list[float],
    different_scores: list[float],
) -> None:
    print("\nTHRESHOLD EVALUATION")
    print("--------------------")

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
    ]

    for threshold in thresholds:
        false_rejects = sum(
            score < threshold
            for score in same_scores
        )

        false_accepts = sum(
            score >= threshold
            for score in different_scores
        )

        same_total = len(same_scores)
        different_total = len(different_scores)

        false_reject_rate = (
            false_rejects / same_total
            if same_total
            else 0
        )

        false_accept_rate = (
            false_accepts / different_total
            if different_total
            else 0
        )

        print(
            f"Threshold {threshold:.2f} "
            f"(API value {threshold * 100:.0f}) | "
            f"False Rejects: "
            f"{false_rejects}/{same_total} "
            f"({false_reject_rate:.2%}) | "
            f"False Accepts: "
            f"{false_accepts}/{different_total} "
            f"({false_accept_rate:.2%})"
        )


def recommend_threshold(
    same_scores: list[float],
    different_scores: list[float],
) -> None:
    print("\nRECOMMENDATION")
    print("--------------------")

    if not same_scores or not different_scores:
        print(
            "Not enough data to recommend a threshold."
        )
        return

    same_min = min(same_scores)
    different_max = max(different_scores)

    print(
        f"Lowest same-person score:       {same_min:.4f}"
    )
    print(
        f"Highest different-person score: {different_max:.4f}"
    )

    if different_max < same_min:
        suggested = (same_min + different_max) / 2

        print("Clean score separation found.")
        print(
            "Suggested threshold range: "
            f"{different_max:.4f} to {same_min:.4f}"
        )
        print(
            "Suggested midpoint threshold: "
            f"{suggested:.4f}"
        )
        print(
            "Suggested API threshold value: "
            f"{suggested * 100:.2f}"
        )
    else:
        print(
            "Same-person and different-person scores overlap."
        )
        print(
            "Add more people and images before selecting "
            "a production threshold."
        )


def main() -> None:
    people = load_people_images()
    embeddings = compute_embeddings(people)

    same_scores = evaluate_same_person(embeddings)
    different_scores = evaluate_different_people(
        embeddings
    )

    summarize_scores("SAME PERSON", same_scores)
    summarize_scores(
        "DIFFERENT PEOPLE",
        different_scores,
    )

    evaluate_thresholds(
        same_scores,
        different_scores,
    )

    recommend_threshold(
        same_scores,
        different_scores,
    )


if __name__ == "__main__":
    main()

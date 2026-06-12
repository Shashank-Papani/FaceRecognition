import itertools
from pathlib import Path
import numpy as np

from app.face_engine import FaceEngine


engine = FaceEngine()

PEOPLE_DIR = Path("threshold_tests/people")
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def get_image_paths(person_folder: Path):
    return [
        path for path in person_folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]


def load_people_images():
    if not PEOPLE_DIR.exists():
        raise FileNotFoundError(
            "Missing threshold_tests/people folder. "
            "Create threshold_tests/people/<person_name>/ and add images."
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
            "Add images for at least 2 different people inside threshold_tests/people/"
        )

    return people


def get_embedding(image_path: Path):
    return engine.get_embedding(str(image_path))


def cosine_similarity(embedding_1, embedding_2):
    return float(np.dot(embedding_1, embedding_2))


def compute_embeddings(people):
    embeddings = {}

    print("\nLOADING EMBEDDINGS")
    print("--------------------")

    for person_name, image_paths in people.items():
        embeddings[person_name] = []

        for image_path in image_paths:
            embedding = get_embedding(image_path)

            embeddings[person_name].append(
                {
                    "image_path": image_path,
                    "embedding": embedding
                }
            )

            print(f"Loaded {person_name}: {image_path.name}")

    return embeddings


def test_same_person(embeddings):
    print("\nSAME PERSON TESTS")
    print("--------------------")

    scores = []

    for person_name, records in embeddings.items():
        if len(records) < 2:
            print(f"Skipping {person_name}: needs at least 2 images for same-person test")
            continue

        for record_1, record_2 in itertools.combinations(records, 2):
            score = cosine_similarity(
                record_1["embedding"],
                record_2["embedding"]
            )

            scores.append(score)

            print(
                f"{person_name}: "
                f"{record_1['image_path'].name} vs {record_2['image_path'].name}: "
                f"{score:.4f}"
            )

    return scores


def test_different_people(embeddings):
    print("\nDIFFERENT PEOPLE TESTS")
    print("--------------------")

    scores = []

    person_names = list(embeddings.keys())

    for person_1, person_2 in itertools.combinations(person_names, 2):
        for record_1 in embeddings[person_1]:
            for record_2 in embeddings[person_2]:
                score = cosine_similarity(
                    record_1["embedding"],
                    record_2["embedding"]
                )

                scores.append(score)

                print(
                    f"{person_1}/{record_1['image_path'].name} vs "
                    f"{person_2}/{record_2['image_path'].name}: "
                    f"{score:.4f}"
                )

    return scores


def summarize_scores(label: str, scores: list[float]):
    print(f"\n{label} SUMMARY")
    print("--------------------")

    if not scores:
        print("No scores found")
        return

    print(f"count: {len(scores)}")
    print(f"min:   {min(scores):.4f}")
    print(f"max:   {max(scores):.4f}")
    print(f"avg:   {sum(scores) / len(scores):.4f}")


def evaluate_thresholds(same_scores: list[float], different_scores: list[float]):
    print("\nTHRESHOLD EVALUATION")
    print("--------------------")

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    for threshold in thresholds:
        false_rejects = sum(score < threshold for score in same_scores)
        false_accepts = sum(score >= threshold for score in different_scores)

        same_total = len(same_scores)
        different_total = len(different_scores)

        false_reject_rate = false_rejects / same_total if same_total else 0
        false_accept_rate = false_accepts / different_total if different_total else 0

        print(
            f"Threshold {threshold:.2f} | "
            f"False Rejects: {false_rejects}/{same_total} ({false_reject_rate:.2%}) | "
            f"False Accepts: {false_accepts}/{different_total} ({false_accept_rate:.2%})"
        )


def recommend_threshold(same_scores: list[float], different_scores: list[float]):
    print("\nRECOMMENDATION")
    print("--------------------")

    if not same_scores or not different_scores:
        print("Not enough data to recommend a threshold.")
        return

    same_min = min(same_scores)
    different_max = max(different_scores)

    print(f"Lowest same-person score:     {same_min:.4f}")
    print(f"Highest different-person score: {different_max:.4f}")

    if different_max < same_min:
        suggested = (same_min + different_max) / 2

        print(f"Clean separation found.")
        print(f"Suggested threshold range: {different_max:.4f} to {same_min:.4f}")
        print(f"Suggested midpoint threshold: {suggested:.4f}")

        if 0.65 <= suggested <= 0.75:
            print("0.70 is still a good practical default.")
        elif suggested < 0.65:
            print("Consider using a lower threshold like 0.60 or 0.65.")
        else:
            print("Consider using a stricter threshold like 0.75.")
    else:
        print("Scores overlap. Threshold tuning is risky with this dataset.")
        print("Add more images, improve image quality, or consider a stronger recognizer.")


if __name__ == "__main__":
    people = load_people_images()
    embeddings = compute_embeddings(people)

    same_scores = test_same_person(embeddings)
    different_scores = test_different_people(embeddings)

    summarize_scores("SAME PERSON", same_scores)
    summarize_scores("DIFFERENT PEOPLE", different_scores)

    evaluate_thresholds(same_scores, different_scores)
    recommend_threshold(same_scores, different_scores)
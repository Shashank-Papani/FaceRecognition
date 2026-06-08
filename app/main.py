from app.face_engine import FaceEngine

engine = FaceEngine()

result = engine.compare_faces(
    "test_images/person1.jpg",
    "test_images/person2.jpg"
)

print(result)
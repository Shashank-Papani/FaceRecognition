from app.face_engine import FaceEngine

engine = FaceEngine()

enroll_result = engine.enroll_face(
    "person_001",
    "test_images/person1.jpg"
)

print("Enroll result:")
print(enroll_result)

verify_result = engine.verify_face(
    "test_images/person2.jpg"
)

print("Verify result:")
print(verify_result)
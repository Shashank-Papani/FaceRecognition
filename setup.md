clone the repo and activate virtual env
    python -m venv .venv
    .venv\Scripts\activate

Install packages
    pip install opencv-python numpy fastapi uvicorn python-multipart

Correct loading of model files check:
    python -c "import cv2; print(hasattr(cv2, 'FaceDetectorYN')); print(hasattr(cv2, 'FaceRecognizerSF'))"

Add two images of yourself and other one of you and other in test_images and run:
    python -m app.main
        example output: {'similarity': 0.71, 'match': True}
    
    **Make sure the images you add are jpg and if there's any errors
    ** If you get error: frozen runpy, result = engine.compare_faces change score_threshold to lower for the start in face_engine.py

To run API endpoints:
    run, uvicorn app.main:app --reload

for API endpoint checks, use swagger
    http://127.0.0.1:8000/docs

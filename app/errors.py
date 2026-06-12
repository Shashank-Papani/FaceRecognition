from fastapi import HTTPException

ERROR_CODES = {
    "No face detected": "NO_FACE_DETECTED",
    "No valid face detected": "LOW_QUALITY_FACE",
    "Multiple faces detected": "MULTIPLE_FACES_DETECTED",
    "Could not read image": "INVALID_IMAGE",
    "Uploaded image is empty": "EMPTY_IMAGE",
    "Only JPG, JPEG, and PNG": "UNSUPPORTED_IMAGE_TYPE",
    "Invalid or missing API key": "UNAUTHORIZED",
}

def get_error_code(message: str) -> str:
    for key, code in ERROR_CODES.items():
        if key in message:
            return code
        
    return "INTERNAL_ERROR"

def raise_api_error(error: Exception, status_code:int = 400):
    message = str(error)
    error_code = get_error_code(message)

    raise HTTPException(\
        status_code=status_code,
        detail={
            "success": False,
            "error_code": error_code,
            "message": message
        }
    )
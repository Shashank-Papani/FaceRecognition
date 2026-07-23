"use strict";

const camera = document.getElementById("camera");
const frameCanvas = document.getElementById("frameCanvas");
const cameraPlaceholder = document.getElementById(
    "cameraPlaceholder"
);

const apiKeyInput = document.getElementById("apiKey");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");

const statusText = document.getElementById("status");
const instructionText = document.getElementById(
    "instruction"
);
const progressBar = document.getElementById("progress");
const progressText = document.getElementById(
    "progressText"
);
const errorMessage = document.getElementById(
    "errorMessage"
);

let mediaStream = null;
let sessionId = null;
let challengeRunning = false;


function updateProgress(value) {
    const safeValue = Math.max(
        0,
        Math.min(100, Number(value) || 0)
    );

    progressBar.value = safeValue;
    progressText.textContent = `${safeValue}%`;
}


function getErrorMessage(data, fallbackMessage) {
    if (
        data
        && data.detail
        && typeof data.detail === "object"
        && data.detail.message
    ) {
        return data.detail.message;
    }

    if (
        data
        && data.detail
        && typeof data.detail === "string"
    ) {
        return data.detail;
    }

    if (data && data.message) {
        return data.message;
    }

    return fallbackMessage;
}


async function readResponse(response) {
    let data = null;

    try {
        data = await response.json();
    } catch {
        data = null;
    }

    if (!response.ok) {
        throw new Error(
            getErrorMessage(
                data,
                `Request failed with status ${response.status}`
            )
        );
    }

    return data;
}


async function startCamera() {
    mediaStream =
        await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: {
                    ideal: 640
                },
                height: {
                    ideal: 480
                }
            },
            audio: false
        });

    camera.srcObject = mediaStream;

    await camera.play();

    camera.style.display = "block";
    cameraPlaceholder.style.display = "none";
}


function stopCamera() {
    if (mediaStream) {
        for (const track of mediaStream.getTracks()) {
            track.stop();
        }
    }

    mediaStream = null;
    camera.srcObject = null;
    camera.style.display = "none";
    cameraPlaceholder.style.display = "flex";
}


function stopChallenge(message = "Challenge stopped") {
    challengeRunning = false;

    stopCamera();

    startButton.disabled = false;
    stopButton.disabled = true;
    apiKeyInput.disabled = false;

    statusText.textContent = message;
}


async function createLivenessSession(apiKey) {
    const response = await fetch(
        "/liveness/sessions",
        {
            method: "POST",
            headers: {
                "x-api-key": apiKey
            }
        }
    );

    return readResponse(response);
}


async function captureFrameBlob() {
    if (
        !camera.videoWidth
        || !camera.videoHeight
    ) {
        throw new Error(
            "Camera is not ready to capture frames."
        );
    }

    frameCanvas.width = camera.videoWidth;
    frameCanvas.height = camera.videoHeight;

    const context = frameCanvas.getContext("2d");

    if (!context) {
        throw new Error(
            "Could not initialize the camera canvas."
        );
    }

    context.drawImage(
        camera,
        0,
        0,
        frameCanvas.width,
        frameCanvas.height
    );

    return new Promise((resolve, reject) => {
        frameCanvas.toBlob(
            (blob) => {
                if (!blob) {
                    reject(
                        new Error(
                            "Could not capture camera frame."
                        )
                    );

                    return;
                }

                resolve(blob);
            },
            "image/jpeg",
            0.85
        );
    });
}


async function uploadFrame(apiKey) {
    const frameBlob = await captureFrameBlob();

    const formData = new FormData();

    formData.append(
        "image",
        frameBlob,
        "frame.jpg"
    );

    formData.append(
        "threshold",
        "80"
    );

    const response = await fetch(
        `/liveness/sessions/${sessionId}/frames`,
        {
            method: "POST",
            headers: {
                "x-api-key": apiKey
            },
            body: formData
        }
    );

    return readResponse(response);
}


async function getFinalResult(apiKey) {
    const response = await fetch(
        `/liveness/sessions/${sessionId}/results`,
        {
            method: "GET",
            headers: {
                "x-api-key": apiKey
            }
        }
    );

    return readResponse(response);
}


function displayFrameResult(result) {
    statusText.textContent =
        result.status || "IN_PROGRESS";

    instructionText.textContent =
        result.instruction
        || "Follow the instructions";

    updateProgress(
        result.challengeProgress
    );
}


async function runFrameLoop(apiKey) {
    while (challengeRunning) {
        try {
            const result = await uploadFrame(apiKey);

            displayFrameResult(result);

            const passed =
                result.status === "SUCCEEDED"
                || result.challengeStage === "PASSED"
                || result.activeLivenessPassed === true;

            const ended =
                result.status === "FAILED"
                || result.status === "EXPIRED";

            if (passed) {
                const finalResult =
                    await getFinalResult(apiKey);

                instructionText.textContent =
                    "Liveness challenge passed";

                updateProgress(100);

                console.log(
                    "Final liveness result:",
                    finalResult
                );

                stopChallenge(
                    "SUCCEEDED — live person verified"
                );

                return;
            }

            if (ended) {
                instructionText.textContent =
                    result.instruction
                    || "Liveness challenge failed";

                stopChallenge(result.status);

                return;
            }
        } catch (error) {
            const message =
                error instanceof Error
                    ? error.message
                    : "Frame upload failed.";

            errorMessage.textContent = message;

            stopChallenge("Challenge stopped");

            return;
        }

        await new Promise((resolve) => {
            window.setTimeout(resolve, 250);
        });
    }
}


async function startChallenge() {
    const apiKey = apiKeyInput.value.trim();

    if (!apiKey) {
        errorMessage.textContent =
            "Enter the API key first.";

        return;
    }

    errorMessage.textContent = "";
    statusText.textContent = "Starting camera...";
    instructionText.textContent =
        "Allow camera access when prompted";

    updateProgress(0);

    startButton.disabled = true;
    stopButton.disabled = false;
    apiKeyInput.disabled = true;

    try {
        await startCamera();

        statusText.textContent =
            "Creating liveness session...";

        const session =
            await createLivenessSession(apiKey);

        sessionId = session.sessionId;

        instructionText.textContent =
            session.instruction
            || "Look at the camera";

        updateProgress(
            session.challengeProgress
        );

        statusText.textContent =
            session.status || "CREATED";

        challengeRunning = true;

        console.log(
            "Created liveness session:",
            sessionId
        );

        await runFrameLoop(apiKey);
    } catch (error) {
        const message =
            error instanceof Error
                ? error.message
                : "Could not start liveness challenge.";

        errorMessage.textContent = message;

        stopChallenge("Unable to start");
    }
}


startButton.addEventListener(
    "click",
    startChallenge
);


stopButton.addEventListener(
    "click",
    () => {
        stopChallenge("Challenge stopped by user");
    }
);


window.addEventListener(
    "beforeunload",
    () => {
        challengeRunning = false;
        stopCamera();
    }
);

console.log("Interactive liveness page loaded.");

const API_URL =
    "http://127.0.0.1:8000";


async function predictFace(imageBlob) {

    if (!imageBlob) {
        return null;
    }

    const formData = new FormData();

    formData.append(
        "file",
        imageBlob,
        "face.jpg"
    );

    formData.append(
        "platform",
        window.location.hostname
    );

    try {

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",
                body: formData
            }
        );

        if (!response.ok) {
            throw new Error(
                `API returned ${response.status}`
            );
        }

        const result =
            await response.json();

        return result;

    } catch (error) {

        console.error(
            "[API] Prediction failed:",
            error
        );

        return null;
    }
}
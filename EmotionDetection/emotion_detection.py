"""
Module for detecting emotions in text using Watson NLP.
"""

import requests

WATSON_EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)

HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def empty_emotion_response():
    """
    Return the standard response for invalid or blank input.
    """
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None
    }


def emotion_detector(text_to_analyze):
    """
    Analyze text and return emotion scores plus the dominant emotion.

    Args:
        text_to_analyze: Customer feedback text to analyze.

    Returns:
        A dictionary containing anger, disgust, fear, joy, sadness,
        and dominant_emotion values.
    """
    if not text_to_analyze or not text_to_analyze.strip():
        return empty_emotion_response()

    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    try:
        response = requests.post(
            WATSON_EMOTION_URL,
            json=payload,
            headers=HEADERS,
            timeout=10
        )
    except requests.RequestException:
        return empty_emotion_response()

    if response.status_code == 400:
        return empty_emotion_response()

    if response.status_code != 200:
        return empty_emotion_response()

    try:
        formatted_response = response.json()
        emotions = formatted_response["emotionPredictions"][0]["emotion"]

        anger = emotions["anger"]
        disgust = emotions["disgust"]
        fear = emotions["fear"]
        joy = emotions["joy"]
        sadness = emotions["sadness"]

        dominant_emotion = max(emotions, key=emotions.get)

        return {
            "anger": anger,
            "disgust": disgust,
            "fear": fear,
            "joy": joy,
            "sadness": sadness,
            "dominant_emotion": dominant_emotion
        }

    except (KeyError, IndexError, ValueError, TypeError):
        return empty_emotion_response()
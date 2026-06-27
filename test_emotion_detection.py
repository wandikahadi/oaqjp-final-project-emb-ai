"""
Unit tests for the emotion detection application.
"""

import unittest

from EmotionDetection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """
    Test suite for emotion_detector.
    """

    def test_joy(self):
        """
        Verify joy detection.
        """
        response = emotion_detector("I am glad this happened")
        self.assertEqual(response["dominant_emotion"], "joy")

    def test_anger(self):
        """
        Verify anger detection.
        """
        response = emotion_detector("I am really mad about this")
        self.assertEqual(response["dominant_emotion"], "anger")

    def test_disgust(self):
        """
        Verify disgust detection.
        """
        response = emotion_detector(
            "I feel disgusted just hearing about this"
        )
        self.assertEqual(response["dominant_emotion"], "disgust")

    def test_sadness(self):
        """
        Verify sadness detection.
        """
        response = emotion_detector("I am so sad about this")
        self.assertEqual(response["dominant_emotion"], "sadness")

    def test_fear(self):
        """
        Verify fear detection.
        """
        response = emotion_detector(
            "I am really afraid that this will happen"
        )
        self.assertEqual(response["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()
import unittest

from fastapi.testclient import TestClient

from backend.main import InputGuardrail, app, extract_video_id


class YouTubeUrlTests(unittest.TestCase):
    def test_extracts_supported_youtube_urls(self):
        video_id = "zjkBMFhNj_g"
        urls = [
            f"https://www.youtube.com/watch?v={video_id}",
            f"https://youtu.be/{video_id}?si=example",
            f"https://www.youtube.com/shorts/{video_id}",
            f"https://www.youtube.com/embed/{video_id}",
        ]

        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(extract_video_id(url), video_id)

    def test_rejects_non_youtube_and_malformed_urls(self):
        urls = [
            "https://example.com/watch?v=zjkBMFhNj_g",
            "https://youtube.com.evil.example/watch?v=zjkBMFhNj_g",
            "https://www.youtube.com/watch?v=too-short",
        ]

        for url in urls:
            with self.subTest(url=url):
                self.assertIsNone(extract_video_id(url))
                self.assertFalse(InputGuardrail.validate_url(url)[0])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_ingest_rejects_invalid_url_before_model_loading(self):
        response = self.client.post("/ingest", json={"url": "https://example.com/video"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Enter a valid YouTube video URL.")


if __name__ == "__main__":
    unittest.main()

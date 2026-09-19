import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.sign_dictionary.dictionary_service import SignDictionary
from app.cwasa.adapter import CWASAAdapter

client = TestClient(app)


class TestGenericSignDictionaryAndCWASAPipeline:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.dict_service = SignDictionary()

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ui_dashboard_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "SignBridge" in response.text or "Speech to Sign Language" in response.text



    def test_real_known_sign_afraid(self):
        # Test lookup for real 'afraid' sign file
        res = self.dict_service.lookup_sign("afraid")
        assert res["found"] is True
        assert res["query"] == "afraid"
        assert res["matched_key"] == "afraid"
        assert res["status"] == "READY_FOR_CWASA"
        assert res["format"] == "sigml"
        assert "<sigml>" in res["sigml"]
        assert '<hns_sign gloss="afraid">' in res["sigml"]

        # Validate SiGML XML structure via CWASA adapter
        val = CWASAAdapter.validate_sigml(res["sigml"])
        assert val["valid"] is True

    def test_case_and_whitespace_normalization(self):
        res1 = self.dict_service.lookup_sign("Afraid")
        res2 = self.dict_service.lookup_sign("  afraid  ")
        assert res1["found"] is True
        assert res2["found"] is True
        assert res1["sigml"] == res2["sigml"]

    def test_missing_sign_returns_sign_not_available(self):
        res = self.dict_service.lookup_sign("unobtainium_xyz")
        assert res["found"] is False
        assert res["status"] == "SIGN_NOT_AVAILABLE"
        assert "sigml" not in res or res.get("sigml") is None
        # Ensure no fabricated HamNoSys or SiGML is returned
        assert res.get("message") == "Sign not available for 'unobtainium_xyz'"

    def test_api_sign_lookup_endpoint_success(self):
        response = client.post("/sign/lookup", json={"text": "afraid"})
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is True
        assert data["query"] == "afraid"
        assert data["status"] == "READY_FOR_CWASA"
        assert data["format"] == "sigml"
        assert "<sigml>" in data["sigml"]

    def test_api_sign_lookup_endpoint_missing(self):
        response = client.post("/sign/lookup", json={"text": "nonexistent_word_123"})
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is False
        assert data["status"] == "SIGN_NOT_AVAILABLE"
        assert data["sigml"] is None

    def test_api_gloss_to_sign_cwasa_sequence(self):
        response = client.post("/gloss-to-sign", json={"gloss": ["afraid", "water"]})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "READY_FOR_CWASA"
        assert len(data["results"]) == 2
        assert data["results"][0]["found"] is True
        assert data["results"][1]["found"] is True
        assert data["combined_sigml"] is not None
        assert "<sigml>" in data["combined_sigml"]
        assert 'gloss="afraid"' in data["combined_sigml"]
        assert 'gloss="water"' in data["combined_sigml"]

    def test_api_gloss_to_sign_all_missing(self):
        response = client.post("/gloss-to-sign", json={"gloss": ["xyz1", "xyz2"]})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SIGN_NOT_AVAILABLE"
        assert data["combined_sigml"] is None
        assert data["results"][0]["found"] is False

    def test_transcribe_undersized_audio_returns_422(self):
        # Test uploading a file under 2000 bytes returns 422 with structured AUDIO_DECODE_FAILED
        small_bytes = b"RIFF" + b"\x00" * 50
        files = {"file": ("recording.wav", small_bytes, "audio/wav")}
        response = client.post("/transcribe", files=files)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert data["detail"]["code"] == "AUDIO_DECODE_FAILED"
        assert "couldn't read that recording" in data["detail"]["message"].lower()

    def test_transcribe_corrupted_audio_returns_422(self):
        # Test uploading corrupted non-audio bytes returns 422
        bad_bytes = b"\x00" * 3000
        files = {"file": ("corrupt.webm", bad_bytes, "audio/webm")}
        response = client.post("/transcribe", files=files)
        assert response.status_code in [422, 400]
        data = response.json()
        assert "detail" in data
        assert data["detail"]["code"] == "AUDIO_DECODE_FAILED"

    def test_stt_post_processing_fuzzy_matching(self):
        from app.stt.post_processor import correct_transcript
        corrected, is_speech = correct_transcript("afrad")
        assert is_speech is True
        assert corrected == "afraid"

    def test_sign_sequence_endpoint(self):
        response = client.post("/sign/sequence", json={"text": "I want water"})
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert "sigml" in data
        assert len(data["tokens"]) == 3
        assert data["tokens"][0]["available"] is True
        assert data["tokens"][1]["available"] is True
        assert data["tokens"][2]["available"] is True
        assert "<sigml>" in data["sigml"]
        assert 'gloss="water"' in data["sigml"]



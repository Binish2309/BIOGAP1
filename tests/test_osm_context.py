import pandas as pd
import pytest

import src.ingestion.osm_context as osm_module


class _FakeResponse:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return self._json_data


def test_osm_ingestion_sends_descriptive_user_agent(monkeypatch, tmp_path):
    captured = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse(200, {"elements": []})

    monkeypatch.setattr(osm_module.requests, "post", fake_post)
    monkeypatch.setattr(osm_module, "RAW_DIR", tmp_path)
    monkeypatch.setattr(osm_module, "write_provenance",
                         lambda record, filename: (tmp_path / filename).write_text(""))

    osm_module.run_osm_context_ingestion()
    assert "User-Agent" in captured["headers"]
    assert "python-requests" not in captured["headers"]["User-Agent"]


def test_osm_ingestion_falls_back_to_second_mirror_on_406(monkeypatch, tmp_path):
    calls = []

    def fake_post(url, data=None, headers=None, timeout=None):
        calls.append(url)
        if url == osm_module.OVERPASS_MIRRORS[0]:
            return _FakeResponse(406)
        return _FakeResponse(200, {"elements": [
            {"id": 1, "tags": {"highway": "path"},
             "geometry": [{"lat": 19.0, "lon": 72.8}, {"lat": 19.01, "lon": 72.81}]},
        ]})

    monkeypatch.setattr(osm_module.requests, "post", fake_post)
    monkeypatch.setattr(osm_module, "RAW_DIR", tmp_path)
    monkeypatch.setattr(osm_module, "write_provenance",
                         lambda record, filename: (tmp_path / filename).write_text(""))

    record = osm_module.run_osm_context_ingestion(raw_output_name="test_osm.csv")

    assert len(calls) == 2  # first mirror tried and failed, second succeeded
    assert record.status == "SUCCESS"
    assert record.source_url_or_api == osm_module.OVERPASS_MIRRORS[1]
    written = pd.read_csv(tmp_path / "test_osm.csv")
    assert len(written) == 2  # two vertices in the one fake way


def test_osm_ingestion_reports_failure_when_all_mirrors_fail(monkeypatch, tmp_path):
    def fake_post(url, data=None, headers=None, timeout=None):
        return _FakeResponse(406)

    monkeypatch.setattr(osm_module.requests, "post", fake_post)
    monkeypatch.setattr(osm_module, "RAW_DIR", tmp_path)
    monkeypatch.setattr(osm_module, "write_provenance",
                         lambda record, filename: (tmp_path / filename).write_text(""))

    record = osm_module.run_osm_context_ingestion()
    assert record.status == "FAILED"
    assert "406" in record.error_detail
    assert "2 mirror" in record.extra_notes

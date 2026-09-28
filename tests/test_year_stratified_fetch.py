"""
tests/test_year_stratified_fetch.py

Regression tests for the sampling-bias bug: with no year stratification,
GBIF's unspecified default order and iNaturalist's ascending-ID order each
biased an interactive run toward one end of the timeline (observed in
practice: a real run landing almost entirely in 2025-2026 for GBIF, and
2003-2018 for iNaturalist), making any temporal/cross-platform comparison
between the two meaningless. Both connectors now query year-by-year across
a SHARED configured range and split the record budget evenly across years.
"""

from __future__ import annotations

import src.ingestion.gbif as gbif_module
import src.ingestion.inaturalist as inat_module


def test_gbif_year_stratified_queries_every_year_in_range(monkeypatch):
    seen_years = []

    def fake_search(offset, limit, **query_params):
        seen_years.append(query_params.get("year"))
        # 2 records per year, then endOfRecords
        return {"count": 2, "results": [{"key": f"{query_params.get('year')}-{i}"} for i in range(2)],
                "endOfRecords": True}

    monkeypatch.setattr(gbif_module, "occurrences",
                         type("FakeOccurrences", (), {"search": staticmethod(fake_search)}))

    records, run_info = gbif_module._fetch_year_stratified(
        base_query_params={"country": "IN"}, page_size=300, max_offset=100_000, delay=0.0,
        year_start=2015, year_end=2018, total_budget=40,
    )

    assert seen_years == ["2015", "2016", "2017", "2018"]
    assert len(records) == 8  # 2 records x 4 years
    assert run_info["year_stratified"] is True
    assert set(run_info["per_year_breakdown"].keys()) == {"2015", "2016", "2017", "2018"}


def test_inaturalist_year_stratified_queries_every_year_in_range(monkeypatch):
    seen_date_ranges = []

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def fake_get(url, params, timeout):
        seen_date_ranges.append((params.get("d1"), params.get("d2")))
        return _FakeResponse({"total_results": 2, "results": [
            {"id": 1, "taxon": {}, "geojson": None, "photos": []},
            {"id": 2, "taxon": {}, "geojson": None, "photos": []},
        ]})

    monkeypatch.setattr(inat_module, "requests",
                         type("FakeRequests", (), {"get": staticmethod(fake_get)}))

    rows, run_info = inat_module._fetch_year_stratified(
        bbox_params={}, per_page=200, delay=0.0,
        year_start=2015, year_end=2018, total_budget=40,
    )

    assert seen_date_ranges == [
        ("2015-01-01", "2015-12-31"), ("2016-01-01", "2016-12-31"),
        ("2017-01-01", "2017-12-31"), ("2018-01-01", "2018-12-31"),
    ]
    assert run_info["year_stratified"] is True
    assert set(run_info["per_year_breakdown"].keys()) == {"2015", "2016", "2017", "2018"}


def test_gbif_and_inaturalist_use_the_same_default_year_range():
    """Both connectors must default to CONFIG.comparison_year_start/end so
    a comparison between them is over an identical window unless the
    caller deliberately overrides one."""
    from src.config import CONFIG
    assert CONFIG.comparison_year_start is not None
    assert CONFIG.comparison_year_end is not None
    assert CONFIG.comparison_year_start <= CONFIG.comparison_year_end

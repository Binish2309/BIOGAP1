from src.analysis.cross_platform import (
    taxonomic_composition_by_source, temporal_range_by_source, spatial_extent_by_source,
    cross_platform_summary, PLATFORM_STRUCTURE_CAVEATS,
)


def test_taxonomic_composition_by_source_has_all_sources(mock_standard_df):
    comp = taxonomic_composition_by_source(mock_standard_df)
    assert set(comp["source"]) == {"GBIF", "iNaturalist"}


def test_taxonomic_composition_harmonizes_plant_categories_across_sources():
    import pandas as pd
    from src.processing.schema import STANDARD_COLUMNS

    def _row(source, record_id, kingdom, taxon_class):
        return dict(
            source=source, record_id=record_id, scientific_name=f"Sp {record_id}",
            taxonomic_rank="species", kingdom=kingdom, phylum="Tracheophyta",
            taxon_class=taxon_class, order=None, family=None, genus=None,
            latitude=19.0, longitude=72.8, observation_date="2022-01-01", year=2022,
            basis_of_record="HUMAN_OBSERVATION", dataset="d", publisher="p", license="CC0",
            has_media="UNKNOWN", effort_distance_km=pd.NA, effort_duration_min=pd.NA,
            effort_complete_checklist=pd.NA, is_test_data=False,
        )

    # GBIF: real class-level resolution -- one dicot, one monocot, one "other" plant.
    # iNaturalist: our connector never populates taxon_class for plants (see
    # cleaning.py::normalise_inaturalist) -- all three land as kingdom=Plantae,
    # taxon_class=None, exactly mirroring real ingested data.
    rows = [
        _row("GBIF", "g1", "Plantae", "Magnoliopsida"),
        _row("GBIF", "g2", "Plantae", "Liliopsida"),
        _row("GBIF", "g3", "Plantae", "Pinopsida"),  # neither dicot nor monocot -> "Plants (other)"
        _row("iNaturalist", "i1", "Plantae", None),
        _row("iNaturalist", "i2", "Plantae", None),
        _row("iNaturalist", "i3", "Plantae", None),
    ]
    df = pd.DataFrame(rows)[STANDARD_COLUMNS]

    comp = taxonomic_composition_by_source(df)

    # Neither of the two FINER raw sub-buckets should survive harmonization
    # (they collapse into "Plants (other)", the floor resolution iNaturalist's
    # iconic-taxon scheme can supply).
    finer_subcategories = {"Flowering plants (dicots)", "Flowering plants (monocots)"}
    assert not finer_subcategories.intersection(set(comp["major_group"]))

    # Both sources' 3 plant records collapse into ONE "Plants (other)" row each.
    gbif_plants = comp[(comp["source"] == "GBIF") & (comp["major_group"] == "Plants (other)")]
    inat_plants = comp[(comp["source"] == "iNaturalist") & (comp["major_group"] == "Plants (other)")]
    assert len(gbif_plants) == 1 and gbif_plants.iloc[0]["record_count"] == 3
    assert len(inat_plants) == 1 and inat_plants.iloc[0]["record_count"] == 3


def test_temporal_range_by_source(mock_standard_df):
    ranges = temporal_range_by_source(mock_standard_df)
    gbif_row = ranges[ranges["source"] == "GBIF"].iloc[0]
    assert gbif_row["earliest_year"] == 2022
    assert gbif_row["latest_year"] == 2022
    inat_row = ranges[ranges["source"] == "iNaturalist"].iloc[0]
    assert inat_row["earliest_year"] == 2020
    assert inat_row["latest_year"] == 2024


def test_spatial_extent_by_source(mock_standard_df):
    extents = spatial_extent_by_source(mock_standard_df, grid_size_deg=0.05)
    assert (extents["n_records_with_coordinates"] > 0).all()


def test_cross_platform_summary_includes_caveats(mock_standard_df):
    summary = cross_platform_summary(mock_standard_df, grid_size_deg=0.05)
    for source in summary["sources_present"]:
        assert source in PLATFORM_STRUCTURE_CAVEATS
        assert len(summary["platform_structure_caveats"][source]) > 0

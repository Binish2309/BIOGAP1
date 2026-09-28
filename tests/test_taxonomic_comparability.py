"""
tests/test_taxonomic_comparability.py

Regression tests for the taxonomic-category comparability bug: GBIF
records were grouped via kingdom/class (full Linnaean hierarchy), but
this connector's iNaturalist records had kingdom/phylum/taxon_class
always NA (only iconic_taxon_name is fetched -- see
src/ingestion/inaturalist.py's module docstring), so EVERY iNaturalist
record fell into "Unclassified" in taxonomic_composition_by_source()
instead of a real, comparable category like "Plants" or "Birds".

assign_major_group() now falls back to iconic_taxon_name when no
kingdom/phylum/class is populated, so a real-world iNaturalist row gets
classified. assign_comparison_group() additionally coarsens GBIF's finer
labels (e.g. dicot/monocot split) down to the same resolution iNaturalist
can supply, so a cross-platform comparison is never comparing categories
at two different resolutions.
"""

from __future__ import annotations

import pandas as pd

from src.processing.taxonomy import assign_major_group, assign_comparison_group
from src.processing.schema import STANDARD_COLUMNS


def test_inaturalist_style_row_falls_back_to_iconic_taxon():
    """A row shaped exactly like what normalise_inaturalist() actually
    produces: kingdom/phylum/taxon_class all NA, only iconic_taxon_name
    populated (as the bonus column)."""
    kingdom = pd.Series([pd.NA])
    taxon_class = pd.Series([pd.NA])
    phylum = pd.Series([pd.NA])
    iconic = pd.Series(["Plantae"])

    groups = assign_major_group(kingdom, taxon_class, phylum=phylum, iconic_taxon=iconic)

    assert groups.iloc[0] == "Plants (other)"
    assert "Unclassified" not in groups.iloc[0]


def test_gbif_style_row_ignores_iconic_taxon_when_class_is_present():
    """A GBIF row has its own class field -- iconic_taxon (which it
    doesn't have anyway) must never override real, finer GBIF data."""
    kingdom = pd.Series(["Plantae"])
    taxon_class = pd.Series(["Magnoliopsida"])
    phylum = pd.Series(["Tracheophyta"])
    iconic = pd.Series([pd.NA])

    groups = assign_major_group(kingdom, taxon_class, phylum=phylum, iconic_taxon=iconic)
    assert groups.iloc[0] == "Flowering plants (dicots)"


def _row(source, kingdom=pd.NA, taxon_class=pd.NA, phylum=pd.NA, iconic=pd.NA,
         record_id="r1", scientific_name=pd.NA):
    base = {c: pd.NA for c in STANDARD_COLUMNS}
    base.update({
        "source": source, "record_id": record_id, "scientific_name": scientific_name,
        "kingdom": kingdom, "phylum": phylum, "taxon_class": taxon_class,
    })
    base["inaturalist_iconic_taxon_name"] = iconic
    return base


def test_comparison_group_harmonises_gbif_plant_split_with_inaturalist():
    """The actual bug scenario: GBIF has a dicot record (finer detail),
    iNaturalist has a Plantae-iconic record (coarser detail) for a
    DIFFERENT species. For a fair comparison, both must land in the same
    'Plants (other)' bucket -- not one under a fine label and the other
    under 'Unclassified'."""
    df = pd.DataFrame([
        _row("GBIF", kingdom="Plantae", taxon_class="Magnoliopsida", phylum="Tracheophyta",
             record_id="g1", scientific_name="Ficus benghalensis"),
        _row("iNaturalist", iconic="Plantae", record_id="i1", scientific_name="Areca catechu"),
    ])

    fine = assign_major_group(df["kingdom"], df["taxon_class"], phylum=df["phylum"],
                               iconic_taxon=df["inaturalist_iconic_taxon_name"])
    # At full resolution these genuinely differ -- GBIF is finer.
    assert fine.iloc[0] == "Flowering plants (dicots)"
    assert fine.iloc[1] == "Plants (other)"

    comparison = assign_comparison_group(df)
    # Harmonised for comparison, both are "Plants (other)".
    assert comparison.iloc[0] == "Plants (other)"
    assert comparison.iloc[1] == "Plants (other)"


def test_comparison_group_harmonises_gbif_chondrichthyes_with_inaturalist_animalia():
    """iNaturalist's iconic-taxon scheme has no separate bucket for
    cartilaginous fish -- they'd fall under its generic 'Animalia'. GBIF's
    finer 'Cartilaginous fishes' label must collapse to match for a fair
    comparison, rather than appearing as a GBIF-only category."""
    df = pd.DataFrame([
        _row("GBIF", kingdom="Animalia", taxon_class="Chondrichthyes", phylum="Chordata",
             record_id="g1"),
        _row("iNaturalist", iconic="Animalia", record_id="i1"),
    ])
    comparison = assign_comparison_group(df)
    assert comparison.iloc[0] == comparison.iloc[1] == "Other animals"

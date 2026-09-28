"""
src/processing/taxonomy.py

Taxonomic normalisation helpers. These standardise FORMATTING (casing,
whitespace) and provide a documented, literature-grounded grouping of
kingdom/class into coarse "major taxonomic group" labels for the taxonomic-
gap analysis. They never invent or correct a taxonomic identification --
if a source says a record is identified only to genus, it stays identified
only to genus.

TWO GROUPING FUNCTIONS, TWO PURPOSES:

- assign_major_group(): the FINEST grouping each record's own data
  supports (e.g. GBIF's Magnoliopsida/Liliopsida class fields split
  flowering plants into dicots/monocots). Used for single-source or
  combined-but-not-compared views, like the Taxonomic Gaps page.

- assign_comparison_group(): a DELIBERATELY COARSER grouping, capped at
  the resolution of the LEAST detailed source actually being compared
  (in practice, iNaturalist's `iconic_taxon_name`, which mixes kingdom-
  and class-level distinctions and cannot tell a monocot from a dicot,
  or a snail from a clam). Used ONLY by src/analysis/cross_platform.py's
  taxonomic_composition_by_source(), so a GBIF-vs-iNaturalist comparison
  is never comparing categories at two different resolutions and calling
  it a real difference. See RESEARCH_METHOD.md's "Comparable taxonomic
  categories" section for why this matters and what it costs (some real
  GBIF detail is deliberately not shown on the Cross-Platform page).
"""

from __future__ import annotations

import pandas as pd

# Coarse taxonomic groups used in charts/analysis. This mapping is a
# PRESENTATION/GROUPING choice for readability, documented here so it is
# auditable -- it is not a taxonomic claim beyond what the source recorded.
# Unmapped kingdom/class combinations fall through to "Other/Unclassified"
# rather than being forced into a group they don't belong in.
_CLASS_TO_GROUP = {
    "Aves": "Birds",
    "Mammalia": "Mammals",
    "Reptilia": "Reptiles",
    "Amphibia": "Amphibians",
    "Insecta": "Insects",
    "Arachnida": "Arachnids",
    "Actinopterygii": "Ray-finned fishes",
    "Chondrichthyes": "Cartilaginous fishes",
    "Magnoliopsida": "Flowering plants (dicots)",
    "Liliopsida": "Flowering plants (monocots)",
}
_PHYLUM_TO_GROUP = {
    "Mollusca": "Molluscs",
}
_KINGDOM_TO_GROUP = {
    "Plantae": "Plants (other)",
    "Fungi": "Fungi",
    "Bacteria": "Bacteria",
    "Protozoa": "Protozoa",
    "Chromista": "Chromista",
}

# iNaturalist's `iconic_taxon_name` is the coarsest common ground between
# GBIF (which supplies full kingdom/phylum/class/order/family/genus) and
# this connector (which -- see src/ingestion/inaturalist.py's module
# docstring -- does not fetch the full Linnaean hierarchy, only this one
# top-level label per observation). It deliberately mixes ranks (e.g.
# "Aves" is a class, "Plantae" is a kingdom, "Mollusca" is a phylum) --
# that is iNaturalist's own UI/API design, not something invented here.
# Mapped onto the SAME target labels as _CLASS_TO_GROUP/_PHYLUM_TO_GROUP/
# _KINGDOM_TO_GROUP wherever a matching concept exists, so a record
# grouped this way and a GBIF record grouped via class/kingdom land in the
# identical bucket name. Where iNaturalist's resolution is coarser than
# GBIF's (all flowering plants under one "Plantae" label, no dicot/monocot
# split; all non-iconic invertebrates other than Mollusca/Arachnida/Insecta
# folded into "Animalia"), the target label reflects that coarser
# resolution -- see assign_comparison_group(), which additionally coarsens
# GBIF's OWN finer labels to match, for genuinely apples-to-apples
# cross-platform comparison.
_ICONIC_TAXON_TO_GROUP = {
    "Aves": "Birds",
    "Mammalia": "Mammals",
    "Reptilia": "Reptiles",
    "Amphibia": "Amphibians",
    "Insecta": "Insects",
    "Arachnida": "Arachnids",
    "Actinopterygii": "Ray-finned fishes",
    "Mollusca": "Molluscs",
    "Plantae": "Plants (other)",
    "Fungi": "Fungi",
    "Protozoa": "Protozoa",
    "Chromista": "Chromista",
    "Animalia": "Other animals",  # iNaturalist's catch-all for animals with no more specific iconic taxon
}

# Collapses assign_major_group()'s finer labels down to whatever
# assign_comparison_group() can actually resolve for EVERY source being
# compared -- i.e. the resolution ceiling is iNaturalist's iconic-taxon
# scheme above. Only labels finer than that scheme need collapsing; labels
# already at or below that resolution map to themselves.
_COARSEN_FOR_COMPARISON = {
    "Flowering plants (dicots)": "Plants (other)",
    "Flowering plants (monocots)": "Plants (other)",
    "Cartilaginous fishes": "Other animals",  # iNaturalist's iconic-taxon scheme has no separate bucket for these
}


def normalise_text_field(series: pd.Series) -> pd.Series:
    """Trim whitespace and collapse internal multi-spaces. Case is left
    exactly as the source provided it -- we do not guess correct
    capitalisation of scientific names."""
    return series.apply(
        lambda v: " ".join(v.split()) if isinstance(v, str) else v
    )


def assign_major_group(kingdom: pd.Series, taxon_class: pd.Series,
                        phylum: pd.Series | None = None,
                        iconic_taxon: pd.Series | None = None) -> pd.Series:
    """
    Assign each record a coarse, human-readable major taxonomic group for
    charting, at the FINEST resolution the record's own data supports.
    Falls through class -> phylum -> kingdom -> iconic_taxon (iNaturalist's
    own top-level label, used ONLY when nothing more specific is present,
    e.g. this connector's raw kingdom/phylum/class fields are always
    missing -- see src/processing/cleaning.py::normalise_inaturalist) ->
    "Other/Unclassified". Documented mapping above; extend it there, not
    ad hoc in analysis code.
    """
    n = len(kingdom)
    phylum = phylum if phylum is not None else pd.Series([pd.NA] * n, index=kingdom.index)
    iconic_taxon = iconic_taxon if iconic_taxon is not None else pd.Series([pd.NA] * n, index=kingdom.index)

    def _one(k, c, p, ic):
        if isinstance(c, str) and c in _CLASS_TO_GROUP:
            return _CLASS_TO_GROUP[c]
        if isinstance(p, str) and p in _PHYLUM_TO_GROUP:
            return _PHYLUM_TO_GROUP[p]
        if isinstance(k, str) and k in _KINGDOM_TO_GROUP:
            return _KINGDOM_TO_GROUP[k]
        if isinstance(k, str) and k == "Animalia":
            return "Other animals"
        if isinstance(ic, str) and ic in _ICONIC_TAXON_TO_GROUP:
            return _ICONIC_TAXON_TO_GROUP[ic]
        if pd.isna(k) and pd.isna(c) and pd.isna(p) and pd.isna(ic):
            return "Unclassified (no taxonomic group field populated by this source)"
        return "Other/Unclassified"

    return pd.Series(
        [_one(k, c, p, ic) for k, c, p, ic in zip(kingdom, taxon_class, phylum, iconic_taxon)],
        index=kingdom.index,
    )


def assign_comparison_group(df: pd.DataFrame) -> pd.Series:
    """
    Assign each record a major taxonomic group HARMONISED to the coarsest
    resolution any currently-ingested source can supply -- in practice,
    iNaturalist's iconic-taxon scheme (see module docstring). Use this
    (never assign_major_group()) whenever a group label from one source is
    being compared to or plotted alongside a group label from another
    source, so "Plants" from iNaturalist and "Plants" from GBIF mean the
    same underlying resolution rather than one being finer than the other.

    Requires columns kingdom, taxon_class, phylum, and (if present)
    inaturalist_iconic_taxon_name; missing columns are treated as entirely
    NA, exactly like assign_major_group().
    """
    kingdom = df["kingdom"] if "kingdom" in df.columns else pd.Series([pd.NA] * len(df), index=df.index)
    taxon_class = df["taxon_class"] if "taxon_class" in df.columns else pd.Series([pd.NA] * len(df), index=df.index)
    phylum = df["phylum"] if "phylum" in df.columns else pd.Series([pd.NA] * len(df), index=df.index)
    iconic = (df["inaturalist_iconic_taxon_name"] if "inaturalist_iconic_taxon_name" in df.columns
              else pd.Series([pd.NA] * len(df), index=df.index))

    fine = assign_major_group(kingdom, taxon_class, phylum, iconic)
    return fine.apply(lambda g: _COARSEN_FOR_COMPARISON.get(g, g))


def taxonomic_completeness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return boolean columns describing how far down the taxonomic hierarchy
    each record was identified, based on which fields are populated (not
    on any external checklist). Purely descriptive of THIS dataset.
    """
    return pd.DataFrame({
        "identified_to_species": df["scientific_name"].notna() & (df["taxonomic_rank"].fillna("") == "species"),
        "identified_to_genus_only": df["genus"].notna() & df["scientific_name"].isna(),
        "missing_all_taxonomy": df[["kingdom", "phylum", "taxon_class", "order", "family", "genus"]].isna().all(axis=1),
    })

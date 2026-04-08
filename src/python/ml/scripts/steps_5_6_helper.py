"""
Notebook helper for enrichment steps 5 and 6.

Usage from a notebook:

    %run ../../scripts/steps_5_6_helper.py
    df_5_6 = add_steps_5_6(df)

`df` must already contain:
    - h3_index
    - lat
    - lng
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl


def _load_enrich_module():
    script_path = Path(__file__).with_name("enrich_h3_exog.py")
    spec = importlib.util.spec_from_file_location("enrich_h3_exog", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_enrich = _load_enrich_module()


def add_steps_5_6(
    df: pl.DataFrame,
    transmission_checkpoint_name: str = "transmission_complete_subset",
    roads_checkpoint_name: str = "roads_complete_subset",
) -> pl.DataFrame:
    """
    Add:
        - h3_dist_to_transmission_km
        - h3_dist_to_major_road_km

    to an existing dataframe that already has H3 + EE columns.
    """
    required_cols = {"h3_index", "lat", "lng"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"df is missing required columns: {missing_cols}")

    unique_df = _enrich.unique_h3_centroids(df)

    trans_data = _enrich.load_or_compute_checkpoint(
        transmission_checkpoint_name,
        unique_df,
        _enrich.fetch_transmission_distances,
        "transmission distances",
    )

    road_data = _enrich.load_or_compute_checkpoint(
        roads_checkpoint_name,
        unique_df,
        _enrich.fetch_road_distances,
        "road distances",
    )

    rows = []
    for h3_index in unique_df["h3_index"].to_list():
        row = {"h3_index": h3_index}
        row.update(trans_data.get(h3_index, {}))
        row.update(road_data.get(h3_index, {}))
        rows.append(row)

    extra_df = pl.DataFrame(rows)
    return df.join(extra_df, on="h3_index", how="left")


# Notebook example:
#
# %run ../../scripts/steps_5_6_helper.py
# df_5_6 = add_steps_5_6(df)
# df_5_6.write_csv("../../data/final_df_with_steps_5_6.csv")

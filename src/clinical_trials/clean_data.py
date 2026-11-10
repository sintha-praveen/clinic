"""Clean the supplied clinical-trial CSV without modifying the source."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "clinical_trials_raw_patient2trial_conditions (1).csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "cleaning_report.json"
CHUNK_SIZE = 50_000
AGE_PATTERN = (
    r"(?i)^(?P<value>\d+(?:\.\d+)?)\s+"
    r"(?P<unit>years?|months?|weeks?|days?|hours?|minutes?)$"
)
AGE_UNIT_IN_YEARS = {
    "year": 1.0,
    "years": 1.0,
    "month": 1 / 12,
    "months": 1 / 12,
    "week": 7 / 365.25,
    "weeks": 7 / 365.25,
    "day": 1 / 365.25,
    "days": 1 / 365.25,
    "hour": 1 / (365.25 * 24),
    "hours": 1 / (365.25 * 24),
    "minute": 1 / (365.25 * 24 * 60),
    "minutes": 1 / (365.25 * 24 * 60),
}
METADATA_COLUMNS = ("overall_status", "study_type", "phase", "sex")


def parse_age_years(values: pd.Series) -> tuple[pd.Series, int]:
    """Convert age strings to years and count nonblank unparseable values."""
    parts = values.str.extract(AGE_PATTERN)
    magnitude = pd.to_numeric(parts["value"], errors="coerce")
    factor = parts["unit"].str.lower().map(AGE_UNIT_IN_YEARS)
    age_years = (magnitude * factor).astype("Float64")
    unparsed = int((values.notna() & age_years.isna()).sum())
    return age_years, unparsed


def clean_chunk(chunk: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize one chunk while retaining original records and raw ages."""
    for column in chunk.columns:
        chunk[column] = chunk[column].str.strip().replace("", pd.NA)

    missing_values = {column: int(chunk[column].isna().sum()) for column in chunk.columns}

    for column in METADATA_COLUMNS:
        chunk[column] = chunk[column].str.upper()

    volunteer_values = chunk["healthy_volunteers"].str.casefold()
    invalid_volunteer_values = volunteer_values.notna() & ~volunteer_values.isin(
        ("true", "false")
    )
    if invalid_volunteer_values.any():
        raise ValueError("Unexpected value in healthy_volunteers; refusing lossy conversion.")
    chunk["healthy_volunteers"] = volunteer_values.map(
        {"true": True, "false": False}
    ).astype("boolean")

    minimum_age_years, invalid_minimum_age = parse_age_years(chunk["minimum_age"])
    maximum_age_years, invalid_maximum_age = parse_age_years(chunk["maximum_age"])
    chunk["minimum_age_years"] = minimum_age_years
    chunk["maximum_age_years"] = maximum_age_years
    chunk["maximum_age"] = chunk["maximum_age"].fillna("Not reported")

    both_ages_present = minimum_age_years.notna() & maximum_age_years.notna()
    invalid_age_ranges = int(
        (both_ages_present & (minimum_age_years > maximum_age_years)).sum()
    )
    missing_keys = int(
        (chunk["source_condition_query"].isna() | chunk["nct_id"].isna()).sum()
    )
    invalid_nct_ids = int(
        (~chunk["nct_id"].str.fullmatch(r"NCT\d{8}", case=False, na=False)).sum()
    )
    invalid_urls = int(
        (
            ~chunk["clinicaltrials_url"].str.fullmatch(
                r"https://clinicaltrials\.gov/study/NCT\d{8}/?",
                case=False,
                na=False,
            )
        ).sum()
    )
    chunk["nct_id"] = chunk["nct_id"].str.upper()

    metrics = {
        "rows": len(chunk),
        "missing_key_rows": missing_keys,
        "invalid_nct_ids": invalid_nct_ids,
        "invalid_clinicaltrials_urls": invalid_urls,
        "unparsed_minimum_ages": invalid_minimum_age,
        "unparsed_maximum_ages": invalid_maximum_age,
        "invalid_age_ranges": invalid_age_ranges,
    }
    metrics.update({f"missing_{column}": count for column, count in missing_values.items()})
    return chunk, metrics


def clean_csv(input_path: Path, output_path: Path, report_path: Path) -> dict[str, object]:
    """Clean a CSV in chunks and write the result and aggregate quality report."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".tmp")
    totals: Counter[str] = Counter()
    metadata_counts = {column: Counter() for column in METADATA_COLUMNS}
    volunteer_counts: Counter[str] = Counter()
    seen_keys: set[tuple[str, str]] = set()
    duplicate_key_rows = 0
    wrote_header = False

    try:
        with temporary_output.open("w", encoding="utf-8", newline="") as output_file:
            chunks = pd.read_csv(
                input_path,
                dtype="string",
                keep_default_na=False,
                chunksize=CHUNK_SIZE,
            )
            for chunk in chunks:
                cleaned, metrics = clean_chunk(chunk)
                totals.update(metrics)
                for column, counts in metadata_counts.items():
                    counts.update(cleaned[column].dropna().value_counts().to_dict())
                volunteer_counts.update(
                    cleaned["healthy_volunteers"].dropna().astype("string").value_counts().to_dict()
                )
                keys = cleaned[["source_condition_query", "nct_id"]].fillna("")
                for key in keys.itertuples(index=False, name=None):
                    if key in seen_keys:
                        duplicate_key_rows += 1
                    else:
                        seen_keys.add(key)
                cleaned.to_csv(output_file, index=False, header=not wrote_header)
                wrote_header = True

        if not wrote_header:
            raise ValueError("The input CSV contains no data rows.")
        temporary_output.replace(output_path)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    report: dict[str, object] = {
        "source_file": input_path.name,
        "cleaned_file": str(output_path.relative_to(PROJECT_ROOT)),
        "rows_read_and_written": totals["rows"],
        "rows_removed": 0,
        "duplicate_source_condition_trial_keys": duplicate_key_rows,
        "missing_values_by_column": {
            name.removeprefix("missing_"): count
            for name, count in totals.items()
            if name.startswith("missing_") and count
        },
        "missing_key_rows": totals["missing_key_rows"],
        "invalid_nct_ids": totals["invalid_nct_ids"],
        "invalid_clinicaltrials_urls": totals["invalid_clinicaltrials_urls"],
        "unparsed_minimum_ages": totals["unparsed_minimum_ages"],
        "unparsed_maximum_ages": totals["unparsed_maximum_ages"],
        "minimum_age_greater_than_maximum_age": totals["invalid_age_ranges"],
        "metadata_value_counts": {
            column: dict(counts) for column, counts in metadata_counts.items()
        },
        "healthy_volunteers_value_counts": dict(volunteer_counts),
        "age_conversion": "Years are exact; months, weeks, days, hours, and minutes use 365.25 days per year.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = clean_csv(args.input, args.output, args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
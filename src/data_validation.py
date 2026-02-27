"""
TFDV Data Validation Pipeline for a tabular classification dataset.

What it does:
1) Reads CSV -> TFRecord (tf.Example)
2) Generates TFDV statistics on train + test
3) Infers schema from train stats
4) Validates test vs schema and writes anomalies
5) Computes drift/skew statistics (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdv


FEATURE_COLUMNS = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
    "internet_service",
    "contract_type",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "churn",
]


def _to_feature(v):
    # Best-effort conversion to tf.train.Feature
    if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v == ""):
        # represent missing by omitting the feature
        return None

    if isinstance(v, (int, bool)) or (isinstance(v, float) and float(v).is_integer()):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))

    if isinstance(v, (float,)):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))

    # fall back to bytes
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(v).encode("utf-8")]))


def csv_to_tfrecord(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    with tf.io.TFRecordWriter(str(out_path)) as w:
        for _, row in df.iterrows():
            feat = {}
            for col in df.columns:
                f = _to_feature(row[col])
                if f is not None:
                    feat[col] = f
            ex = tf.train.Example(features=tf.train.Features(feature=feat))
            w.write(ex.SerializeToString())


def generate_statistics(tfrecord_path: Path, output_stats_path: Path) -> tfdv.Statistics:
    stats = tfdv.generate_statistics_from_tfrecord(data_location=str(tfrecord_path))
    tfdv.write_stats_text(stats, str(output_stats_path))
    return stats


def infer_and_save_schema(train_stats: tfdv.Statistics, schema_path: Path) -> tfdv.Schema:
    schema = tfdv.infer_schema(train_stats)
    # Mark label as categorical int with 2 values
    tfdv.set_domain(schema, "churn", tfdv.schema_pb2.IntDomain(min=0, max=1, is_categorical=True))
    tfdv.write_schema_text(schema, str(schema_path))
    return schema


def validate(stats: tfdv.Statistics, schema: tfdv.Schema, anomalies_path: Path) -> tfdv.Anomalies:
    anomalies = tfdv.validate_statistics(stats, schema)
    tfdv.write_anomalies_text(anomalies, str(anomalies_path))
    return anomalies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Folder containing train/test CSVs")
    ap.add_argument("--out_dir", type=str, required=True, help="Folder to write TFDV outputs")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert CSV -> TFRecord for fast stats generation
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    anom_csv = data_dir / "test_anomalous.csv"

    train_tfr = out_dir / "train.tfrecord"
    test_tfr = out_dir / "test.tfrecord"
    anom_tfr = out_dir / "test_anomalous.tfrecord"

    csv_to_tfrecord(train_csv, train_tfr)
    csv_to_tfrecord(test_csv, test_tfr)
    csv_to_tfrecord(anom_csv, anom_tfr)

    train_stats_path = out_dir / "train_stats.txt"
    test_stats_path = out_dir / "test_stats.txt"
    anom_stats_path = out_dir / "test_anom_stats.txt"

    train_stats = generate_statistics(train_tfr, train_stats_path)
    test_stats = generate_statistics(test_tfr, test_stats_path)
    anom_stats = generate_statistics(anom_tfr, anom_stats_path)

    schema_path = out_dir / "schema.pbtxt"
    schema = infer_and_save_schema(train_stats, schema_path)

    test_anom_path = out_dir / "anomalies_test.txt"
    validate(test_stats, schema, test_anom_path)

    anom_anom_path = out_dir / "anomalies_test_anomalous.txt"
    validate(anom_stats, schema, anom_anom_path)

    # Drift & Skew (basic example): compare train vs test for a few features
    # Note: for more advanced drift, you can tune thresholds per feature.
    drift_skew = tfdv.validate_statistics(
        statistics=test_stats,
        schema=schema,
        previous_statistics=train_stats,
    )
    tfdv.write_anomalies_text(drift_skew, str(out_dir / "drift_skew_report.txt"))

    print("Done. Outputs written to:", out_dir)


if __name__ == "__main__":
    main()

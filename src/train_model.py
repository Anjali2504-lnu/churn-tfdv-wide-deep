"""
Train a Wide & Deep Keras model on a tabular churn dataset.

Highlights:
- tf.data input pipeline from CSV
- Keras preprocessing layers (StringLookup / IntegerLookup / Normalization)
- Wide & Deep model (functional API)
- Metrics: AUC, Precision, Recall
- Saves model + training history
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import tensorflow as tf


CATEGORICAL_STR = ["gender", "internet_service", "contract_type", "payment_method"]
CATEGORICAL_INT = ["senior_citizen", "partner", "dependents", "paperless_billing"]
NUMERIC = ["tenure_months", "monthly_charges", "total_charges"]
LABEL = "churn"
ID_COL = "customer_id"


def make_dataset(csv_path: Path, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.experimental.make_csv_dataset(
        file_pattern=str(csv_path),
        batch_size=batch_size,
        label_name=LABEL,
        num_epochs=1,
        header=True,
        na_value="",
        shuffle=shuffle,
        shuffle_buffer_size=4096,
        ignore_errors=True,
    )
    # Remove ID from features
    ds = ds.map(lambda x, y: ({k: v for k, v in x.items() if k != ID_COL}, y))
    return ds.prefetch(tf.data.AUTOTUNE)


def build_preprocessing_layers(train_df: pd.DataFrame):
    inputs = {}
    encoded = []

    # String categorical -> embeddings
    for col in CATEGORICAL_STR:
        inp = tf.keras.Input(shape=(1,), name=col, dtype=tf.string)
        lookup = tf.keras.layers.StringLookup(output_mode="int", name=f"{col}_lookup")
        lookup.adapt(train_df[col].astype(str).values)
        vocab_size = lookup.vocabulary_size()

        x = lookup(inp)
        x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=min(16, max(4, vocab_size // 2)), name=f"{col}_emb")(x)
        x = tf.keras.layers.Reshape((-1,), name=f"{col}_reshape")(x)
        inputs[col] = inp
        encoded.append(x)

    # Integer categorical -> one-hot (wide)
    wide_features = []
    for col in CATEGORICAL_INT:
        inp = tf.keras.Input(shape=(1,), name=col, dtype=tf.int32)
        lookup = tf.keras.layers.IntegerLookup(output_mode="one_hot", name=f"{col}_onehot")
        lookup.adapt(train_df[col].astype(int).values)
        x = lookup(inp)
        inputs[col] = inp
        wide_features.append(x)

    # Numeric -> normalization
    for col in NUMERIC:
        inp = tf.keras.Input(shape=(1,), name=col, dtype=tf.float32)
        norm = tf.keras.layers.Normalization(name=f"{col}_norm")
        norm.adapt(train_df[col].astype(float).values.reshape(-1, 1))
        x = norm(inp)
        inputs[col] = inp
        encoded.append(x)

    wide = tf.keras.layers.Concatenate(name="wide_concat")(wide_features) if wide_features else None
    deep = tf.keras.layers.Concatenate(name="deep_concat")(encoded) if encoded else None
    return inputs, wide, deep


def build_model(inputs, wide, deep):
    # Deep tower
    x = deep
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)

    # Combine with wide tower
    if wide is not None:
        combined = tf.keras.layers.Concatenate(name="wide_deep_concat")([wide, x])
    else:
        combined = x

    out = tf.keras.layers.Dense(1, activation="sigmoid", name="churn_prob")(combined)
    model = tf.keras.Model(inputs=inputs, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    # Load a pandas frame just for adapt() (preprocessing layers)
    train_df = pd.read_csv(train_csv)

    train_ds = make_dataset(train_csv, args.batch_size, shuffle=True)
    val_ds = make_dataset(val_csv, args.batch_size, shuffle=False)
    test_ds = make_dataset(test_csv, args.batch_size, shuffle=False)

    inputs, wide, deep = build_preprocessing_layers(train_df)
    model = build_model(inputs, wide, deep)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(out_dir / "checkpoints.keras"), monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(out_dir / "training_log.csv")),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Evaluate
    eval_metrics = model.evaluate(test_ds, return_dict=True)
    (out_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))

    # Save final model
    model.save(out_dir / "saved_model", include_optimizer=False)

    # Save history
    (out_dir / "history.json").write_text(json.dumps(history.history, indent=2))

    print("Saved model and metrics to:", out_dir)


if __name__ == "__main__":
    main()

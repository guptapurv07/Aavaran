"""Stage 7: score the live transaction stream with the global federated model.

Reads transactions from Kafka in micro-batches, builds the same features used in
training, scores them, and publishes every transaction above the threshold to
the fraud-alerts topic. When the stream goes quiet it writes throughput and
latency numbers to results/streaming.json (or the file named by --out).

  python src/scorer.py                      # uses models/random/seed_42/federated.pt
  python src/scorer.py --model federated_dp_eps5
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from confluent_kafka import Consumer, Producer

from config import (
    ALERTS_TOPIC, KAFKA_BOOTSTRAP, RESULTS_DIR, SEED, SPLITS, THRESHOLD, TRANSACTIONS_TOPIC, model_dir, processed_dir,
)
from model import FraudNet, predict_proba
from preprocess import build_features, scale


def load_scorer(name: str, split: str = "random", seed: int = SEED) -> tuple[FraudNet, np.ndarray, np.ndarray]:
    model = FraudNet()
    model.load_state_dict(torch.load(model_dir(split, seed) / f"{name}.pt"))
    with open(processed_dir(split) / "scaler.json") as f:
        scaler = json.load(f)
    return model, np.array(scaler["mean"]), np.array(scaler["std"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="federated", help="model name, without .pt")
    parser.add_argument("--split", choices=SPLITS, default="random", help="which experiment the model comes from")
    parser.add_argument("--seed", type=int, default=SEED, help="which training seed the model comes from")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--batch", type=int, default=500, help="max transactions per micro-batch")
    parser.add_argument("--max-wait-ms", type=int, default=100, help="max time to wait while filling a micro-batch")
    parser.add_argument("--idle-timeout", type=float, default=15, help="stop after this many idle seconds; 0 = never")
    parser.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    parser.add_argument("--out", default="streaming.json", help="stats file name in results/")
    args = parser.parse_args()

    model, mean, std = load_scorer(args.model, args.split, args.seed)
    torch.set_num_threads(1)  # micro-batches are small; extra threads only add overhead
    consumer = Consumer({
        "bootstrap.servers": args.bootstrap,
        "group.id": "aavaran-scorer",
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe(
        [TRANSACTIONS_TOPIC],
        on_assign=lambda _, parts: print(f"Ready: assigned {len(parts)} partition(s)", flush=True),
    )
    alerts = Producer({"bootstrap.servers": args.bootstrap, "linger.ms": 5})

    scored = flagged = true_alerts = missed = 0
    latencies_ms, scoring_seconds = [], 0.0
    first_seen = last_seen = None
    model_path = model_dir(args.split, args.seed) / f"{args.model}.pt"
    print(f"Scoring '{TRANSACTIONS_TOPIC}' with {model_path.relative_to(RESULTS_DIR.parent)} (threshold {args.threshold}) ...")
    try:
        while True:
            batch = consumer.consume(num_messages=args.batch, timeout=args.max_wait_ms / 1000)
            msgs = [m for m in batch if not m.error()]
            if not msgs:
                if args.idle_timeout and last_seen and time.time() - last_seen > args.idle_timeout:
                    break
                continue
            first_seen = first_seen or time.time()
            df = pd.DataFrame.from_records([json.loads(m.value()) for m in msgs])

            t0 = time.perf_counter()
            prob = predict_proba(model, scale(build_features(df), mean, std))
            scoring_seconds += time.perf_counter() - t0

            is_alert = prob >= args.threshold
            for rec, p in zip(df[is_alert].to_dict("records"), prob[is_alert]):
                alerts.produce(ALERTS_TOPIC, key=str(rec["bank"]), value=json.dumps({
                    "tx_id": rec["tx_id"], "bank": rec["bank"], "type": rec["type"],
                    "amount": rec["amount"], "nameOrig": rec["nameOrig"],
                    "fraud_probability": round(float(p), 4), "scored_at": time.time(),
                }))
            alerts.poll(0)

            last_seen = time.time()
            latencies_ms.extend((last_seen - df["sent_at"].to_numpy()) * 1000)
            fraud = df["isFraud"].to_numpy() == 1
            scored += len(df)
            flagged += int(is_alert.sum())
            true_alerts += int((is_alert & fraud).sum())
            missed += int((~is_alert & fraud).sum())
            if scored // 20_000 != (scored - len(df)) // 20_000:
                print(f"  scored {scored:,}  alerts {flagged:,}")
    except KeyboardInterrupt:
        pass
    finally:
        alerts.flush()
        consumer.close()

    if not scored:
        print("No transactions received.")
        return
    lat = np.array(latencies_ms)
    stats = {
        "model": str(model_path.relative_to(RESULTS_DIR.parent)),
        "transactions": scored,
        "alerts": flagged,
        "true_alerts": true_alerts,
        "false_alarms": flagged - true_alerts,
        "missed_fraud": missed,
        "throughput_tx_per_s": round(scored / max(last_seen - first_seen, 1e-9), 1),
        "scoring_capacity_tx_per_s": round(scored / scoring_seconds, 1),
        "latency_ms": {
            "mean": round(float(lat.mean()), 2),
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "p99": round(float(np.percentile(lat, 99)), 2),
        },
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

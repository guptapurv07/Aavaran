"""Stage 7: replay held-out PaySim transactions into Kafka as a live stream.

Only test-set transactions are replayed, so the model has never seen them. Each
message is keyed by the bank that received the transaction. The fraud label
rides along only so the scorer can measure live precision and recall; the
scorer never uses it to make a decision.

  python src/producer.py --rate 2000 --limit 100000
"""

import argparse
import json
import time

import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from config import ALERTS_TOPIC, KAFKA_BOOTSTRAP, SPLITS, TRANSACTIONS_TOPIC, processed_dir


def ensure_topics(bootstrap: str) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    existing = admin.list_topics(timeout=10).topics
    new = [NewTopic(t, num_partitions=1, replication_factor=1)
           for t in (TRANSACTIONS_TOPIC, ALERTS_TOPIC) if t not in existing]
    if not new:
        return
    for topic, future in admin.create_topics(new).items():
        future.result()
        print(f"Created topic {topic}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rate", type=float, default=2000, help="transactions per second; 0 = as fast as possible")
    parser.add_argument("--limit", type=int, default=100_000, help="number of transactions to send")
    parser.add_argument("--split", choices=SPLITS, default="random", help="which test set to replay")
    parser.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    args = parser.parse_args()

    ensure_topics(args.bootstrap)
    stream = processed_dir(args.split) / "test_stream.parquet"
    records = pd.read_parquet(stream).head(args.limit).to_dict("records")
    producer = Producer({"bootstrap.servers": args.bootstrap, "linger.ms": 5, "compression.type": "lz4"})

    print(f"Sending {len(records):,} transactions to '{TRANSACTIONS_TOPIC}' at "
          f"{'max speed' if not args.rate else f'{args.rate:,.0f}/s'} ...")
    start = time.perf_counter()
    for i, rec in enumerate(records):
        rec["tx_id"] = i
        rec["sent_at"] = time.time()
        while True:
            try:
                producer.produce(TRANSACTIONS_TOPIC, key=str(rec["bank"]), value=json.dumps(rec))
                break
            except BufferError:  # local queue full: let it drain
                producer.poll(0.05)
        if i % 500 == 0:
            producer.poll(0)
            if args.rate:
                ahead = start + i / args.rate - time.perf_counter()
                if ahead > 0:
                    time.sleep(ahead)
            if i and i % 20_000 == 0:
                print(f"  sent {i:,}")
    producer.flush()
    elapsed = time.perf_counter() - start
    print(f"Sent {len(records):,} transactions in {elapsed:.1f}s ({len(records) / elapsed:,.0f}/s)")


if __name__ == "__main__":
    main()

# Aavaran: Privacy-Preserving Federated Fraud Detection

## Problem Statement

Banks lose billions every year to payment fraud. Fraud gangs reuse the same tricks across many banks, but each bank can only learn from its own customers' transactions. Privacy laws (such as India's DPDP Act and Europe's GDPR), banking secrecy and competition stop banks from pooling customer data.

The result: every bank has to learn about each new fraud pattern separately, on its own, and usually only after its own customers have already lost money. Fraudsters take advantage of this by moving the same scheme from one bank to the next.

**Aavaran lets several banks train one shared fraud detection model together without any bank ever sharing a single customer record.**

## A Simple Example

A fraud gang buys a batch of stolen card details, from cards issued by many different banks. They use the same playbook on every card:

1. Make two or three tiny "test" payments of ₹1 to ₹10 to check the card works.
2. Within minutes, make one large purchase at a newly registered online merchant, often late at night.

The cards are different, but the **behaviour pattern is the same**.

- **Without Aavaran:** Bank A gets hit first, and after enough losses its model learns to flag this pattern. Banks B, C and D have never seen it, so their models let it through, and their customers lose money too. Bank A cannot share its data with them.
- **With Aavaran:** Bank A's model learns the pattern locally. Only the model's learned weights (numbers, not customer data) are sent to a central aggregator, which combines them with the updates from the other banks. The improved global model goes back to every bank. Now Banks B, C and D can catch the pattern before it hits them, and no customer data has left Bank A.

## How It Works (The Big Idea)

Aavaran uses **federated learning**:

1. Each bank (client) keeps its transaction data on its own servers.
2. Each bank trains a copy of the fraud model on its own data.
3. Each bank sends only its model updates (weights) to a central server.
4. The server combines the updates using **FedAvg** (Federated Averaging) to make a better global model.
5. The global model is sent back to all banks, and the cycle repeats for several rounds.
6. **Differential privacy** adds carefully controlled noise to the updates, so nobody, not even the server, can work backwards to individual customers.

## Technology Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Language | Python 3.10+ | Core development |
| Data processing | Pandas, NumPy, scikit-learn | Cleaning, splitting, scaling, metrics |
| Model | PyTorch | Neural network for fraud classification |
| Federated learning | Flower (flwr) | Client and server orchestration, FedAvg |
| Privacy | Opacus | Differential privacy during local training |
| Imbalance handling | Class weights / SMOTE (imbalanced-learn) | Fraud is rare, so the classes are heavily imbalanced |
| Dashboard | Streamlit, Plotly | Visualising training rounds and results |
| Deployment | Docker, Docker Compose | Local Kafka broker now; each bank as its own container in phase 2 |
| Streaming | Apache Kafka | Real-time transaction scoring |
| Explanations (phase 2) | SHAP + an LLM | Plain-English reasons for each fraud alert |

**Dataset:** PaySim synthetic mobile money dataset (Kaggle), about 6.3 million transactions with fraud labels. IEEE-CIS Fraud Detection is an optional alternative.

## Step-by-Step Implementation

### Step 1: Data Preparation

- Download PaySim and load it with Pandas.
- Engineer features: transaction amount, type, balance changes before and after, and time step.
- Scale numeric features and encode transaction types.
- Split the data into **5 simulated banks**, using a **non-IID split** so each bank sees a different mix (for example, Bank 1 is heavy on TRANSFER fraud and Bank 2 on CASH\_OUT fraud). This is what makes collaboration valuable.
- Keep a common held-out test set for fair evaluation.

### Step 2: Baseline Model (Single Bank)

- Build a small feed-forward neural network in PyTorch (for example 3 layers with ReLU and dropout, and a sigmoid output).
- Handle class imbalance with a weighted loss.
- Train on one bank's data alone.
- Evaluate with **precision, recall, F1-score and PR-AUC**. Accuracy alone is misleading because fraud is under 1% of transactions.

### Step 3: Federated Learning with Flower

- Wrap the training code in a Flower `NumPyClient` (methods: `get_parameters`, `fit`, `evaluate`).
- Create a Flower server using the **FedAvg** strategy.
- Run 5 clients (one per bank) and train for about 20 rounds.
- Log each round's global model metrics.

### Step 4: Differential Privacy with Opacus

- Attach Opacus's `PrivacyEngine` to each client's optimiser.
- Train at several privacy levels (epsilon values such as 1, 5 and 10).
- Record the **privacy vs performance trade-off**.

### Step 5: Comparison Experiments

Compare three setups on the same test set:

1. **Isolated:** each bank trains alone.
2. **Federated:** Aavaran, with and without differential privacy.
3. **Centralised:** all data pooled (the theoretical upper limit, not allowed in real life).

The goal is to show that federated performance gets close to centralised performance and clearly beats isolated banks, especially on fraud types a bank has rarely seen.

### Step 6: Real-Time Scoring with Kafka

- Run a single Kafka broker locally with Docker Compose (KRaft mode, so no ZooKeeper).
- Write a producer (`producer.py`) that replays PaySim transactions into a `transactions` topic at a steady rate, each tagged with its bank.
- Write a scoring consumer (`scorer.py`) that loads the latest global federated model, scores each transaction, and publishes flagged ones to a `fraud-alerts` topic.
- Measure throughput (transactions per second) and average scoring latency, so the resume line has real numbers.
- Use the `confluent-kafka` Python client.

### Step 7: Dashboard

A Streamlit app showing:

- Global model metrics across training rounds.
- Per-bank improvement before and after federation.
- The privacy vs accuracy trade-off chart.
- A transaction tester: enter transaction details and get a fraud score, plus a live feed of flagged transactions from the fraud-alerts topic.

### Step 8: Phase 2 Extensions (After the Deadline)

- Containerise each bank and the server with Docker Compose.
- Scale the Kafka pipeline with multiple partitions and consumer groups.
- Add SHAP explanations and an LLM layer that turns each alert into a plain-English reason.
- Experiment with stronger strategies such as FedProx for very uneven bank data.
- Add secure aggregation.

## Project Structure

```
aavaran/
├── data/              # raw and processed data, bank splits
├── src/
│   ├── preprocess.py  # cleaning, features, non-IID split
│   ├── model.py       # PyTorch fraud model
│   ├── client.py      # Flower client
│   ├── server.py      # Flower server with FedAvg
│   ├── privacy.py     # Opacus setup
│   ├── evaluate.py    # metrics and comparisons
│   ├── producer.py    # streams transactions into Kafka
│   └── scorer.py      # scores the stream, publishes fraud alerts
├── dashboard/app.py   # Streamlit dashboard
├── results/           # logs and charts
├── docker-compose.yml # local Kafka broker
├── requirements.txt
└── README.md
```

## Timeline and Goal

**Goal: about 70% complete by day after tomorrow (25 October 2026).**

| When | Work | Status target |
| --- | --- | --- |
| Today (23 Oct) | Steps 1 and 2: data, non-IID split, baseline model | Done |
| Tomorrow (24 Oct) | Step 3: Flower federation across 5 banks, then Step 6: a basic Kafka pipeline. Submit resume once both work | Done |
| Day after (25 Oct) | Step 4 (privacy) and Step 5 (comparison results) | Done, about 70% |
| Following week | Step 7 dashboard, then Step 8 extensions | Ongoing |

## Resume Note

List the project as **"Aavaran: Federated Fraud Detection (Ongoing)"**. Describe only the parts that are actually built and working when the resume is submitted, with real numbers from your experiments. Talk about the planned features in the interview as next steps. Interviewers will ask detailed questions, and being able to explain everything you've listed is what makes the project impressive.

## How to Run (Planned)

```bash
pip install -r requirements.txt
python src/preprocess.py
python src/server.py          # terminal 1
python src/client.py --bank 1 # terminals 2 to 6, banks 1 to 5
docker compose up -d          # start the Kafka broker
python src/scorer.py          # score the live stream
python src/producer.py        # replay transactions into Kafka
streamlit run dashboard/app.py
```

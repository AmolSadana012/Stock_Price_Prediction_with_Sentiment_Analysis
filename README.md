# Stock Return Prediction - Market Historical Data

The market-data pipeline predicts next-day close-to-close returns for AAPL,
MSFT, and AMZN using exactly three models: LSTM, GRU, and Hybrid LSTM-GRU.
The hybrid combines parallel 64-unit LSTM and GRU branches and has return and
UP/DOWN classification heads. The existing dataset, feature engineering,
splitting, scaling, and training methodology are unchanged.

## Setup

Install dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
```

## Run models from the project root

```powershell
python main.py --model lstm
python main.py --model gru
python main.py --model hybrid_lstm_gru
```

To rerun the validation-only hybrid selection and test comparison after the
baseline predictions exist:

```powershell
python -m Market_historical_data.experiments.hybrid_lstm_gru
```

## Tests and checks

Run all tests from the project root:

```powershell
python -m unittest discover -s Market_historical_data/tests -v
```

Compile the market-data package:

```powershell
python -m compileall -q Market_historical_data main.py
```

## Project structure

```text
Major_GR41/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── Market_historical_data/
    ├── data/                 Yahoo Finance download and cleaning
    ├── preprocessing/        Features, chronological splits, scaling, windows
    ├── models/               LSTM, GRU, Hybrid LSTM-GRU and shared trainer
    ├── experiments/          Validation selection and test comparison workflow
    ├── tests/                Offline unit and output-path tests
    ├── utils/                Shared visualization helpers
    └── outputs/
        ├── comparison/       model_comparison.csv and final_summary.csv
        ├── lstm/              LSTM checkpoints, predictions, representations,
        │                      plots, metadata, and metrics.csv
        ├── gru/               GRU outputs in the same model-specific layout
        ├── hybrid_lstm_gru/   Hybrid outputs, including losses/
        └── experiments/       Selection artifacts and preserved prior results
```

Yahoo Finance raw downloads and processed datasets remain under
`Market_historical_data/data/`. Running a model saves results only under that
model's directory and updates the shared comparison CSVs without replacing
other models' rows. Historical model checkpoints, predictions, metrics, and
experiment records have been retained.

.PHONY: setup reproduce eval label clean download-data

# ── Setup ──────────────────────────────────────────────────────────────
setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "✅ Setup complete. Activate with: source .venv/bin/activate"
	@echo "   Then copy .env.example → .env and add at least one API key."

# ── Download dataset (requires kaggle CLI + credentials) ─────────────
download-data:
	.venv/bin/python scripts/download_dataset.py

# ── Full data pipeline (Phase 1): needs data/twcs.csv ─────────────────
data-prep:
	.venv/bin/python -m src.data_prep

# ── Reproduce headline numbers from cache (<15 min) ──────────────────
reproduce:
	PYTHONHASHSEED=0 .venv/bin/python eval/run_eval.py
	@echo ""
	@echo "✅ Results written to results/metrics.json and results/eval_examples.csv"

# ── Fresh evaluation (calls LLM APIs, slower) ────────────────────────
eval-fresh:
	.venv/bin/python eval/run_eval.py --fresh

# ── Run labeling tool ─────────────────────────────────────────────────
label:
	.venv/bin/streamlit run scripts/label_golden_set.py

# ── Sample golden set ─────────────────────────────────────────────────
sample:
	.venv/bin/python scripts/sample_golden_set.py

# ── Run tests ─────────────────────────────────────────────────────────
test:
	.venv/bin/pytest tests/ -v

# ── Clean generated outputs ──────────────────────────────────────────
clean:
	rm -rf results/metrics.json results/eval_examples.csv
	rm -rf results/cache/*.json
	@echo "Cleaned results/ (data/ and golden set preserved)"

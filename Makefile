.PHONY: setup run-core run-extended analyze report clean test validate

# Configuration
VENV := /opt/isb1/venv
PYTHON := $(VENV)/bin/python
OUTPUT_DIR := results
SWEEP_CONFIG := configs/sweep/core.yaml

setup:
	@echo "Setting up ISB-1..."
	bash scripts/setup_node.sh
	bash scripts/install_telemetry.sh
	bash scripts/download_datasets.sh
	$(PYTHON) scripts/generate_mode_a_configs.py
	$(PYTHON) scripts/generate_traces.py
	@echo "Setup complete."

validate:
	$(PYTHON) -m harness.config_validator --sweep $(SWEEP_CONFIG)

run-core:
	$(PYTHON) -m harness.sweep --config $(SWEEP_CONFIG) --output $(OUTPUT_DIR)

run-core-dry:
	$(PYTHON) -m harness.sweep --config $(SWEEP_CONFIG) --output $(OUTPUT_DIR) --dry-run

run-extended:
	$(PYTHON) -m harness.sweep --config configs/sweep/extended.yaml --output $(OUTPUT_DIR)

analyze:
	$(PYTHON) -m analysis.aggregate --input $(OUTPUT_DIR)/raw --output $(OUTPUT_DIR)/aggregated
	$(PYTHON) -m analysis.claim_evaluator --input $(OUTPUT_DIR)/aggregated --output $(OUTPUT_DIR)/claims

leaderboard:
	$(PYTHON) -m analysis.leaderboard --input $(OUTPUT_DIR)/aggregated --output $(OUTPUT_DIR)/leaderboard

report:
	$(PYTHON) -m analysis.plots.throughput_latency --input $(OUTPUT_DIR)/aggregated --output publication/figures
	$(PYTHON) -m analysis.plots.leaderboard_heatmap --input $(OUTPUT_DIR)/leaderboard --output publication/figures

quality:
	$(PYTHON) -m quality.rouge_eval --reference quality/reference_outputs --test $(OUTPUT_DIR)/raw
	$(PYTHON) -m quality.humaneval_runner --model-url http://localhost:8000

clean:
	rm -rf $(OUTPUT_DIR)/raw/* $(OUTPUT_DIR)/aggregated/* $(OUTPUT_DIR)/claims/* $(OUTPUT_DIR)/leaderboard/*

test:
	$(PYTHON) -m pytest tests/ -v

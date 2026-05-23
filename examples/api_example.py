from at_gan import api as gan

# Run W&B Sweep
print("Launching Architecture Sweep...")
sweep_id = gan.sweep(
    base_config="configs/test_config.yaml",
    sweep_config="configs/test_sweep.yaml",
    count=50
)
print(f"Sweep {sweep_id} finished analyzing 50 combinations.")

# Run GAN training run
print("Training GAN...")
engine = gan.train(
    config="configs/test_config.yaml",
    enable_wandb=False,
    export_generator=True,
    generate_samples=0
)

# Generate synthetic samples
print("Generating 100,000 synthetic samples...")
synthetic_df = gan.generate(
    config_path="configs/test_config.yaml",
    run_id="offline_run",
    samples=100000,
    output_path="data/framingham_synthetic_100k.csv"
)


# Run evaluation suite
print("Running Synthetic Data Evaluation Suite...")
metrics = gan.evaluate(
    real_data_path="datasets/framingham_real.csv",
    synthetic_data_path="data/framingham_synthetic_100k.csv",
    target_column="diabetes"
)

# Print evaluation metrics
min_dcr = metrics["dcr"]["dcr_min"]
print(f"Privacy (Min DCR): {min_dcr:.4f}")

sdv_score = metrics["sdv"]["sdv_overall_score"] * 100
print(f"Fidelity (SDV Score): {sdv_score:.2f}%")

if metrics.get("tstr"):
    utility = metrics["tstr"]["utility_retention"]
    print(f"Utility (F1 Retention): {utility:.2f}%")
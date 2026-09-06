from pathlib import Path
import csv
from ultralytics import YOLO

# File paths
MODEL10_DIR = Path(__file__).resolve().parent
MODELS_DIR = MODEL10_DIR.parent
MODEL6_DIR = MODELS_DIR / "model_06_nd11_4_slope_tpi_yolo"

# Use the corrected common test set for EVERY run.
DATA = MODEL10_DIR / "dataset" / "dataset.yaml"

# Model 6 and Model 10 under three seeds each
runs = [
    ("model6", 123, MODEL6_DIR / "checkpoints" / "model_06_seed123_best.pt"),
    ("model6", 456, MODEL6_DIR / "checkpoints" / "model_06_seed456_best.pt"),
    ("model10", 123, MODEL10_DIR / "checkpoints" / "model_10_seed123_best.pt"),
    ("model10", 456, MODEL10_DIR / "checkpoints" / "model_10_seed456_best.pt"),
]

results = []

# Evaluate every run on the same corrected test set
for model_name, seed, checkpoint in runs:
    print("\n" + "=" * 70)
    print(f"Evaluating {model_name}, seed {seed}")
    print("Checkpoint:", checkpoint)
    print("=" * 70)

    model = YOLO(str(checkpoint))

    metrics = model.val(
        data=str(DATA),
        split="test",
        imgsz=512,
        batch=8,
        device=0,
        workers=0,
        plots=False,
        project=str(MODEL10_DIR / "runs"),
        name=f"{model_name}_seed{seed}_common_test",
        exist_ok=True,
    )

    row = {
        "model": model_name,
        "seed": seed,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }

    results.append(row)

    print(
        f"{model_name} seed {seed}: "
        f"P={row['precision']:.4f}, "
        f"R={row['recall']:.4f}, "
        f"mAP50={row['mAP50']:.4f}, "
        f"mAP50-95={row['mAP50_95']:.4f}"
    )

# Save the results
reports = MODEL10_DIR / "reports"
reports.mkdir(exist_ok=True)

out = reports / "seed_robustness_results.csv"

with open(out, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "seed", "precision", "recall", "mAP50", "mAP50_95"],
    )
    writer.writeheader()
    writer.writerows(results)

print("\nFINAL RESULTS")
print("=" * 70)
for r in results:
    print(
        f"{r['model']:8s} seed {r['seed']:3d} | "
        f"P {r['precision']:.4f} | "
        f"R {r['recall']:.4f} | "
        f"mAP50 {r['mAP50']:.4f} | "
        f"mAP50-95 {r['mAP50_95']:.4f}"
    )

print("\nSaved to:", out)

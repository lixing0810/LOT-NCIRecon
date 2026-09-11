"""CSV and LaTeX summaries for selected OT triplets."""

import csv
import os


def save_selection_summary(summary, output_root, train_size, score_threshold):
    os.makedirs(output_root, exist_ok=True)
    csv_path = os.path.join(output_root, "selection_summary.csv")
    row = {
        "train_size": train_size,
        "score_threshold": score_threshold,
        **summary,
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return csv_path

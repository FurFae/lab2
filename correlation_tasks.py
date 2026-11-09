"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values="NA")
    df = df.dropna(subset=FEATURES + ["MRI_Count"])

    def group_correlations(gender: str) -> dict[str, float]:
        group = df[df["Gender"] == gender]
        return {f: float(group[f].corr(group["MRI_Count"])) for f in FEATURES}

    women_corr = group_correlations("Female")
    men_corr = group_correlations("Male")

    strongest = max(women_corr | men_corr, key=lambda f: max(abs(women_corr[f]), abs(men_corr[f])))

    return BrainCorrelationSummary(
        men_count=int((df["Gender"] == "Male").sum()),
        women_count=int((df["Gender"] == "Female").sum()),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest,
    )

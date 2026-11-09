from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput
from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations

t = analyze_titanic(TitanicInput("titanic.csv"))
print("rows:", t.row_count)
print("missing:", t.missing_by_column)
print("over30:", t.adults_over_30_count)
print("mean age:", t.mean_age_by_pclass)
print("survival:", t.survival_rate_by_pclass)
print("top fares:", t.highest_fares)
assert t.row_count == 891
assert t.missing_by_column["Age"] == 177
assert t.missing_by_column["Cabin"] == 687
assert t.highest_fares == sorted(t.highest_fares, reverse=True)
assert abs(sum(t.survival_rate_by_pclass.values()) - (0.63 + 0.47 + 0.24)) < 0.05

b = analyze_brain_correlations(BrainDataInput("brainsize.txt"))
print("men:", b.men_count, "women:", b.women_count)
print("women corr:", b.women_mri_correlation)
print("men corr:", b.men_mri_correlation)
print("strongest:", b.strongest_mri_feature)

print("LAB2 OK")

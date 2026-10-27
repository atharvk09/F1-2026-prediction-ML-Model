# Formula 1 Race Winner Prediction

## Project overview

This project predicts Formula 1 race winners using historical race results,
qualifying performance, team performance, and recent driver form.

The project compares three models:
- Model A: Logistic Regression baseline
- Model B: Recency-weighted adaptive Logistic Regression
- Model C: LightGBM

## Dataset

- Historical seasons: 2022–2025
- Historical training dataset: 1,823 rows
- Historical evaluation dataset: 1,803 rows
- 2026 evaluation: 16 rounds, 22 drivers per round

The 2026 comparison contains 352 driver-round predictions per model.

## Features

The models use five pre-race features:

1. Grid position
2. Qualifying gap percentage
3. Indicator for missing qualifying data
4. Team season points per race
5. Driver's average finishing position across previous races in the current season

The features are intended to represent information available before the race.
Historical driver and team form are calculated using prior results.

## Models

### Model A: Logistic Regression

A baseline model using median imputation, feature standardization,
and logistic regression.

### Model B: Adaptive Logistic Regression

A logistic regression model that gives more weight to recent historical
seasons and incorporates earlier 2026 results when predicting later rounds.

### Model C: LightGBM

A gradient-boosted decision tree classifier using the same five features.
For each 2026 round, the experiment incorporates only earlier 2026 results,
along with weighted historical data.

## Historical evaluation

Chronological evaluation trained on earlier seasons and tested on later seasons.

| Model | Test season | Correct winners | Races | Accuracy | Race-level log loss |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 2023 | 19 | 22 | 86.36% | 0.8870 |
| LightGBM | 2023 | 17 | 22 | 77.27% | 0.6963 |
| Logistic Regression | 2024 | 11 | 24 | 45.83% | 1.7493 |
| LightGBM | 2024 | 12 | 24 | 50.00% | 1.9829 |
| Logistic Regression | 2025 | 12 | 23 | 52.17% | 1.2358 |
| LightGBM | 2025 | 14 | 23 | 60.87% | 0.9735 |

Across the 69 historical test races:
- Logistic Regression: 42 correct winners, 60.87% overall accuracy.
- LightGBM: 43 correct winners, 62.32% overall accuracy.

The historical results show a small overall accuracy advantage for LightGBM,
but performance varies across seasons.

## 2026 retrospective comparison

The following comparison covers the same 16 rounds.

| Metric | Model A: Logistic Regression | Model B: Adaptive Logistic | Model C: LightGBM |
|---|---:|---:|---:|
| Correct winners | 9/16 | 11/16 | 10/16 |
| Winner accuracy | 56.25% | 68.75% | 62.50% |
| Race-level log loss | 1.4770 | 1.6982 | 1.1473 |
| Total podium hits | 29/48 | 30/48 | 31/48 |
| Average actual podium drivers in predicted top three | 1.812 | 1.875 | 1.938 |

All three models' normalized win probabilities sum to 100% within each evaluated race.

## Interpretation

The adaptive logistic regression model had the highest winner accuracy in the
16-round 2026 comparison. LightGBM had the lowest race-level log loss and
the highest podium overlap.

These metrics measure different aspects of performance. Winner accuracy
measures how often the model ranked the actual winner first. Race-level
log loss measures the probability assigned to the actual winner, with larger
penalties when that probability is very low.

The results do not establish that one model will always outperform the others.
The 2026 results have already been examined during model development, so this
comparison is retrospective rather than a completely untouched final test.

Normalizing driver scores within a race makes the probabilities sum to 100%,
but does not guarantee probability calibration.

## Current conclusion

Keep the adaptive logistic regression as the primary winner-prediction model
for now because it achieved the highest winner accuracy in the 2026 comparison.
Retain LightGBM as an alternative because of its lower log loss and slightly
higher podium overlap.

Further evaluation on future races is needed before concluding that either
model generalizes better.

## Potential next steps

1. Preserve a future race or season as an untouched test set.
2. Evaluate probability calibration.
3. Compare against simple baselines such as the pole sitter and the
   highest-ranked qualifying driver.
4. Investigate errors by circuit, grid position, and team.
5. Reassess the models as additional race results become available.

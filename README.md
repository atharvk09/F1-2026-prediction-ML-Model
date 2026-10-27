# Formula 1 Race Winner Prediction

A machine learning project that predicts Formula 1 race winners using historical race results, qualifying performance, and pre-race team and driver form.

## Project Overview

This project compares three classification approaches:

- **Logistic Regression:** A baseline model using historical race and qualifying features.
- **Adaptive Logistic Regression:** Gives more weight to recent seasons and previously completed races in the current season.
- **LightGBM:** A gradient-boosting model compared against logistic regression.

Predictions use information available before each race. The current race's finishing position is used for evaluation, not as a prediction feature.

## Features

- Grid position
- Qualifying gap to the fastest qualifying time
- Indicator for missing qualifying data
- Team's average season points per race before the current race
- Driver's average finishing position across their previous three races

## Dataset

The project includes historical race results from 2022 through 2025, qualifying data from those seasons, and 2026 race and qualifying data.

## 2026 Model Comparison

| Model | Winner Accuracy | Race-Level Log Loss | Average Podium Hits |
|---|---:|---:|---:|
| Logistic Regression | 56.25% | 1.4770 | 1.812/3 |
| Adaptive Logistic Regression | 68.75% | 1.6982 | 1.875/3 |
| LightGBM | 62.50% | 1.1473 | 1.938/3 |

Winner accuracy measures how often the model ranked the actual winner first. Average podium hits measures the number of actual podium finishers included in the predicted top three, averaged across races. Lower race-level log loss indicates a better-scoring probability distribution.

The 2026 comparison is retrospective because these races were examined during model development. It is not an untouched test of future performance. Normalizing predictions to sum to 100% does not guarantee probability calibration.

## Project Structure

- `notebooks/01_explore_race.ipynb`: Main exploration, modeling, and evaluation notebook
- `data/`: Raw and processed race data, qualifying data, and comparison results
- `model_report.md`: Detailed project report

## Limitations

Race outcomes depend on factors not necessarily captured in the features, including weather, reliability, strategy, safety cars, and incidents. Performance can change between seasons. Future untouched testing is needed to assess generalization.

## Future Improvements

- Add weather, circuit, and circuit-specific performance features.
- Evaluate probability calibration.
- Test on future races not used during model development.
- Automate data collection and predictions.

## Technologies

Python, pandas, NumPy, scikit-learn, LightGBM, Jupyter Notebook, and FastF1.

See `model_report.md` for the detailed project report.

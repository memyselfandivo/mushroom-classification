# Mushroom Classification – zero kills, minimal waste

A scikit-learn project on **asymmetric classification**: predict whether a mushroom is poisonous from six visible features, without ever labelling a poisonous mushroom as edible, and while throwing away as few edible mushrooms as possible.

The project was built for an internal competition in a data science course. Submissions were ranked by the number of "deaths" (false negatives) first and by "edible but uneaten" (false positives) second.

## Result

| Upload | Model | Deaths (FN) | Edible but uneaten (FP) |
|---|---|---|---|
| 1 | Random forest, threshold tuned with a cost function | 2 | 89 |
| 2 | **Random forest, `min_samples_leaf=2`, threshold 0.10** | **0** | **95** |
| 3 | Random forest, threshold lowered by hand to 0.05 | 2 | 92 |

The final model reached **0 deaths and 95 uneaten** out of 1625 competition mushrooms. For all three uploads the number of uneaten mushrooms matched a formula derived beforehand (uneaten = 866 − predicted edible + deaths), and the result of the final model was predicted exactly.

## Key findings

- **The data has a hard limit.** Six features produce only 378 distinct feature combinations. In 18 of them, poisonous and edible mushrooms look identical. Zero kills therefore cost roughly 80 edible mushrooms, whatever the model.
- **The threshold matters more than the model family.** At the default threshold of 0.5 every model tested produced kills, while accuracy looked fine (0.959).
- **Blind spots are the real risk.** A model that trusts a combination after seeing a single edible example will eventually meet a poisonous mushroom that looks the same. Cross-validation and the hold-out set announced this, and the first upload confirmed it.
- **`min_samples_leaf=2` fixed what a lower threshold could not.** Lowering the threshold changes nothing when the forest has no doubt at all. The doubt only appears once rare combinations are judged together with their neighbours.
- **Zero kills cannot be guaranteed** with these six features, only made likely. A clean leaderboard result does not prove that a model is safe.

## Approach

1. **Data exploration** – poison rate per category and an analysis of identical-looking mushrooms.
2. **Pipeline** – imputer, one-hot encoder, classifier. The six categorical features become 42 columns.
3. **Baseline** – decision tree at threshold 0.5.
4. **Experiments** – four model families, one-hot versus ordinal encoding, grid search with `roc_auc`, and a stability check over 15 different fold splits.
5. **Threshold** – ROC curve, a custom cost function, `TunedThresholdClassifierCV` and nested cross-validation, including a sensitivity analysis of the cost ratio.
6. **Finalists** – compared with repeated 10-fold cross-validation (50 runs) and one look at a 20 % hold-out set.
7. **Second round** – after the first upload produced two deaths, the random forest was re-tuned with a stricter criterion: zero kills in all 50 cross-validation runs first, then as few mushrooms thrown away as possible.

## Final model

```
SimpleImputer(strategy="most_frequent")
→ OneHotEncoder(handle_unknown="ignore")
→ RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=42)
```

A mushroom is labelled poisonous when its predicted poison probability is above **0.10**.

## Repository structure

```
├── mushroom_competition.ipynb   # the full analysis, with outputs
├── data/
│   ├── train.csv                # 6499 labelled mushrooms
│   ├── test.csv                 # 1625 competition mushrooms, no labels
│   └── sample_submission.csv    # defines the submission format
├── submissions/                 # one file per model variant (A to E)
├── requirements.txt
├── LICENSE
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook mushroom_competition.ipynb
```

The notebook runs top to bottom in two to three minutes and rewrites the files in `submissions/`. It needs scikit-learn 1.5 or newer for `TunedThresholdClassifierCV`.

Tested with Python 3.13, scikit-learn 1.7.2, pandas 2.3.3, numpy 2.3.5, matplotlib 3.10.6 and seaborn 0.13.2.

## Data

The data is a subset of the [UCI Mushroom dataset](https://archive.ics.uci.edu/dataset/73/mushroom), reduced to six features that are visible to the naked eye:

| Feature | Categories |
|---|---|
| `cap.shape` | bell, conical, convex, flat, knobbed, sunken |
| `cap.color` | brown, buff, cinnamon, gray, green, pink, purple, red, white, yellow |
| `bruises` | bruises, no bruises |
| `stalk.color.above.ring` | brown, buff, cinnamon, gray, orange, pink, red, white, yellow |
| `stalk.color.below.ring` | brown, buff, cinnamon, gray, orange, pink, red, white, yellow |
| `population` | abundant, clustered, numerous, scattered, several, solitary |

The target `poisonous` is 1 for poisonous and 0 for edible.

Source: Mushroom [Dataset]. (1981). UCI Machine Learning Repository. https://doi.org/10.24432/C5959T

## Limitations

- The dataset describes gilled mushrooms from a North American field guide. Nothing here is advice on which real mushrooms are safe to eat.
- The differences between the best model variants rest on a handful of individual mushrooms. The cross-validation figures show a tendency, not a certainty.
- The number of edible mushrooms in the competition set (866) was derived from the public score of the sample submission and used only to predict results before uploading, not to tune the model.

## License

The code in this repository is released under the [MIT License](LICENSE). The data originates from the UCI Mushroom dataset and remains subject to its original licence terms.

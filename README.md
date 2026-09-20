# Four Runners: Marathon Profiles and Performance

Sport Data Analytics project based on the **Run Club Marathon Performance Dataset**, a synthetic dataset containing 80,000 simulated runners.

*What separates a fast marathon runner from a slow one — and can a synthetic dataset tell us?*

## Start here

The final data story, including the complete narrative and all visualizations, is available in:

➡️ **[10_four_runners_full_story.ipynb](notebooks/10_four_runners_full_story.ipynb)**

This is the main deliverable. The other notebooks document the supporting data preparation, profiling and modelling work.

The same story is also published as an interactive web app, where the charts respond to the reader:

➡️ **Live app:** [Four Runners — the interactive story](https://sda-marathon-profiles-p3angtmevmrbminnenlgrt.streamlit.app/)

The app is a companion, not a replacement — the notebook remains the full write-up.

## Research questions

**1. Can distinct runner profiles be identified** based on training behaviour, physical condition, recovery and psychological preparation, and how do these profiles differ in marathon performance?

**2. Can this dataset support those conclusions at all?**

The project answers the first question in full: four profiles emerge, the extremes finish 51 minutes apart, and the payoff from experience is steep early and flattens later.

It then answers the second question with a no — and that turn is the main finding. The dataset is a formula, not a record of running: finish time is mostly experience, plus a handful of small adjustments, plus about 21 minutes of pure randomness. Most of what the file measures has no bearing on the result, and weekly mileage — one of the strongest predictors in real marathon research — barely matters here.

Both answers matter. The shapes the dataset produces are realistic; the relationships inside it are not.

## Story overview

The final notebook introduces four descriptive runner profiles:

- **Newcomers**
- **Regulars**
- **Naturals**
- **Veterans**

It then compares their finish times and drop-out rates, examines the relationship between experience and performance, and explores other factors associated with finish time.

Section 5 turns the analysis on the dataset itself and works out how the file was generated. This is the payoff of the story, and it reframes everything before it: Sections 1 to 4 describe the file accurately, but the file turns out to be describing itself rather than marathon running.

The profiles should be understood as regions along a continuum rather than four naturally separated kinds of runner.

## Important limitation

All observations and results describe a **simulated population**, not real marathon runners. The project is descriptive and should not be interpreted as training advice or as evidence of real-world causal relationships.

## Repository structure

- `data/` — source, cleaned and profile-assignment datasets
- `figures/` — exported visualizations
- `notebooks/01_data_review.ipynb` — initial data review
- `notebooks/02_cleaning.ipynb` — data cleaning
- `notebooks/03_profiles.ipynb` — runner profiling and cluster validation
- `notebooks/04_models.ipynb` — profile-level modelling
- `notebooks/05_report_model.ipynb` — interpretable performance model
- `notebooks/06_generator.ipynb` — synthetic-data generator investigation
- `notebooks/07_storytelling.ipynb` — initial storytelling structure
- `notebooks/08_sections_1_2.ipynb` — opening story sections
- `notebooks/09_section_5.ipynb` — synthetic-data credibility section
- `notebooks/10_four_runners_full_story.ipynb` — **final complete story**
- `streamlit_app.py` — the interactive version of the final story
- `requirements.txt`, `.streamlit/config.toml` — app dependencies and theme

## Reproducing the final notebook

The analysis uses Python with the following main packages (`seaborn >= 0.13` and
`matplotlib >= 3.6` are required):

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

Clone or download the repository, start Jupyter from the repository root, open `notebooks/10_four_runners_full_story.ipynb`, and run:

`Kernel → Restart & Run All`

The notebook expects `data/clean.csv` and `data/profiles.csv` to remain in the repository's `data/` directory.

## Running the interactive app

From the repository root:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app reads the same `data/clean.csv` and `data/profiles.csv` and refits the same models, so its
numbers and the notebook's always agree. The first load takes around 20 seconds while the
Section 5 models are fitted; after that everything is cached.

### Deploying it

The app is deployed on [Streamlit Community Cloud](https://share.streamlit.io):

1. Sign in with the GitHub account that has access to this repository.
2. **Create app** → pick `PaulaGili/sda-marathon-profiles`, branch `main`, main file `streamlit_app.py`.
3. Choose a subdomain, deploy, and paste the resulting link into **Start here** above.

Free-tier apps sleep after a period of inactivity, so the first visitor after a quiet spell waits
roughly half a minute for the app to wake and refit its models.

## Team

Mayra, Paula, Petra and Ariella  
HSLU — Sport Data Analytics, FS26

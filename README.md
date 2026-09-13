# Four Runners: Marathon Profiles and Performance

Sport Data Analytics project based on the **Run Club Marathon Performance Dataset**, a synthetic dataset containing 80,000 simulated runners.

## Start here

The final data story, including the complete narrative and all visualizations, is available in:

➡️ **[10_four_runners_full_story.ipynb](notebooks/10_four_runners_full_story.ipynb)**

This is the main deliverable. The other notebooks document the supporting data preparation, profiling and modelling work.

## Research question

Can distinct runner profiles be identified based on training behaviour, physical condition, recovery and psychological preparation, and how do these profiles differ in marathon performance?

## Story overview

The final notebook introduces four descriptive runner profiles:

- **Newcomers**
- **Regulars**
- **Naturals**
- **Veterans**

It then compares their finish times and drop-out rates, examines the relationship between experience and performance, explores other factors associated with finish time, and investigates how the synthetic dataset appears to have been generated.

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

## Reproducing the final notebook

The analysis uses Python with the following main packages:

```bash
pip install numpy pandas matplotlib scipy scikit-learn jupyter
```

Clone or download the repository, start Jupyter from the repository root, open `notebooks/10_four_runners_full_story.ipynb`, and run:

`Kernel → Restart & Run All`

The notebook expects `data/clean.csv` and `data/profiles.csv` to remain in the repository's `data/` directory.

## Team

Mayra, Paula, Petra and Ariella  
HSLU — Sport Data Analytics, FS26

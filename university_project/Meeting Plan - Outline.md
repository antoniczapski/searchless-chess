**Meeting Plan**

**Meeting 1 (Feb 24) - Project Introduction & DevOps Requirements**

We will discuss:

* Project goals and grading
* Engineering requirements
* Reproducibility standards
* Example repository structure
* Common mistakes to avoid

After this meeting:
Start exploring datasets and thinking about project ideas.

**Meeting 2 (March 3) - Team Formation & Problem Discussion**

During the meeting each team should briefly present:

* Problem idea
* Dataset source
* Why Deep Learning is appropriate
* Proposed evaluation metric

After the meeting (short written proposal, max 1 page):

* Problem description
* Dataset link
* Target variable
* Metric
* Planned model architecture

**Meeting 3 (March 10) - Proposal Presentation (Graded)**

Each team presents (5-7 minutes):

* Problem definition
* Dataset description
* Evaluation metric
* Planned architecture
* Main risks or challenges

Must include:
After approval, the project scope is fixed.

**Meeting 4 (March 17) - Repository & CI Review**

You must show:

* Public GitHub repository
* Proper project structure (`src/`, `tests/`, `configs/`)
* Working `uv` environment
* Working GitHub Actions (CI)
* Basic tests implemented

CI must pass.

**Meeting 5 - EDA & Baseline Model**

You present:

* Exploratory Data Analysis (EDA)
* Key dataset insights
* Baseline model results

Baseline should be simple but meaningful.

**Meeting 6 - Reproducible Training & W&B**

You must demonstrate:

* Config-based training
* Fixed random seeds
* Fixed train/validation/test splits
* W&B experiment tracking
* At least 2-3 logged experiments

Your experiments must be reproducible.

**Meeting 7 - First Experimental Results**

You present:

* Model comparison
* Performance results
* Observations
* Next improvement plan

Be prepared to justify your decisions.

**Meeting 8 - Model Improvements**

You should show:

* Improvements over baseline
* Hyperparameter tuning or architectural changes
* Clear before/after comparison

**Meeting 9 - Error Analysis**

You must present:

* Where the model fails
* Confusion matrix (if classification)
* Failure examples
* Overfitting/underfitting analysis

Understanding model weaknesses is essential.

**Meeting 10 - Advanced Experiments**

Examples:

* Ablation study
* Removing augmentation
* Comparing optimizers
* Regularization study
* Robustness tests

You must analyze results, not just report numbers.

**Meeting 11 - Final Review & Debugging**

Checklist:

* CI passing
* Clean repository
* Clear README
* Working training pipeline
* Reproducibility verified
* W&B experiments organized
* Presentation draft ready

Ability to explain your decisions is crucial.

**Meeting 12 - Final Presentations**

15-20 minutes per team.

Your presentation must include:

1. Problem and motivation
2. Dataset
3. Model architecture
4. Experimental comparison
5. Error analysis
6. Engineering decisions
7. Conclusions and limitations

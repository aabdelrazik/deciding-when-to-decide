# deciding-when-to-decide

Code for *Deciding when to decide: how recency, urgency, risk, and bias shape
human sequential decision-making, a case study across the obsessive-compulsive
spectrum*.

We model people deciding when to stop gathering evidence and commit, as a
Partially Observable Markov Decision Process, and add a small set of
interpretable departures from optimality: a biased prior, transient
exaggeration of the newest evidence, progressive forgetting of older evidence,
an inflated cost of being wrong, patience and urgency, and misperception of the
deadline. Each is a parameter of the same generative model.

Licence: MIT. See `LICENSE`.

## Start here

`notebooks/example_usage.ipynb` is a primer that runs on a laptop in a couple of
minutes. It builds each model variant, solves it, draws the policy, and replays
a card sequence, none of which needs the dataset. The last section fits one real
subject, and the sections that read fitted results skip with a message when
those are absent, so the notebook runs top to bottom on a fresh clone. Read it
before the pipeline below.

## Install

Two supported routes. The container is the reliable one, since it pins every
version the reported results were produced with.

**Container**

```bash
git clone https://github.com/aabdelrazik/deciding-when-to-decide-internal
cd deciding-when-to-decide-internal
apptainer build pomdp_image.sif pomdp_image.def          # or singularity
apptainer exec --bind "$PWD:$PWD" --pwd "$PWD" pomdp_image.sif python3 scripts/<script>.py
```

The definition installs `requirements.txt` and then checks that every package
imports, so a broken environment fails at build time instead of hours into a
fitting job. Pass `--env PYTHONNOUSERSITE=1` if you have a `~/.local` that might
shadow the container's packages.

**Native**

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 scripts/<script>.py
```

3.12 is the version the reported results were produced with and the one the
container pins. Anything from 3.10 works.

Scripts resolve the repository root from their own location, so they run from
any working directory. Set `POMDP_ROOT` to override.

## Get the data

The behavioural data are not ours to redistribute. They were collected and
first reported by del Rio and colleagues, *Indecision and recency-weighted
evidence integration in non-clinical and clinical settings*, Nature Human
Behaviour 10(4):727-740, 2026, doi:10.1038/s41562-025-02385-1, and are public on
OSF.

Go to https://osf.io/fks97/, open **OSF Storage**, then **TrHu_NHB**, then
**data_MEG**, and take these three files:

```
data/TrHu_NHB_light/data_MEG/behdat.mat          860 KB
data/TrHu_NHB_light/data_MEG/fa_scores.csv        11 KB
data/TrHu_NHB_light/data_MEG/ybocs_scores.csv    2.4 KB
```

Take the three rather than the folder: `data_MEG` also holds the MEG decoding
outputs, one of which is 4 GB, and none of them is read here. No MATLAB either.
The `.m` files are the original authors' own analyses and are never called; the
`.mat` files are read with scipy.

**One file is not on OSF.** `sNdraws8.mat` contains the rewards received by each 
participant. These data are not included in the OSF deposit, yet all results 
measured in points depend on them. We obtained the file from Prof. Tobias Hauser 
and have included it here for the reviewers’ use only. We plan to make it 
available to readers in the future. For reviewers, a password-protected archive 
is available at:
`data/TrHu_NHB_light/data_MEG/sNdraws8_encrypted.zip`

### Decryption Instructions
The password is provided in the **"Data Availability"** section of the submission portal.

* **macOS:** Double-click `sNdraws8_encrypted.zip` and enter the password.
* **Windows:** Right-click `sNdraws8_encrypted.zip` > **Extract All...** > enter the password.
* **Linux / CLI:** Run `cd data/TrHu_NHB_light/data_MEG/ && unzip sNdraws8_encrypted.zip`

Ensure the extracted `sNdraws8.mat` file remains in `data/TrHu_NHB_light/data_MEG/` so scripts run automatically.

## Preprocess, in this order

```bash
python3 scripts/preprocess/preprocess_01_convert_mat.py     # behdat.mat -> behdat.pkl
python3 scripts/preprocess/preprocess_02_add_reward.py      # + sNdraws8.mat -> behdat_reward.pkl
python3 scripts/preprocess/preprocess_03_build_evidence.py  # -> behdat_preprocessed.pkl + evidence dicts
```

**Step 2 is not optional.** `behdat.mat` does not carry the reward a participant
received. That lives in `sNdraws8.mat`, and step 2 matches the two datasets
subject by subject and derives the reward from it. Everything measured in points
depends on this, so skipping it gives results that look plausible and are wrong.
The step prints how many subjects it matched; it should be all 105.

Forgetting models additionally need pre-computed grids of discounted evidence:

```bash
python3 scripts/preprocess/generate_gamma_grids.py
```

## Fit

One model is one config in `data/simulation_configs/`. There are 79 of them,
plus 73 `_commit.py` siblings, one for every model that enters a comparison
against the GLM. The name encodes which mechanisms are on:
a horizon prefix, then twelve slots, uppercase for fitted, lowercase for fixed
on, and a dash for off. So `SB-XT-RPh----` fits belief bias, temperature, lapse
and subjective cost, holds the hazard on, and has no recency mechanism.

```bash
SIM_CONFIG_PATH=$PWD/data/simulation_configs/simulation_params_SB-XT-RPh----.py \
SIM_N_JOBS=8 python3 scripts/fit/fit_data.py
```

`SIM_N_JOBS` sets how many subjects are fitted in parallel through joblib, so a
workstation with N cores fits N at once. Slurm is convenient but not required;
`slurm/` holds one job script per stage.

**The job scripts are written for one cluster and need three edits for another.**
They were used on Viper at the Max Planck Computing and Data Facility, so every
one of them carries:

```
#SBATCH --partition general        # your partition name
#SBATCH --cpus-per-task=128        # a whole node there; match your own
#SBATCH --exclusive=user           # site policy, drop it if yours differs
```

Change all three at once with, for example:

```bash
sed -i 's/--partition general/--partition YOURS/; s/--cpus-per-task=128/--cpus-per-task=32/; /--exclusive=user/d' slurm/*.sh
```

The `module load apptainer` line already tolerates a site that has no such
module, so nothing needs changing there if apptainer or singularity is on the
path. 

Forgetting models use `scripts/fit/fit_data_forgetting.py` instead.

**One pass does not reproduce the deposited fits exactly.** The reported fits are
a multi start estimate: a first pass, a second seeded from the nested model's
solution, and `merge_de_runs.py` keeping whichever fit is better per subject. Run the seeded pass in section 3.1 of `PIPELINE.txt` to close the gap; without it the numbers are close but not the published ones.

**Full and commit likelihoods.** The manuscript uses two. The full likelihood
scores the three way choice, commit yellow, commit blue, or wait, and produces
the main model comparison. The commit likelihood scores only decide against
wait, and is used wherever the POMDP is compared with the GLM, because the GLM
predicts only whether a participant commits on a given draw. Every model has a
`_commit.py` sibling config that flips the flag and redirects output to
`data/POMDP_commit/`, so the two can never overwrite each other:

```bash
# full likelihood
SIM_CONFIG_PATH=.../simulation_params_SB-XT-RPh----.py        python3 scripts/fit/fit_data.py
# commit likelihood
SIM_CONFIG_PATH=.../simulation_params_SB-XT-RPh----_commit.py python3 scripts/fit/fit_data.py
```

**Hardware and cost.** The reported fits were produced on the Viper cluster at
the Max Planck Computing and Data Facility. Each node carries two AMD EPYC 9554
processors, so 128 physical cores and 256 hardware threads, with 480 GB of
memory. One model was fitted per node, all 105 subjects at once.

On that hardware a simple model takes about an hour and a forgetting model up to
nineteen, so the full set of 78 compared models is several hundred node hours.
Wall time is set by the slowest subject rather than by total work. `SIM_N_JOBS` controls how many subjects run at once.

## Make the figures

`scripts/generate_figures.py` is the single entry point, grouped the way the paper
is:

```bash
python3 scripts/generate_figures.py --list      # every target and what it makes
python3 scripts/generate_figures.py methods     # needs no data at all
python3 scripts/generate_figures.py results appendix
python3 scripts/generate_figures.py --only optimality
python3 scripts/generate_figures.py all --jobs 8
```

Targets run in parallel, two thirds of the cores available to the process by
default, which is the allocation rather than the whole machine when you are on a
batch node. `--jobs` overrides it, and `--jobs 1` runs them one at a time.

Each target runs in its own interpreter with the environment it expects, so one
failure does not stop the rest, and the exit status is the number that failed.
Output is captured and printed per target, so parallel runs stay readable, and a
target that fails reports the exception that caused it and the command to
reproduce it on its own.

Some targets refit GLMMs. `slurm/run_generate_figures.sh` runs the same thing as a batch job, and takes `POMDP_DATA_ROOT` if the fits sit on a scratch filesystem rather than inside the clone:

```bash
POMDP_DATA_ROOT=/scratch/pomdp SECTIONS=results sbatch slurm/run_generate_figures.sh
```

## Reproduce a figure without refitting

`results/` holds the per subject fitted parameters for the three winning models
as CSV, the summary tables, and `best_models.json`. That is enough to regenerate
figures in seconds:

```bash
mkdir -p BIC/optimality && cp results/optimality/optimality_cost_informed.csv BIC/optimality/
OPTIMALITY_SRC=$PWD/BIC/optimality/optimality_cost_informed.csv \
python3 scripts/optimality/export_optimality_figure.py
```

The simulation ensembles are not deposited, because they run to roughly 185 GB.
They regenerate from the fitted parameters with `scripts/ensembles/regenerate_ensembles.py`.

## The rest of the pipeline

`scripts/run_pipeline.py` is the entry point for everything that is not a
figure, grouped the way the analysis runs:

```bash
python3 scripts/run_pipeline.py --list       # every stage and what it does
python3 scripts/run_pipeline.py preprocess
python3 scripts/run_pipeline.py comparison recovery
python3 scripts/run_pipeline.py all --dry-run
```

Stages run in dependency order, so a stage never starts before what it reads.
Several describe one model rather than the whole set and need `SIM_CONFIG_PATH`
to say which; those are skipped with a message naming the variable rather than
failing part way in. Fitting all 78 models is a job array rather than a loop
here, and `slurm/` holds the templates.

`PIPELINE.txt` is the long form, with the exact commands and the reasoning for
each stage.

Between the two runners, `scripts/optimality/export_fitted_parameters_csv.py` is what
writes the deposit in `results/`: `results.pkl` is a pandas pickle, fragile
across versions and unreadable outside Python, so the fitted parameters are
also kept as CSV.

## Tests

```bash
python3 -m unittest discover -s tests
```

Plain `unittest`, nothing extra to install. Tests skip rather than fail when the
data or the deposited results are absent, so a fresh clone still exercises the
config layer and the model itself. `tests/test_reproducibility.py` recomputes
the numbers quoted in the paper from the deposit. See `tests/README.md`.

## Things that will catch you out

- **Always set `SIM_CONFIG_PATH`.** The default config is a forgetting model,
  which needs the gamma grids, so a bare `POMDPFactory()` can fail confusingly.
- **The optimiser is `de`.** `ga` is also available. The algorithm is a
  component of the output path, so results from the two never mix.
- **The cost function returns `1e10` when it raises.** A model whose likelihood
  errors will appear to fit and produce noise. Grep job logs for `[Cost Error]`.
- **Ensembles do not know the fits changed.** Rerun
  `scripts/ensembles/regenerate_ensembles.py` after refitting, and set
  `GLM_SIMULATE_FORCE_REFIT=1` to invalidate the cached GLM betas.

## Layout

```
src/            model classes, fitting, GLM, statistics, config, and all plotting
scripts/        54 scripts, behind two entry points: run_pipeline and generate_figures
slurm/          19 job scripts, one per stage, as templates
notebooks/      the primer, data exploration, and three comparison and
                recovery notebooks
data/           152 configs, and where the downloaded dataset goes
BIC/            the winners, the model comparison, and the optimality csvs
BIC_commit/     the same for the commit likelihood family
results/        deposited fitted parameters and summary tables
tests/          unittest suite

Fitting and the ensembles write into `data/`, and the analyses write into `BIC/`
and `BIC_commit/`. Both start with what the manuscript reports, so a figure can
be regenerated before anything has been refitted.
```

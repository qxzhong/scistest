# Framingham Heart Study application

Install `pip install -e '.[experiments]'` and run from the repository root:

```bash
python -m dataApplication.run_framingham --dry-run
python -m dataApplication.run_framingham --workers 4
python -m dataApplication.run_framingham --summarize-only
python -m dataApplication.plot
```

The program loads SurvSet's Framingham dataset at runtime. Across the six
covariates it drops incomplete cases, normalizes F/M or numeric 2/1 sex codes to 0/1, and requires the
reference analysis sample: 4,658 subjects and 1,465 events. No raw dataset is
bundled. Runtime data snapshots, fit checkpoints, aggregate tables and plots
are written only under the requested output directory (default
`results/framingham`, ignored by Git).

Every split estimates preprocessing on its hunting half. The hazard has one
hidden layer of width 32, the direction two layers of width 32, and engression
layer parameter 2 with width 100. All use 100 epochs; learning rates are
0.01/0.01/0.005; hazard and direction L1 penalties are 0.001. Direction fitting
uses the best checkpoint of the standardized hunting objective. The generator
and scoring hazard are fitted on the evaluation half. Both numerical counts
are 100: integration points and generator draws.

Each covariate gets J=10 permutations,
m=1552, B=30 calibration rows and L=3 splits. One pooled rank-transformed
calibration distribution is reused for all 1,000 full-data repetitions.
Gaussian-kernel tails use bw.nrd0 and log-domain evaluation; decimal strings
preserve probabilities below floating-point resolution. There are 18,540
complete single-split fits. `--repetitions 1` reduces the full-data phase, but
still fits all 540 calibration tests.

Execution is checkpointed and can resume with unchanged metadata. Source
fingerprints and analysis settings are saved on first run. Failures remain
explicit in checkpoints and make the run fail. Existing successful fits
must record evaluation-hazard scoring to be reused. A process lock prevents
two runs from writing the same output directory simultaneously.

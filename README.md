# GammaMAE v2 — masked spectrum pretraining for cross-detector radionuclide identification

Code, GATE macros and analysis scripts for the manuscript

> W. Hou, C. Zhang, D. Zhou, J. Shao, Y. Wang, J. Peng, *Masked spectrum pretraining for radionuclide identification across NaI(Tl), GAGG:Ce and HPGe detectors: a Monte Carlo study* (manuscript submitted).

All results are obtained from simulated spectra (GATE 9.4.2 / Geant4 11.4.0). No measured spectra are used.

## Layout

| Path | Content |
|---|---|
| `gate/` | GATE job definitions (`jobs/`, 279 jobs = 3 detectors × 10 nuclides × 3 distances × 3 seeds + 9 background jobs), line tables (`source_lines.csv`, `background_lines.csv`, ENSDF/LiveChart data in `gen/linedata/`), batch runner `run_all.sh`, ROOT helpers and `tools/convert_to_npz.py` |
| `gmae2/` | Python package: dataset synthesis (`build_dataset.py`), models (`models.py`), baselines (`baselines.py`), training (`nn_train.py`), experiments E1–E5 (`run_experiments.py`), summary tables (`summarize.py`) |
| `tools/` | Activity scan, learnability check, validation-set ablation (`tune_gmae.py`), export of the spectra used in the paper figures (`export_for_paper.py`) |
| `paper/` | Scripts that produce every table and figure of the manuscript from `paper/inputs/` (`tables.py`, `figs.py`, `agg.py`) |
| `paper/inputs/` | Per-cell test results of all runs (`results.csv`, `results_low_tr200.csv`), validation ablation log, spectra for the figures (`paper_spectra.npz`) |

## Requirements

Python ≥ 3.10, PyTorch (CUDA recommended), numpy, pandas, scikit-learn, matplotlib, tabulate; `uproot` and `awkward` for reading GATE ROOT output. The experiments were run on one NVIDIA RTX 4060 (8 GB).

## Pipeline

```bash
# 1. GATE event lists (VM with GATE 9.4.2): run all jobs, then convert ROOT -> npz
bash gate/run_all.sh
python gate/tools/convert_to_npz.py --src <gate_output> --dst <events_npz>

# 2. Spectrum synthesis: 1 kBq thinning, energy response, background, 1-100 s windows, perturbed test sets
python -m gmae2.build_dataset --events <events_npz> --out <data> --activity_kbq 1

# 3. Optional checks and validation-set ablation (train/val only)
python -m tools.learn_check --data <data> --n 6000 --epochs 15
python -m tools.tune_gmae --data <data>

# 4. Experiments E1-E5 (resumable), summary tables
python -m gmae2.run_experiments --data <data> --out <results>
python -m gmae2.summarize --out <results>

# 5. Paper tables and figures (cell-averaged accuracy)
python -m tools.export_for_paper --data <data> --npz <events_npz> --out paper_export.zip   # -> paper/inputs/paper_spectra.npz
cp <results>/results.csv paper/inputs/
python paper/tables.py && python paper/figs.py      # output in paper/out/
```

Running step 5 with the files already in `paper/inputs/` reproduces the tables and figures of the manuscript exactly.

## Metric

Accuracy is cell-averaged: the accuracy is computed in each (detector, live time, distance) cell and the cells are averaged with equal weight; neural networks are reported as mean ± s.d. over three training seeds.

## Data

The simulated spectra (training, validation and test splits and the perturbed test sets) are deposited in Science Data Bank (ScienceDB); the DOI will be added here on publication. The GATE event lists can be regenerated with the macros in `gate/` and are available from the corresponding authors on reasonable request.

## License

This code is released under the MIT License; see `LICENSE`.

## Contact

Wendi Hou, School of Nuclear Science and Technology, Lanzhou University — houwd2025@lzu.edu.cn

Corresponding authors: Chunhui Zhang (zhangchunhui@lzu.edu.cn), Dianwei Zhou (zhoudianwei@swip.ac.cn)

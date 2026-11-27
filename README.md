# IVF dose and drug demo (research prototype)
Run: unzip, then `./run.sh` (Mac/Linux) or double-click `run.bat` (Windows). First run installs packages (needs internet once), after that it works offline. Opens at http://localhost:8501.
Needs Python 3.9+. Trained on retrospective de-identified data. Not for clinical use.
- LightGBM (champion): total dose, start daily dose, 4 drug-class flags, 18 brand classifiers.
- MLP (challenger): dose only.
- Inputs are strict Day-0 features only. Locked test rows were never used.

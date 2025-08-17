# Epoxy-Bonded Steel Plate Retrofitting in RC Beams — Prediction App
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16889615.svg)](https://doi.org/10.5281/zenodo.16889615)
Streamlit app and trained artifacts for:
> **Comprehensive Parametric Investigation of Epoxy-Bonded Steel Plate Retrofitting in RC Beams Using Cohesive Zone Finite Element Modeling**

This repository contains:
- `app.py` — Streamlit app for inference
- `final_extratrees_model.pkl` — trained model (add your file)
- `scaler.pkl` — feature scaler (add your file)
- `top_features.json` — list of input features in the correct order (add your file)
- `requirements.txt` — dependencies

> 🔖 **DOI**: *(will appear after Zenodo archives your GitHub release)*  
> 📦 **Latest Release**: *(add link after publishing a GitHub Release)*

---

## 1) Quickstart

```bash
# create and activate a virtual env (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# put your three files alongside app.py:
#  - final_extratrees_model.pkl
#  - scaler.pkl
#  - top_features.json

streamlit run app.py
```

Open the URL that Streamlit prints (typically http://localhost:8501).

---

## 2) How to Publish on GitHub and Mint a DOI (Zenodo)

### A. Prepare the repository locally
1. Place your files in this folder:
   - `app.py`, `requirements.txt`, `.gitignore`, `LICENSE`, `CITATION.cff`, `.zenodo.json`
   - your artifacts: `final_extratrees_model.pkl`, `scaler.pkl`, `top_features.json`
2. Ensure each artifact is **< 100 MB**. If any file ≥ 100 MB:
   - Use **Git LFS** or upload large files as **GitHub Release assets** after pushing the repo.

### B. Create a GitHub repository
**Web UI (simplest):**
1. Go to GitHub → **New repository**.
2. Name it (e.g., `rc-beam-retrofit-app`), set visibility (Public), and click **Create**.
3. Click **Upload files**, drag the contents of this folder, and **Commit**.

**Git (command line):**
```bash
git init
git add .
git commit -m "Initial commit: Streamlit app and artifacts"
git branch -M main
git remote add origin https://github.com/<USER>/rc-beam-retrofit-app.git
git push -u origin main
```

### C. Link GitHub to Zenodo & archive a release
1. Sign into **Zenodo** (use GitHub login).  
2. In Zenodo → **GitHub** tab, **enable** your repository (toggle ON).
3. Back on GitHub, create a **Release**: go to **Releases → Draft a new release**.
   - Tag: `v1.0.0`
   - Title: `v1.0.0 – initial public release`
   - Release notes: short summary (see `RELEASE_NOTES.md` template).
   - (Optional) Upload large `.pkl` files here as **release assets**.
   - Click **Publish release**.
4. Zenodo will automatically archive this release and mint:
   - a **versioned DOI** (e.g., 10.5281/zenodo.1234567)
   - a **Concept DOI** (stable for all versions).

### D. Add the DOI badge
After Zenodo finishes, edit this README and add the badge (Zenodo shows the snippet):

```
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

---

## 3) How to Cite

If citing the **software/repository**:
```bibtex
@software{kholil_rcbeam_2025,
  author       = {Your Name},
  title        = {Epoxy-Bonded Steel Plate Retrofitting in RC Beams — Prediction App},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

If citing the **paper** (after acceptance), add the standard citation here as well.

---

## 4) License

- **Code**: MIT License (see `LICENSE`).  
- **Model weights & scaler**: you may keep the same MIT license **or** choose a different one (e.g., CC BY 4.0, CC BY-NC 4.0). Update this section to reflect your choice.

---

## 5) Reproducibility Notes

- The scaler expects features exactly in the order provided by `top_features.json`.
- The model was trained on the dataset referenced in the paper; ensure feature ranges are realistic.
- For deterministic runs, set `PYTHONHASHSEED` and explicit versions in `requirements.txt`.

---

## 6) Paper Text Snippets

**Code & Data Availability**  
*The trained ExtraTrees model, feature scaler, and Streamlit prediction app used in this study are openly available at GitHub (Repository: <REPO_URL>) and archived on Zenodo (DOI: 10.5281/zenodo.XXXXXXX).*
  
**Citation of the Repository**  
*I. Kholil, “Epoxy-Bonded Steel Plate Retrofitting in RC Beams — Prediction App,” v1.0.0, 2025. Zenodo. doi:10.5281/zenodo.XXXXXXX.*

---

## 7) Acknowledgements

Add funding and acknowledgements here if applicable.

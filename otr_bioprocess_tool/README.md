# OTR Bioprocess Design Tool

A small Streamlit browser app for estimating OTRmax in shaken flask processes.

## Files

- `otr_web_app.py` — Streamlit app
- `requirements.txt` — Python dependencies for Streamlit Cloud
- `README.md` — this instruction file

## Deploy on Streamlit Community Cloud

1. Create or log into your GitHub account at https://github.com
2. Create a new repository, for example `otr-bioprocess-tool`
3. Upload these files into the repository root:
   - `otr_web_app.py`
   - `requirements.txt`
   - `README.md`
4. Open https://share.streamlit.io
5. Sign in with GitHub
6. Click **New app**
7. Select your repository and branch, usually `main`
8. Set the main file path to:

```text
otr_web_app.py
```

9. Click **Deploy**

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run otr_web_app.py
```

## Scientific note

This version is a practical demonstrator using an empirical model structure.

Before scientific publication, customer-facing use, or product decisions, replace or calibrate the coefficients using the exact equation from the referenced paper and/or experimental OTRmax data from the target flask/shaker/media system.

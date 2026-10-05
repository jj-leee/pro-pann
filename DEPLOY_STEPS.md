# Deploying PRO-PANN to https://pro-pann.streamlit.app

Everything local is done and tested:

- the bundle is in `~/pro-pann-streamlit`;
- the smoke test passes;
- predictions match the original app exactly;
- a headless launch works.

The steps below **publish** the app. Do them yourself, in order.

## 0. One-time prerequisites

`git` 2.52 is installed. `gh` is **not** installed, and no git identity is configured.

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"   # ideally the email on your GitHub account
```

`gh` is optional. Install it only if you want to create the repo from the terminal:

```bash
brew install gh
gh auth login          # GitHub.com -> HTTPS -> login with a web browser
```

## 1. Create the GitHub repo

- **Name:** `pro-pann`.
- **Public** is recommended:
  - Community Cloud deploys public repos with no extra permissions.
  - It matches PRO-TONGUE.
  - The repo holds code, trained models and aggregate summaries only. The privacy audit found no patient-level data.
  - Before going public, confirm that your co-authors are comfortable publishing the trained model files.
- **Private** also works. On share.streamlit.io you then need to grant Streamlit access to private repos:
  1. Open your workspace settings.
  2. Go to Linked accounts → GitHub.
  3. Authorize private repo access.

Choose ONE of these options.

- **Option A, web UI:** on https://github.com/new, set the owner to your account, the name to `pro-pann`, and Public. Do **not** add a README, .gitignore or license, so the repo starts empty. Click **Create repository**.
- **Option B, gh:** run this after step 2's commit:
  ```bash
  gh repo create pro-pann --public --source ~/pro-pann-streamlit --remote origin --push
  ```

## 2. Commit and push

```bash
cd ~/pro-pann-streamlit
git init -b main
git add -A
git status --short     # review: about 75 files; NO .parquet/.csv/.xlsx, NO app/all_preds/, NO __pycache__
git commit -m "PRO-PANN Streamlit risk calculator (de-identified deployment bundle)"
git remote add origin https://github.com/<your-github-username>/pro-pann.git   # skip if you used Option B
git push -u origin main
```

GitHub doesn't accept account passwords for HTTPS pushes. Authenticate in one of these ways:

- run `gh auth login` followed by `gh auth setup-git`;
- use a Personal Access Token when prompted. The macOS keychain stores it.

## 3. Deploy on share.streamlit.io

1. Go to https://share.streamlit.io and sign in with GitHub. Authorize Streamlit if asked.
2. Click **Create app**, then deploy from a GitHub repo.
3. Fill in the form:
   - **Repository:** `<your-github-username>/pro-pann`
   - **Branch:** `main`
   - **Main file path:** `app/base_app.py`
   - **App URL:** `pro-pann`, which gives **pro-pann.streamlit.app**. If that subdomain is taken, the form will say so.
4. Click **Advanced settings**:
   - **Python version: 3.12.** This is required. The pickled models are pinned to Python 3.12 builds of the packages. Community Cloud ignores `runtime.txt` and `.python-version`. The Python version **can't be changed after deploy** without deleting and redeploying.
   - **Secrets:** none needed.
5. Click **Save**, then **Deploy**.

The first build takes a few minutes:

- `packages.txt` installs `libgomp1` via apt;
- `requirements.txt` installs about 60 packages, with no torch and no CUDA. XGBoost uses the CPU-only `xgboost-cpu` wheel on Linux.

## 4. Verify the live app

1. Open https://pro-pann.streamlit.app.
2. Tick all 13 outcomes, keep the default inputs, and click **Predict outcomes**.
3. Each outcome should show:
   - a risk;
   - a category;
   - a percentile;
   - an observed band rate;
   - its own SHAP chart, different for each outcome.
4. If anything fails, check **Manage app** (bottom right of the app) → logs.

Local reference values for the smoke-test patient, from `python smoke_test.py`:

| Outcome | Expected risk |
|---|---|
| Wound complication | 11.59% |
| Readmission | 7.02% |
| Any complication | 14.40% |

## Notes

- Community Cloud apps go to sleep after a period without traffic. The first visitor after that wakes the app, which takes about 30 s.
- To update the app:
  1. Push to `main`. The app redeploys automatically.
  2. Follow the "Updating the models" steps in `README.md`.
  3. Never copy `app/all_preds/` or any data files into this repo.

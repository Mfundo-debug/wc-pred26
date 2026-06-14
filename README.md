# WC26 Bayesian Hierarchical Simulation Package

Generated: 2026-06-12 14:31 UTC

## Run immediately
Static dashboard:
```bash
cd wc26_bhm_final
python -m http.server 8000
# open http://localhost:8000/dashboard/
```

Streamlit:
```bash
cd wc26_bhm_final
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

## Included
- PDF report
- PPTX deck
- static dashboard with local SVG flags
- Streamlit app
- notebook
- PyMC/model scripts
- 100,000 simulation CSV outputs

## Verified live results included
- Mexico 2-0 South Africa
- Korea Republic 2-1 Czech Republic

Audit: example images used only as layout references, not model evidence.

## Reproducibility note
- `src/full_build_pipeline.py` contains the full modelling/simulation pipeline used to generate the package.
- `src/final_artifact_builder.py` contains the dashboard/report/deck artifact builder.
- The packaged run uses verified live results available at build time and does not fabricate unverified matches.

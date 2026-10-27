# Fed-AgentGEO public demo

Fed-AgentGEO is a research demonstration of federated retrieval over three demo hospital nodes and public clinical guideline examples.

## Public demo safety

- Bundled case records are de-identified demonstration examples with generic case IDs and clinician labels. Their synthetic status has not been independently verified; treat them as potentially sensitive.
- Tab 2 supports document upload and case entry. Uploaded documents are parsed by the public app, and submitted cases are added to the shared demo index where other visitors may see or retrieve them. Use fictional or de-identified examples only.
- Do not enter real patient information. Queries are processed by the public hosted app.
- Results are for research demonstration only. They are not medical advice and must not guide patient care.
- This demo has no user authentication and is not designed to store or process protected health information. Hosted storage may be temporary and can reset when the app restarts.

## Run locally

```sh
python -m pip install -r requirements.txt
streamlit run app.py
```

The first run creates local ChromaDB indexes from the bundled example cases and guideline examples. The app uses its offline mock mode by default and does not require an API key.

## Deploy on Streamlit Community Cloud

Create a GitHub repository containing this folder, then create a Community Cloud app with `app.py` as the entry point. Streamlit installs dependencies from `requirements.txt`.

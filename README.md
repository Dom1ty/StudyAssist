# Study Assistant

Local-first Streamlit tutor for university courses. It ingests course PDFs into a shared Chroma index filtered by `course_id`, explains concepts in plain language, supports hint/guided/full-solution modes, and keeps a per-course mistake log.

## Run

1. Activate a Python environment with the dependencies in `requirements.txt`.
2. Set `DASHSCOPE_API_KEY` if you want the live DashScope/Qwen tutor instead of the built-in fallback mode.
3. Start the app:

```powershell
streamlit run app.py
```

## Test

```powershell
pytest
```

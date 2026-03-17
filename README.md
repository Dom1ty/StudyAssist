# Study Assistant

Local-first Streamlit tutor for university courses. It ingests course PDFs into a shared Chroma index filtered by `course_id`, explains concepts in plain language, supports hint/guided/full-solution modes, and keeps a per-course mistake log.

## Run

1. Activate a Python environment with the dependencies in `requirements.txt`.
2. Create a `.env` file in the project root or set `DASHSCOPE_API_KEY` in your shell if you want the live DashScope/Qwen tutor instead of the built-in fallback mode.
3. Start the app:

```powershell
streamlit run app.py
```

Example `.env`:

```dotenv
DASHSCOPE_API_KEY=your_api_key_here
```

## Test

```powershell
pytest
```

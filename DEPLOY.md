# Deployment Guide: Streamlit Community Cloud

This guide explains how to deploy the Multi-Agent Data Analyst application on [Streamlit Community Cloud](https://streamlit.io/cloud).

---

## 1. Deploying the App

1. Sign in to **Streamlit Community Cloud**.
2. Click **New app** and select:
   - **Repository**: `samikshagla/multi-agent-data-analyst` (or your fork)
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - Under **Advanced settings**, set **Python version**: `3.11`
3. Click **Deploy**.

---

## 2. Secrets Configuration

In the Streamlit Cloud dashboard, navigate to **Settings** > **Secrets** for your app.

Enter **only** the following line:

```toml
HOSTED = "1"
```

> [!IMPORTANT]
> **Never add API keys to the Secrets box.**
> In hosted mode (`HOSTED = "1"`), server-side keys are strictly disabled to protect your account quota. Each visitor provides their own API key in their browser session, which is held solely in session memory and never saved to disk or logged.

---

## 3. Environment & Resource Settings

All settings can be configured via Streamlit Cloud Secrets (or environment variables). The table below lists all supported settings and their defaults:

| Setting | Description | Hosted Default (`HOSTED=1`) | Local Default (`HOSTED=0`) |
| :--- | :--- | :--- | :--- |
| `HOSTED` | Enables multi-tenant server guards, visitor API key entry, and sandbox isolation | `"1"` | `"0"` |
| `DEMO_ONLY` | Disables file uploads and live execution; runs only recorded demo replay | `"0"` | `"0"` |
| `MAX_UPLOAD_MB` | Maximum allowed file size for uploaded CSV files | `5` | `20` |
| `MAX_ROWS` | Maximum allowed row count for uploaded datasets | `30000` | `200000` |
| `MAX_COLS` | Maximum allowed column count for uploaded datasets | `60` | `100` |
| `MAX_RUN_TOKENS` | Token consumption ceiling per run before aborting | `80000` | `None` (unlimited) |
| `MAX_CONCURRENT_RUNS` | Maximum simultaneous pipeline runs across the server | `2` | `2` |
| `RUN_COOLDOWN_SECONDS` | Cooldown period between successive runs from the same session | `60` | `0` |
| `RUN_WALL_CLOCK_SECONDS` | Execution timeout in seconds per run | `600` | `None` (unlimited) |

---

## 4. How to Redeploy

- **Automatic Deployments**: Pushing new commits to the `main` branch automatically triggers a redeploy on Streamlit Community Cloud.
- **Manual Reboot**: In your app's bottom-right menu (or from the cloud dashboard), click **Manage app** > `...` > **Reboot app**.
- **Clear Cache and Reboot**: If dependencies change or you experience unexpected behavior, select **Clear cache and reboot** from the same menu.

---

## 5. Troubleshooting

- **Build fails during deployment**:
  Ensure **Python 3.11** is selected in Advanced Settings. The app uses pinned dependencies in `requirements.txt` designed for Python 3.11.
- **App out of memory (OOM)**:
  Streamlit Community Cloud free tier provides ~1 GB of container RAM. If users upload large datasets that exceed memory limits during processing, the container may restart. Keep `MAX_UPLOAD_MB` at `5` and `MAX_ROWS` at `30000` to prevent OOM errors.
- **Rate-limit messages**:
  Free-tier AI provider limits (e.g., Groq RPM/TPM) may trigger transient rate limits. Users should verify their API key quota on their provider's dashboard or switch to an alternate provider/model in the sidebar. Visitors can also watch the recorded demo replay without an API key.

# 🚀 Deployment Guide for Wellness Chatbot

## Deploying to Streamlit Cloud

### Step 1: Prepare Your Repository

1. Make sure all files are committed:
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `README.md` (documentation)
   - `.gitignore` (Git ignore rules)
   - `.streamlit/config.toml` (Streamlit configuration)

2. Initialize Git (if not already done):
```bash
git init
git add .
git commit -m "Initial commit: Wellness Chatbot"
```

### Step 2: Push to GitHub

1. Create a new repository on GitHub (don't initialize with README if you already have one)

2. Connect and push:
```bash
git remote add origin https://github.com/YOUR_USERNAME/wellness_chatbot.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub account if not already connected
4. Fill in the deployment form:
   - **Repository**: Select `yourusername/wellness_chatbot`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **Python version**: `3.8` or higher (default)

5. **Important**: Add a build command in Advanced settings:
   ```
   pip install -r requirements.txt && python -m spacy download en_core_web_sm
   ```
   
   OR create a `packages.txt` file (already created) which Streamlit Cloud will use automatically.

6. Click **"Deploy"**

### Step 4: Wait for Deployment

- Streamlit Cloud will install dependencies and build your app
- First deployment may take 3-5 minutes
- You'll get a URL like: `https://your-app-name.streamlit.app`

### Troubleshooting

#### If spaCy model fails to load:
- Ensure `packages.txt` includes the spaCy model wheel URL
- Or use the build command: `python -m spacy download en_core_web_sm`

#### If dependencies fail:
- Check `requirements.txt` for correct package names and versions
- Streamlit Cloud uses Python 3.8+ by default

#### If database errors occur:
- SQLite works on Streamlit Cloud, but data resets on each redeploy
- Consider using an external database (PostgreSQL) for production

## Local Development

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the app
streamlit run app.py
```

### Windows
```bash
# Run the setup script
setup.bat

# Or manually:
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

### Linux/Mac
```bash
# Make setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh

# Or manually:
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

## Environment Variables (Optional)

For production, consider using Streamlit Cloud's secrets management:

1. Go to your app's settings in Streamlit Cloud
2. Add secrets via the "Secrets" tab
3. Create `.streamlit/secrets.toml`:
```toml
[default]
SECRET_KEY = "your-secure-secret-key-here"
ADMIN_EMAIL = "your-admin@email.com"
ADMIN_PASSWORD = "your-secure-password"
```

Then update `app.py` to read from secrets:
```python
import os
SECRET_KEY = os.environ.get("SECRET_KEY", "wellness_secret_key")
```

## Notes

- **Database**: SQLite database (`users.db`) will be created automatically
- **Model**: Intent classification model (`intent_model.pkl`) will be created on first run
- **Admin Access**: Default admin credentials are in the code (change for production)
- **Data Persistence**: On Streamlit Cloud, data persists between sessions but resets on redeploy


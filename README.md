# 💬 Wellness Guide Chatbot

A Streamlit-based wellness chatbot application that provides health advice and wellness tips. The application uses machine learning for intent detection and supports multiple languages (English and Hindi).

## 🌟 Features

- **User Authentication**: Secure signup and login system with JWT tokens
- **Intent Detection**: ML-based intent classification using TF-IDF and Logistic Regression
- **Multi-language Support**: English and Hindi language support with automatic translation
- **Admin Dashboard**: Comprehensive admin panel for managing knowledge base, viewing analytics, and monitoring user queries
- **Knowledge Base Management**: Dynamic knowledge base stored in SQLite with CRUD operations
- **User Feedback System**: Collect and analyze user feedback on chatbot responses
- **Analytics Dashboard**: Visual analytics with charts for queries, feedback, and user demographics
- **Chat History**: Persistent chat history per user session
- **Profile Management**: User profile settings with customizable preferences

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wellness_chatbot.git
cd wellness_chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

4. Run the application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📦 Deployment on Streamlit Cloud

1. Push your code to GitHub (see `DEPLOYMENT.md` for detailed steps)
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Click "New app"
4. Connect your GitHub repository
5. Set the main file path to `app.py`
6. **Important**: In Advanced settings, add this build command:
   ```
   pip install -r requirements.txt && python -m spacy download en_core_web_sm
   ```
   This ensures the spaCy model is downloaded during deployment.
7. Click "Deploy" and wait 3-5 minutes for the first deployment

For detailed deployment instructions, see `DEPLOYMENT.md`.

### Streamlit Cloud Configuration

- **Main file**: `app.py`
- **Python version**: 3.8 or higher
- **Build command**: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`

## 🔑 Admin Access

- **Email**: `anish@gmail.com`
- **Password**: `anish123`

Admin users can access the admin dashboard from the sidebar to:
- Manage knowledge base intents
- View user analytics
- Monitor queries and feedback
- Export data

## 📊 Database Schema

The application uses SQLite with the following tables:

- **users**: User accounts and profiles
- **knowledge_base**: Intent-based health tips and advice
- **queries**: Logged user queries for analytics
- **feedback**: User feedback on chatbot responses

## 🛠️ Technologies Used

- **Streamlit**: Web framework for the UI
- **scikit-learn**: Machine learning for intent classification
- **SQLite**: Database for data persistence
- **spaCy**: NLP for symptom and body part extraction
- **googletrans**: Translation service for Hindi support
- **PyJWT**: JWT token authentication
- **bcrypt**: Password hashing
- **Plotly**: Interactive charts and visualizations
- **pandas**: Data manipulation and analysis

## 📝 Project Structure

```
wellness_chatbot/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore rules
├── users.db             # SQLite database (created automatically)
└── intent_model.pkl     # ML model (created automatically)
```

## 🔒 Security Notes

- Passwords are hashed using bcrypt
- JWT tokens expire after 12 hours
- Admin credentials are hard-coded (change in production)
- SECRET_KEY should be changed to a secure random key in production

## 📈 Analytics Features

The admin dashboard includes:
- User registration metrics
- Query analytics with time-series charts
- Intent distribution visualization
- Feedback analysis
- User demographics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This chatbot provides general wellness tips and is not a substitute for professional medical advice. Users should consult healthcare professionals for serious health concerns.


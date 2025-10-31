# 💬 Wellness Guide Chatbot

An intelligent health and wellness chatbot that provides personalized advice and tips for common health concerns. Built with Streamlit, ML-based intent detection, and multilingual support.

## 🌟 Features

- **User Authentication**: Secure signup and login with bcrypt password hashing
- **Admin Dashboard**: Full analytics and knowledge base management
- **ML-Based Intent Detection**: Uses TF-IDF and Logistic Regression for query classification
- **Multilingual Support**: English and Hindi language support
- **Real-time Analytics**: 
  - Query trends over time
  - Intent distribution charts
  - User demographics
  - Feedback analysis
- **Feedback System**: User feedback collection with comments
- **Knowledge Base Management**: Admin can add/edit/delete health tips
- **Profile Management**: Users can update their profile information

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, SQLite
- **ML**: scikit-learn (TF-IDF, Logistic Regression)
- **NLP**: spaCy, langdetect
- **Analytics**: Plotly, Pandas
- **Translation**: googletrans
- **Authentication**: JWT, bcrypt

## 🚀 Quick Deployment

**Ready to deploy?** Check out these guides:
- ⚡ [QUICK_START.md](QUICK_START.md) - Get deployed in 5 minutes
- 📖 [DEPLOYMENT.md](DEPLOYMENT.md) - Detailed deployment instructions
- ✅ [CHECKLIST.md](CHECKLIST.md) - Pre-deployment checklist

## 📦 Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd wellness_chatbot
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

   The app will be available at `http://localhost:8501`

## 🚀 Deployment on Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/wellness_chatbot.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository and branch
   - Set the main file path to `app.py`
   - Click "Deploy"

3. **Admin Access**
   - Email: `anish@gmail.com`
   - Password: `anish123`

## 📊 Admin Features

The admin dashboard provides:
- **Knowledge Base Manager**: Add, edit, or delete health intents and tips
- **Analytics Dashboard**: View user queries, feedback, and trends
- **User Management**: View registered users and demographics
- **Data Export**: Export queries to CSV

## 🔐 Security

- Password hashing with bcrypt
- JWT token-based authentication
- SQL injection protection via parameterized queries
- Session management

## 📝 Default Intents

The chatbot currently supports:
- Headache relief
- Fever tips
- Wound care
- Cough relief
- Cold care
- Stomach pain relief
- Chest pain guidance
- Stress management
- General wellness tips
- Greeting/goodbye

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This chatbot provides general health information and tips. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

Made with ❤️ using Streamlit

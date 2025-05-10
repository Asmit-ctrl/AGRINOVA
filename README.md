
# 🌾 AgriNova – AI-Powered Smart Farming Platform

**AgriNova** is an end-to-end intelligent farming assistant designed to empower farmers with personalized crop planning, pest prediction, community collaboration, voice-assisted support, and real-time task tracking — all in a seamless, locally accessible interface.

---

## 👥 Team Name  
**Project 404** – Bennett University

---

## 🚀 Project Description

AgriNova uses AI and real-time data to assist farmers through every stage of farming — from crop recommendations and multi-level farming logic to pest control and financial tools. It supports offline use, regional languages, and includes a community-driven help system with live chat and alerts.

---

## 🧠 Problem Statement

Development of a smart farm management dashboard targeted at small and medium farmers to provide real-time crop insights, reduce dependency on guesswork, and improve yield and sustainability.

---

## 🛠️ Tech Stack

| Layer         | Technology                          |
|---------------|--------------------------------------|
| Frontend      | HTML5, CSS3 (TailwindCSS), Vanilla JS |
| Backend       | Node.js, Express                     |
| Database      | MongoDB, Local JSON (for testing)    |
| AI Models     | Python (Flask) – Custom-built CNN & ML |
| Real-time     | Socket.IO                            |
| Auth          | Google OAuth 2.0                     |
| Hosting       | Localhost / Render (for live demo)   |

---

## 📦 Features

- ✅ AI-Based Crop Recommendation (Based on season, soil, history, and NPK values)
- ✅ Multilayer Crop Planning Support
- ✅ Dynamic Task Management System (updates based on weather, pests, and progress)
- ✅ Pest Detection & Live Alert System
- ✅ Farmer Community Chat (WhatsApp-style UI)
- ✅ Voice Assistant + Regional Language Support
- ✅ Google Login + Secure User Management
- ✅ Loan & Subsidy Info Dashboard
- ✅ Real-time Graph Updates and Weather Sync

---

## 📁 Folder Structure

```
/AgriNova
│
├── /frontend/                  # HTML/CSS/JS landing pages
│   ├── landing_page.html
│   ├── login.html
│   ├── signup.html
│   └── questionnaire.html
│
├── /backend/                   # Express server
│   ├── server.js
│   ├── new_server.js
│   └── users.json              # (local user storage)
│
├── /chat/                      # Community chat module
│   ├── socket.js
│   ├── index.html
│   └── messages.json
│
├── /models/                    # AI model folders
│   ├── /shikhar/               # Chatbot
│   │   └── app.py
│   ├── /crop_recommendation/  # Crop ML
│   │   └── crops.py
│
└── README.md
```

---

## 🔧 Setup Instructions

### Frontend
1. Open `landing_page.html` using **Live Server** or your browser.
2. Ensure all internal paths (CSS/JS) are updated relative to your system.

### Backend
1. In terminal:
   ```bash
   node backend/server.js
   node backend/new_server.js
   ```

### AI Models
1. Navigate to `models/shikhar/`  
   Run:  
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
2. Navigate to `models/crop_recommendation/`  
   Run:  
   ```bash
   python crops.py
   ```

---

## 🧾 Dataset Notice

**⚠️ Due to government data protection laws, the training dataset has been removed from this repository.**


---

## 🛡️ Licensing & Usage

> This project is licensed **only for evaluation under HACKEMON**.  
> Any commercial or unauthorized usage will be served with legal notice.

---

## 📈 Future Plans

- Fully mobile app with offline sync  
- Satellite-based irrigation mapping  
- AI voice-guided task assistant  
- Scalable backend with cloud NPK prediction  
- Open APIs for government scheme integration  

---

## 🙌 Built with love by  
**Project 404 – Bennett University**  
🚀 All code, AI, and datasets are **custom-built** — no third-party black-box models.

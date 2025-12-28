# 🧠 NeuroScan: Brain Tumor Detection Using CNN (VGG19)

An AI-powered deep learning system for automated detection and classification of brain tumors from MRI images using Convolutional Neural Networks based on VGG19 architecture.

![NeuroScan Banner](https://img.shields.io/badge/AI-Brain%20Tumor%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![Flask](https://img.shields.io/badge/Flask-3.0.0-lightgrey)
![React](https://img.shields.io/badge/React-18.2.0-61DAFB)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)

## 📋 Table of Contents
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Technologies](#-technologies)
- [Contributors](#-contributors)

---

## ✨ Features

### Core Functionality
- **🔍 Automated Brain Tumor Detection** - Classifies MRI scans into 4 categories:
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - Normal (No Tumor)

### Advanced Features
- **🔥 Grad-CAM Heatmaps** - Visual explainability showing WHERE the model detected abnormalities
- **📄 PDF Report Generation** - Comprehensive medical reports with:
  - Patient information
  - MRI scan analysis
  - Probability distributions
  - Medical recommendations
  - Doctor signature space
  
- **💬 AI-Powered Chatbot** - Medical assistant using Ollama (LLaMA 3.2)
- **👤 User Authentication** - Secure login/registration system
- **📊 Admin Dashboard** - Advanced analytics with:
  - Real-time statistics
  - Predictions over time (Line Charts)
  - Tumor type distribution (Pie Charts)
  - Detection frequency (Bar Charts)
  - User management

- **📈 Prediction History** - Track all previous scans and results
- **🎨 Dark/Light Mode** - Customizable UI theme
- **🐳 Docker Support** - One-command deployment

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Upload UI  │  │  Results View│  │  Admin Panel │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ REST API
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Flask Backend                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  CNN Model   │  │   Grad-CAM   │  │  PDF Report  │      │
│  │   (VGG19)    │  │  Explainer   │  │  Generator   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │   SQLite DB  │  │    Ollama    │                        │
│  │   (Users,    │  │   Chatbot    │                        │
│  │  Predictions)│  │  (LLaMA 3.2) │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (optional)
- CUDA-compatible GPU (recommended)

### Option 1: Docker Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/neuroscan.git
cd neuroscan

# Start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend: http://localhost:5000
# Ollama: http://localhost:11434
```

### Option 2: Manual Installation

#### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the model weights
# Place Brain_Tumor_weights.pt in the backend directory

# Run the server
python app.py
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

#### Ollama Setup (for Chatbot)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3.2:3b

# Start Ollama server
ollama serve
```

---

## 💻 Usage

### 1. Create Superadmin Account
```bash
cd backend
python create_superadmin.py
```

### 2. Upload and Analyze MRI Scans
1. Login or register an account
2. Click "Upload MRI Image"
3. Select a brain MRI scan (JPEG/PNG)
4. (Optional) Fill in patient information
5. Click "Analyze" to get predictions

### 3. View Results
- **Prediction**: Tumor type classification
- **Confidence**: Model certainty percentage
- **Probability Distribution**: All class probabilities
- **Grad-CAM Heatmap**: Visual explanation (click "🔥 View Heatmap")
- **PDF Report**: Download comprehensive report

### 4. Admin Dashboard (Superadmin Only)
- Click "Admin" button in header
- View system statistics
- Analyze prediction trends
- Manage users
- Export data

---

## 📡 API Documentation

### Authentication Endpoints

#### POST `/register`
```json
{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

#### POST `/login`
```json
{
  "username": "string",
  "password": "string"
}
```

### Prediction Endpoints

#### POST `/predict`
- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `image`: MRI scan file
  - `patient_name` (optional)
  - `patient_age` (optional)
  - `patient_gender` (optional)
  - `scan_date` (optional)

**Response:**
```json
{
  "prediction_id": 1,
  "prediction": "glioma",
  "confidence": 87.5,
  "is_tumor": true,
  "probabilities": {
    "glioma": 87.5,
    "meningioma": 8.3,
    "notumor": 2.1,
    "pituitary": 2.1
  }
}
```

#### GET `/gradcam/<prediction_id>`
Returns Grad-CAM heatmap visualization

#### GET `/generate-report/<prediction_id>`
Downloads PDF report

### Admin Endpoints

#### GET `/admin/stats`
Returns system statistics

#### GET `/admin/users`
Returns all users

#### GET `/admin/predictions`
Returns all predictions

---

## 📊 Model Performance

### Dataset
- **Total Images**: 7,023 MRI scans
- **Train/Validation Split**: 80/20
- **Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Image Size**: 224×224 pixels
- **Augmentation**: Yes

### Performance Metrics
| Metric | Value |
|--------|-------|
| **Training Accuracy** | 94.23% |
| **Validation Accuracy** | 94.23% |
| **Test Accuracy** | 91.61% |
| **Average Confidence** | 89.3% |

### Model Architecture
- **Base**: VGG19 (Transfer Learning)
- **Framework**: PyTorch
- **Optimizer**: Adam
- **Loss**: Cross-Entropy
- **Layers**:
  - Conv layers: 4 (8 → 16 → 32 → 64 filters)
  - Pooling: Max pooling
  - FC layers: 2 (100 → 4 units)
  - Dropout: 0.25
  - Activation: ReLU, LogSoftmax

---

## 🛠️ Technologies

### Backend
- **Flask** - Web framework
- **PyTorch** - Deep learning
- **TorchVision** - Image preprocessing
- **ReportLab** - PDF generation
- **OpenCV** - Grad-CAM visualization
- **SQLite** - Database
- **Ollama** - AI chatbot

### Frontend
- **React** - UI framework
- **Lucide React** - Icons
- **Recharts** - Data visualization
- **Fetch API** - HTTP client

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Orchestration
- **Git** - Version control

---

## 📁 Project Structure

```
neuroscan/
├── backend/
│   ├── app.py                      # Main Flask application
│   ├── gradcam_utils.py            # Grad-CAM implementation
│   ├── pdf_report.py               # PDF generation
│   ├── create_superadmin.py        # Admin creation script
│   ├── migrate_db.py               # Database migration
│   ├── requirements.txt            # Python dependencies
│   ├── Dockerfile                  # Backend container
│   └── .dockerignore
│
├── frontend/
│   ├── src/
│   │   ├── App.js                  # Main React component
│   │   ├── Login.js                # Authentication UI
│   │   ├── AdminDashboard.js       # Admin panel
│   │   ├── ChatbotToggle.js        # AI chatbot
│   │   ├── PatientInfoModal.js     # Patient form
│   │   └── index.js                # Entry point
│   ├── package.json                # Node dependencies
│   ├── Dockerfile                  # Frontend container
│   └── .dockerignore
│
├── docker-compose.yml              # Container orchestration
├── README.md                       # This file
└── IMPLEMENTATION_GUIDE.md         # Development guide
```

---

## 🔒 Security Features

- **Password Hashing**: Werkzeug SHA-256
- **Session Management**: Flask sessions with secure cookies
- **CORS Protection**: Configured for localhost:3000
- **Rate Limiting**: 20 requests/minute
- **Input Validation**: File type and size checks
- **SQL Injection Prevention**: Parameterized queries
- **Authentication Required**: Protected routes

---

## 🧪 Testing

### Run Backend Tests
```bash
cd backend
pytest tests/
```

### Run Frontend Tests
```bash
cd frontend
npm test
```

### Manual Testing Checklist
- [ ] Register new user
- [ ] Login with credentials
- [ ] Upload MRI image
- [ ] View prediction results
- [ ] Generate PDF report
- [ ] View Grad-CAM heatmap
- [ ] Chat with AI assistant
- [ ] Access admin dashboard (superadmin)
- [ ] Export user data (admin)
- [ ] Logout and re-login

---

## 📈 Future Enhancements

- [ ] Multi-modal MRI support (T1, T2, FLAIR)
- [ ] 3D tumor segmentation
- [ ] Batch processing
- [ ] Email notifications
- [ ] Mobile app (React Native)
- [ ] Integration with PACS systems
- [ ] Multi-language support
- [ ] Cloud deployment (AWS/Azure)
- [ ] Real-time collaboration
- [ ] Advanced analytics dashboard

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.10+

# Verify dependencies
pip list

# Check port availability
lsof -i :5000
```

### Frontend connection errors
```bash
# Verify backend is running
curl http://localhost:5000/me

# Check environment variables
cat frontend/.env
```

### Ollama chatbot not working
```bash
# Check Ollama status
ollama list

# Restart Ollama
ollama serve

# Pull model again
ollama pull llama3.2:3b
```

### Docker issues
```bash
# Check Docker status
docker ps

# View logs
docker-compose logs -f

# Restart services
docker-compose restart
```

---

## 👥 Contributors

- **Kamala Thapa** (21185159) - Frontend Development, UI/UX
- **Sankalpa Adhikari** (21185177) - Backend Development, Model Training
- **Sushil Adhikari** (21185185) - Database, DevOps, Documentation

**Supervisor**: [Professor Name]  
**Institution**: Pokhara University  
**Program**: Bachelor of Software Engineering  
**Date**:  2025

---

## 📄 License

This project is developed as part of an academic requirement for Pokhara University.

**Important Medical Disclaimer**: This system is designed as a screening tool to assist medical professionals and should NOT be used as the sole basis for diagnosis or treatment decisions. Always consult qualified healthcare providers for medical advice.

---

## 📞 Contact

For questions or collaboration:
- **Email**: sankalpaadhikari38@gmail.com
- **GitHub**: [github.com/yourusername/neuroscan](https://github.com/Sankalpaadhikariii/neuroscan)
- **Issues**: [Report bugs](https://github.com/yourusername/neuroscan/issues)

---

## 🙏 Acknowledgments

- Dataset: [Brain MRI Images](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- VGG19 Architecture: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- Grad-CAM: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- Icons: [Lucide Icons](https://lucide.dev/)
- Charts: [Recharts](https://recharts.org/)

---

## ⭐ Star Us!

If you found this project helpful, please give it a star ⭐ on GitHub!

```bash
# Quick start command
git clone https://github.com/Sankalpaadhikariii/neuroscan.git
cd neuroscan
docker-compose up --build
```

**Built with ❤️ by Team NeuroScan**
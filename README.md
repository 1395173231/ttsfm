# TTSFM

[![Docker Pulls](https://img.shields.io/docker/pulls/dbcccc/ttsfm?style=flat-square&logo=docker)](https://hub.docker.com/r/dbcccc/ttsfm)
[![License](https://img.shields.io/github/license/dbccccccc/ttsfm?style=flat-square)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/dbccccccc/ttsfm?style=social)](https://github.com/dbccccccc/ttsfm)

> ⚠️ **Disclaimer**  
> This project is for learning & testing purposes only. For production use, please use the [official OpenAI TTS service](https://platform.openai.com/docs/guides/audio).

> 🚨 **IMPORTANT DEVELOPMENT NOTICE** 🚨  
> ⚠️ The v2 branch is currently under active development and is not recommended for production use. 
> 📚 For stable documentation and usage, please refer to the [v1 documentation](v1/README_v1.md).

English | [中文](README_CN.md)

## 🌟 Project Overview

TTSFM is a API server that's fully compatible with OpenAI's Text-to-Speech (TTS) API format.

> 🎮 Try it now: [Official Demo](https://ttsapi.site/)

## 🏗️ Project Structure

```text
ttsfm/
├── app.py              # Main Flask application
├── celery_worker.py    # Celery configuration and tasks
├── requirements.txt    # Python dependencies
├── static/            # Frontend resources
│   ├── index.html     # English interface
│   ├── index_zh.html  # Chinese interface
│   ├── script.js      # Frontend JavaScript
│   └── styles.css     # Frontend styles
├── voices/            # Voice samples
├── Dockerfile         # Docker configuration
├── docker-entrypoint.sh # Docker startup script
├── .env.example       # Environment variables template
├── .env              # Environment variables
├── .gitignore        # Git ignore rules
├── LICENSE           # MIT License
├── README.md         # English documentation
├── README_CN.md      # Chinese documentation
├── test_api.py       # API test suite
├── test_queue.py     # Queue test suite
└── .github/          # GitHub workflows
```

## 🚀 Quick Start

### System Requirements
- Python 3.13 or higher
- Redis server
- Docker (optional)

### Using Docker (Recommended)
```bash
# Pull the latest image
docker pull dbcccc/ttsfm:latest

# Run the container
docker run -d \
  --name ttsfm \
  -p 7000:7000 \
  -p 6379:6379 \
  -v $(pwd)/voices:/app/voices \
  dbcccc/ttsfm:latest
```

### Manual Installation
1. Clone the repository:
```bash
git clone https://github.com/dbccccccc/ttsfm.git
cd ttsfm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Redis server:
```bash
# On Windows
redis-server

# On Linux/macOS
sudo service redis-server start
```

4. Start Celery worker:
```bash
celery -A celery_worker.celery worker --pool=solo -l info
```

5. Start the server:
```bash
# Development (not recommended for production)
python app.py

# Production (recommended)
waitress-serve --host=0.0.0.0 --port=7000 app:app
```

### Environment Variables
Copy `.env.example` to `.env` and modify as needed:
```bash
cp .env.example .env
```

## 🔧 Configuration

### Server Configuration
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 7000)
- `VERIFY_SSL`: SSL verification (default: true)
- `MAX_QUEUE_SIZE`: Maximum queue size (default: 100)
- `RATE_LIMIT_REQUESTS`: Rate limit requests per window (default: 30)
- `RATE_LIMIT_WINDOW`: Rate limit window in seconds (default: 60)

### Celery Configuration
- `CELERY_BROKER_URL`: Redis broker URL (default: redis://localhost:6379/0)
- `CELERY_RESULT_BACKEND`: Redis result backend URL (default: redis://localhost:6379/0)

## 📚 API Documentation

### Text-to-Speech
```http
POST /v1/audio/speech
```

Request body:
```json
{
  "input": "Hello, world!",
  "voice": "alloy",
  "response_format": "mp3"
}
```

### Queue Status
```http
GET /api/queue-size
```

### Voice Samples
```http
GET /api/voice-sample/{voice}
```

### Version
```http
GET /api/version
```

## 🧪 Testing
Run the test suite:
```bash
python test_api.py
python test_queue.py
```

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- [OpenAI](https://openai.com/) for the TTS API format
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Celery](https://docs.celeryq.dev/) for task queue management
- [Waitress](https://docs.pylonsproject.org/projects/waitress/) for the production WSGI server 
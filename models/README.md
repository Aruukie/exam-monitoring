# Exam Proctoring System with PT Algorithm

AI-powered behavior detection for exam monitoring.

## Setup

1. Install dependencies:
pip install -r requiremejnts.txt

2. Configure database in `app/db.py`

3. Run SQL schema in SSMS

4. Start server:
python -m app.main

5. Open browser: `http://localhost:8000`

## Replace PT Algorithm

Edit `app/cv_processor.py` -> `detect_behaviors()` method with your model.

## API Endpoints

- POST /api/process-frakme - Process webcam frame
- GET /api/behavior-events?camera_id=1 - Get recent behaviors
- POST /api/behavior-event - Manual behavior logging
- GET /api/cameras - List cameras

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m app.main

# Open browser
http://localhost:8000



Replace the PT algorithm in app/cv_processor.py line 55-75 with your actual model!

This is a complete working system ready to integrate your behavior detection model!
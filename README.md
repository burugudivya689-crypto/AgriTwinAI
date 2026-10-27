# AgriTwinAI

AgriTwinAI is an AI-based crop digital twin for crop health and yield prediction. The project uses a synthetic agriculture dataset to train classification and regression models, then exposes them through a Django backend and a beginner-friendly dashboard.

## Project structure

- `dataset/05_smart_agriculture_crop_digital_twin.csv` — source dataset
- `artifacts/` — saved trained models and evaluation metadata
- `dashboard/` — Django app with dashboard UI and API views
- `agritwin/` — Django project settings and routing
- `train_models.py` — data processing, model comparison, and artifact generation
- `requirements.txt` — Python dependencies
- `manage.py` — Django entry point

## Setup instructions

1. Open PowerShell in the project folder.
2. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
4. Train the machine learning models:
   ```powershell
   python train_models.py
   ```
5. Start the Django app:
   ```powershell
   python manage.py runserver
   ```
6. Open the browser at `http://127.0.0.1:8000/`

## API endpoints

- `/api/status/`
- `/api/options/`
- `/api/predict/`
- `/api/simulate/`
- `/api/fields/`
- `/api/metrics/`

## Notes

- The project deliberately uses the provided synthetic dataset as a training source and does not claim live field monitoring.
- The what-if comparison is model-based and is not a guaranteed physical simulation.

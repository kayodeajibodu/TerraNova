# TerraNova
A Disaster Recovery Cost Forecasting Framework that provides accurate, real-time projections of recovery expenditures when a disaster is declared. TerraNova will leverage historical disaster data, financial obligations, and socio-economic indicators to predict recovery costs at the earliest possible stage of a disaster.
uvicorn api.main:app --reload
streamlit run dashboard/app.py 

# install required libraries
pip install -r requirements.txt

# Train models first
python run_pipeline.py

# Start API (in one terminal)
uvicorn api.main:app --reload
http://localhost:8000/docs

# Start mlflow ui (in another)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Start dashboard (in another)
streamlit run dashboard/app.py
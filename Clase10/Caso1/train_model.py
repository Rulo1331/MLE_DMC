import pandas as pd
from pycaret.clustering import *
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from pycaret.clustering import setup, save_model
# Cargar datos
df = pd.read_csv("D:\CURSOS\DMC\Especialización en Machine Learning Engineer\DMC_codigos\MLE_DMC\Clase10\Caso1\customer_segmentation.csv")

# Configurar MLflow local
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("clustering_cust")

# Configuración de PyCaret
s = setup(
    data=df.drop(columns=["customer_id"]),  # excluimos ID
    session_id=123,
    log_experiment=True,
    experiment_name="clustering_cust",
    verbose=True,
    profile=False,
    use_gpu=False # Al ser un cluster de un modelo no supervisado, puedes elegir el uso de un GPU para el despliegue
)

# Comparar modelos de clustering automáticamente
best_model = compare_models()

# Registrar el mejor modelo en MLflow
mlflow.pycaret.log_model(best_model, "best_clustering_model")

# Guardar el modelo localmente
save_model(best_model, "cluster_model")

print("Entrenamiento y registro completado.")

# mlflow ui --backend-store-uri file:./mlruns --port 5000
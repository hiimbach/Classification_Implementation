"""
THE BASE MLCHAIN SERVER 
"""
# Import mlchain 
from mlchain.base import ServeModel
from mlchain import mlconfig 

# Init classifier
from tools.infer import MushroomClassifier # Import your class here 
model = MushroomClassifier(file_names_path=mlconfig.file_names_path, model_path=mlconfig.model_path)




# Wrap your class by mlchain ServeModel
serve_model = ServeModel(model)

# GO TO CONSOLE: 
# mlchain run -c mlconfig.yaml 
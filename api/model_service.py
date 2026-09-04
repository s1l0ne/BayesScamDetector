from pathlib import Path
import joblib
import json


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models/'
METADATA_JSON_PATH = BASE_DIR / 'models' / 'metadata.json'


class ModelService:
    def __init__(self, metadata_json_path = METADATA_JSON_PATH):
        with open(metadata_json_path, 'r', encoding='utf-8') as file:
            self.metadata = json.load(file)

        self.current_model_name = self.models()[0]
        self.model = joblib.load(MODELS_DIR / self.metadata[self.current_model_name]['file'])


    def models(self) -> tuple:
        return tuple(model_name for model_name in self.metadata)


    def model_data(self, model_name: str):
        return self.metadata[model_name]


    def set_model(self, model_name: str):
        if model_name not in self.models():
            raise ValueError("Model with such name doesn't exists")

        self.current_model_name = model_name
        self.model = joblib.load(MODELS_DIR / model_name)

if __name__ == '__main__':
    model_service = ModelService()
    print(model_service.current_model_name)
    print(model_service.model_data(model_service.models()[0]))

    print(model_service.model.predict_proba(['john']))
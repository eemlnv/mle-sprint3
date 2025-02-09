# импортируйте необходимую библиотеку
from catboost import CatBoostClassifier

def load_churn_model(model_path: str):
    """Загружаем обученную модель оттока.
    Args:
        model_path (str): Путь до модели.
    """
    try:
        print("Start model loading")
        model = CatBoostClassifier()
        model.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
    print("Model loaded successfully")
    return model

if __name__ == "__main__":
    # вызовите функцию load_churn_model с нужным путём
    # ваш код здесь  
    # выведите параметры модели через print(f"Model parameter names: {}") 
    # ваш код здесь 
    model = load_churn_model(model_path='models/catboost_churn_model.bin')
    print(f'Model parameter names: {model.feature_names_}')
    
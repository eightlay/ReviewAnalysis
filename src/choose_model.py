import numpy as np
from typing import List, Tuple, Any
from sklearn.metrics import accuracy_score


def ChooseBestModel(models_: List[Any],
                    train_data: Tuple[np.ndarray],
                    test_data: Tuple[np.ndarray]):
    """
        Takes list of potential models and returns the most accurate model 
        and its accuracy.

        Parameters
        ----------
        - models_ : list of models to choose each model must support 'fit' and
                    'predict' methods
        - train_date : tuple {trainX: np.ndarray, trainY: np.ndarray}
                       data to fit model
        - test_date : tuple {testX: np.ndarray, testY: np.ndarray}
                      data to test fitted model

        Returns
        ------- 
        tuple {the most accurate model: any model class, its accuracy: float}
    """
    models = models_.copy()

    # Fit each model
    for model in models:
        model.fit(train_data[0], train_data[1])

    # Calculate accuracy
    scores = []
    for model in models:
        pred = model.predict(test_data[0])
        score = accuracy_score(test_data[1], pred)
        scores.append(score)

    # Return the most accurate model and its score
    return models[np.argmax(scores)], max(scores)

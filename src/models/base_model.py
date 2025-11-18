from abc import ABC, abstractmethod

class BaseModel(ABC):
    # Question: Is it better to write fit? train? or what?
    @abstractmethod
    def fit(self, X, y):
        """Fit the model with X and y.

        Args:
            X (_type_): Features for training.
            y (_type_): Target values for training.
        """
        pass
    @abstractmethod
    def predict(self, X, return_std=False):
        """
        Predict target values of X matrix

        Args:
            X (_type_): Features for prediction.
            return_std (bool, optional): Whether to return uncertainty.
            Defaults to False.
        Returns:
            _type_: Predicted target values.
            _type_: (optional) Uncertainty estimates.

        """
        pass

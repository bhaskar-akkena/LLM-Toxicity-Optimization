## @file VariationOperators.py
#  @brief Abstract base class for all variation operators used in prompt evolution.
#
#  This file defines the VariationOperator interface that all mutation or crossover
#  operators must inherit from. It enforces a standard interface for applying variation
#  to input prompts or embeddings.

from abc import ABC, abstractmethod

## @class VariationOperator
#  @brief Abstract base class for all variation operators (e.g., mutation, crossover).
#
#  Each variation operator must implement the `apply()` method, which defines how
#  the input string is modified. Additional metadata can be attached for experiment tracking.
class VariationOperator(ABC):
    """
    Abstract base class for all variation operators (mutation, crossover, etc.).
    Each operator must implement the `apply()` method, which returns a modified string.
    """

    def __init__(self, name=None, operator_type="mutation", description=""):
        """
        :param name: Name of the operator (used in metadata logs).
        :param operator_type: 'mutation', 'crossover', or 'hybrid'.
        :param description: Short description of what the operator does.
        """
        self.name = name or self.__class__.__name__
        self.operator_type = operator_type
        self.description = description

    @abstractmethod
    def apply(self, text: str) -> str:
        """
        Apply the variation to a given input string.
        Must be implemented by all subclasses.

        :param text: Input string (e.g., generated_response)
        :return: Modified output string
        """
        pass

    def __str__(self):
        return f"{self.name} ({self.operator_type})"

    def get_metadata(self) -> dict:
        """
        Optional method to retrieve metadata about the operator.
        Useful for logging or experiment tracking.
        """
        return {
            "name": self.name,
            "type": self.operator_type,
            "description": self.description
        }
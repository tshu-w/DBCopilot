from functools import wraps


class APICostCalculator:
    # fmt: off
    _model_cost_per_1m_tokens = {
        # https://openai.com/api/pricing/
        "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
        "gpt-3.5-turbo-0125": {"prompt": 0.5, "completion": 1.5},
        "gpt-3.5-turbo-instruct": {"prompt": 1.5, "completion": 2.0},
        "gpt-4o": {"prompt": 5, "completion": 15},
        "gpt-4o-2024-05-13": {"prompt": 5, "completion": 15},
        "gpt-4-turbo": {"prompt": 10, "completion": 30},
        "gpt-4-turbo-2024-04-09": {"prompt": 10, "completion": 30},
        "gpt-4": {"prompt": 30, "completion": 60},
        # https://platform.openai.com/docs/deprecations/
        "gpt-3.5-turbo-0301": {"prompt": 1.5, "completion": 2.0},
        "gpt-3.5-turbo-0613": {"prompt": 1.5, "completion": 2.0},
        "gpt-3.5-turbo-16k-0613": {"prompt": 3, "completion": 4.0},
        "gpt-3.5-turbo-1106": {"prompt": 1.0, "completion": 2.0},
    }
    # fmt: on

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        if model_name not in self._model_cost_per_1m_tokens:
            raise ValueError(f"Unknown model name: {model_name}")
        self._model_name = model_name
        self._cost = 0

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            response = func(*args, **kwargs)
            cost = (
                self._model_cost_per_1m_tokens[self._model_name]["prompt"]
                * response.usage.prompt_tokens
                + self._model_cost_per_1m_tokens[self._model_name]["completion"]
                * response.usage.completion_tokens
            ) / 1000000.0
            self._cost += cost
            return response

        return wrapper

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value: int):
        self._cost = value

import collections

import guidance

MODEL_COST_PER_1K_TOKENS = {
    "gpt-4": 0.03,
    "gpt-4-0314": 0.03,
    "gpt-4-completion": 0.06,
    "gpt-4-0314-completion": 0.06,
    "gpt-4-32k": 0.06,
    "gpt-4-32k-0314": 0.06,
    "gpt-4-32k-completion": 0.12,
    "gpt-4-32k-0314-completion": 0.12,
    "gpt-3.5-turbo": 0.002,
    "gpt-3.5-turbo-0301": 0.002,
    "text-ada-001": 0.0004,
    "ada": 0.0004,
    "text-babbage-001": 0.0005,
    "babbage": 0.0005,
    "text-curie-001": 0.002,
    "curie": 0.002,
    "text-davinci-003": 0.02,
    "text-davinci-002": 0.02,
    "code-davinci-002": 0.02,
}


def get_openai_token_cost_for_model(
    model_name: str, num_tokens: int, is_completion: bool = False
) -> float:
    suffix = "-completion" if is_completion and model_name.startswith("gpt-4") else ""
    model = model_name.lower() + suffix
    if model not in MODEL_COST_PER_1K_TOKENS:
        raise ValueError(
            f"Unknown model: {model_name}. Please provide a valid OpenAI model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_TOKENS.keys())
        )
    return MODEL_COST_PER_1K_TOKENS[model] * num_tokens / 1000


class OpenAI(guidance.llms.OpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage = collections.Counter()

    def get_usage_cost_usd(self, usage=None):
        usage = self.usage if usage is None else usage
        return get_openai_token_cost_for_model(
            self.model_name, usage["completion_tokens"], is_completion=True
        ) + get_openai_token_cost_for_model(self.model_name, usage["prompt_tokens"])

    def session(self, asynchronous=False):
        if asynchronous:
            return OpenAISession(self)
        else:
            return guidance.llms._llm.SyncSession(OpenAISession(self))


class OpenAISession(guidance.llms._openai.OpenAISession):
    async def __call__(self, *args, **kwargs):
        result = await super().__call__(*args, **kwargs)
        for k, v in result.get("usage", {}).items():
            self.llm.usage[k] += v

        return result


guidance.llms.OpenAI = OpenAI


if __name__ == "__main__":
    guidance.llm = guidance.llms.OpenAI("text-davinci-003")
    response = guidance(
        """The best thing about the beach is {{~gen 'best' temperature=0.7 max_tokens=7 stream=False}}"""
    )()
    print(response.llm.usage)
    response = guidance(
        """The best thing about the sunshine is {{~gen 'best' temperature=0.7 max_tokens=7 stream=False}}"""
    )()
    print(response.llm.usage)
    print(guidance.llm.usage)

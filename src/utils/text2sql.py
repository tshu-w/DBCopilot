import guidance
from tenacity import retry, stop_after_attempt, wait_exponential

from . import openai_with_usage  # noqa: F401

SINGLE_DB_PROMPT = """
{{#system~}}
{{llm.default_system_prompt}}
{{~/system}}

{{#user~}}
### Complete sqlite SQL query only and with no explanation
### Sqlite SQL tables, with their properties:
#
# {{#each tables}}{{this.name}}({{#each this.columns}}"{{this}}"{{#unless @last}}, {{/unless}}{{/each}})
# {{/each}}
### {{question}}
SELECT
{{~/user}}

{{#assistant~}}
{{gen "query" temperature=0 stop=";"}}
{{~/assistant}}
"""

MULTI_DB_PROMPT = """
{{#system~}}
{{llm.default_system_prompt}}
{{~/system}}

{{#user~}}
### Complete sqlite SQL query only and with no explanation
### Sqlite SQL databases, with their tables and properties:
#
# {{#each databases}}{{this.name}}
# {{#each this.tables}}{{this.name}}({{#each this.columns}}"{{this}}"{{#unless @last}}, {{/unless}}{{/each}})
# {{/each}} {{#unless @last}}\n# {{/unless}}{{/each}}
### {{question}}
SELECT
{{~/user}}

{{#assistant~}}
{{gen "query" temperature=0 stop=";"}}
{{~/assistant}}
"""


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1))
def text2sql(
    question: str,
    schemas: list[dict],
    model: guidance.llms.LLM | str,
):
    if isinstance(model, str):
        model = guidance.llms.OpenAI(model)
    assert model.chat_mode
    if len(schemas) == 1:
        program = guidance(SINGLE_DB_PROMPT)
        tables = schemas[0]["tables"]
        response = program(question=question, tables=tables, llm=model)
    else:
        program = guidance(MULTI_DB_PROMPT)
        response = program(question=question, databases=schemas, llm=model)

    joined_query = " ".join(map(str.strip, response["query"].split("\n")))
    return (
        f"SELECT {joined_query}",
        model.usage if hasattr(model, "usage") else None,
    )


# fmt: off
if __name__ == "__main__":
    sql, usage = text2sql(
        question="Return the names of the contestants whose names contain the substring 'Al'.",
        schemas=[
            {
                "name": "singer",
                "tables": [
                    {"name": "singer", "columns": ["singer_id", "name", "birth_year", "net_worth_millions", "citizenship"]},
                    {"name": "song", "columns": ["song_id", "title", "singer_id", "sales", "highest position"]},
                ]
            },
            {
                "name": "voting",
                "tables": [
                    {"name": "area_code_state", "columns": ["area_code", "state"]},
                    {"name": "contestants", "columns": ["contestant_number", "contestant_name"]},
                    {"name": "votes", "columns": ["vote_id", "phone_number", "state", "contestant_number", "created"]},
                ]
            },
        ],
        model="gpt-3.5-turbo",
    )
    print(sql)
    print(usage)

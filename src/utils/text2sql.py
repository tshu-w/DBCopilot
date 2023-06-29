import guidance


def text2sql(
    question: str,
    database: list[dict],
    model_name: str = "gpt-3.5-turbo",
):
    model = guidance.llms.OpenAI(model_name)
    if model.chat_mode:
        program = guidance(
            """
{{#system~}}
{{llm.default_system_prompt}}
{{~/system}}

{{#user~}}
### Complete sqlite SQL query only and with no explanation
### Sqlite SQL tables, with their properties:
#
# {{#each database}}{{this.name}}({{#each this.columns}}{{this}}{{#unless @last}}, {{/unless}}{{/each}})
# {{/each}}
### {{question}}
SELECT
{{~/user}}

{{#assistant~}}
{{gen "query" temperature=0 stop=";"}}
{{~/assistant}}"""
        )
    else:
        program = guidance(
            """### Complete sqlite SQL query only and with no explanation
### Sqlite SQL tables, with their properties:
#
# {{#each database}}{{this.name}}({{#each this.columns}}{{this}}{{#unless @last}}, {{/unless}}{{/each}})
# {{/each}}
### {{question}}
SELECT {{~gen "query" temperature=0 stop=";"}}"""
        )

    response = program(question=question, database=database, llm=model)
    joined_query = " ".join(map(str.strip, response["query"].split("\n")))
    return f"SELECT {joined_query}"


# fmt: off
if __name__ == "__main__":
    print(
        text2sql(
            question="Return the names of the contestants whose names contain the substring 'Al'.",
            database=[
                {"name": "AREA_CODE_STATE", "columns": {"area_code", "state"}},
                {"name": "CONTESTANTS", "columns": {"contestant_number", "contestant_name"}},
                {"name": "VOTES", "columns": {"vote_id", "phone_number", "state", "contestant_number", "created"}},
            ],
        )
    )

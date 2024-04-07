import asyncio
import re

import guidance
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

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

COT_PROMPT = """
{{#system~}}
{{llm.default_system_prompt}}
{{~/system}}

{{#user~}}
Based on the provided natural language question, find the database that can best answer this question from the list schemas below. Only output the corresponding database schema number in the [id] format, without any additional information.

Question: {{question}}

Sqlite SQL databases, with their tables and properties:

{{#each databases}}[{{@index}}] {{this.name}}
{{#each this.tables}}{{this.name}}({{#each this.columns}}"{{this}}"{{#unless @last}}, {{/unless}}{{/each}})
{{/each}}{{#unless @last}}\n{{/unless}}{{/each}}
{{~/user}}

{{#assistant~}}
{{gen "best" temperature=0}}
{{~/assistant}}

{{#user~}}
### Complete sqlite SQL query only and with no explanation for the most relevant schema.
#
# {{#each (select_database best databases)}}{{this.name}}({{#each this.columns}}"{{this}}"{{#unless @last}}, {{/unless}}{{/each}})
# {{/each}}
### {{question}}
SELECT
{{~/user}}

{{#assistant~}}
{{gen "query" temperature=0 stop=";"}}
{{~/assistant}}
"""


def select_database(best, databases):
    idx = re.search(r"([\d+])", best)
    if idx:
        idx = int(idx.group(1))
        idx = max(idx, 0)
        idx = min(idx, len(databases) - 1)
    else:
        idx = 0
    return databases[idx]["tables"]


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
async def text2sql(
    question: str,
    schemas: list[dict],
    model: guidance.llms.LLM,
    chain_of_thought: bool = False,
) -> str:
    assert model.chat_mode
    if len(schemas) == 1:
        tables = schemas[0]["tables"]
        program = guidance(SINGLE_DB_PROMPT, llm=model, async_mode=True, silent=True)
        response = await program(question=question, tables=tables)
    else:
        if chain_of_thought:
            program = guidance(COT_PROMPT, llm=model, async_mode=True, silent=True)
            response = await program(
                question=question,
                databases=schemas,
                select_database=select_database,
            )
        else:
            program = guidance(MULTI_DB_PROMPT, llm=model, async_mode=True, silent=True)
            response = await program(question=question, databases=schemas)

    joined_query = " ".join(map(str.strip, response["query"].split("\n")))
    sql = f"SELECT {joined_query}"

    if chain_of_thought:
        idx = re.search(r"([\d+])", response["best"])
        if idx:
            idx = int(idx.group(1))
        else:
            idx = 0
        return sql, idx
    return sql


async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await tqdm_asyncio.gather(*(sem_coro(c) for c in coros))


# Hide program executor tracebacks
async def program_executor_run(self, llm_session):
    """Execute the program."""
    self.llm_session = llm_session
    # first parse all the whitespace control
    # self.whitespace_control_visit(self.parse_tree)

    # now execute the program
    self.program._variables["@raw_prefix"] = ""
    await self.visit(
        self.parse_tree,
        guidance._variable_stack.VariableStack([self.program._variables], self),
    )


guidance._program_executor.ProgramExecutor.run = program_executor_run


# fmt: off
if __name__ == "__main__":
    model = guidance.llms.OpenAI("gpt-3.5-turbo", caching=False)
    question="Return the names of the contestants whose names contain the substring 'Al'."
    schemas=[
        {
            "name": "singer",
            "tables": [
                {"name": "singer", "columns": ["singer_id", "name", "birth_year", "net_worth_millions", "citizenship"]},
                {"name": "song", "columns": ["song_id", "title", "singer_id", "sales", "highest position"]},
            ]
        },
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
        {
            "name": "singer",
            "tables": [
                {"name": "singer", "columns": ["singer_id", "name", "birth_year", "net_worth_millions", "citizenship"]},
                {"name": "song", "columns": ["song_id", "title", "singer_id", "sales", "highest position"]},
            ]
        },
        {
            "name": "singer",
            "tables": [
                {"name": "singer", "columns": ["singer_id", "name", "birth_year", "net_worth_millions", "citizenship"]},
                {"name": "song", "columns": ["song_id", "title", "singer_id", "sales", "highest position"]},
            ]
        },
    ]

    sqls = asyncio.run(
        gather_with_concurrency(
            3,
            *[
                text2sql(question=question, schemas=schemas, model=model, chain_of_thought=_)
                for _ in range(2)
            ]
        )
    )
    print(sqls)
    print(model.usage)

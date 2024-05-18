import re

from diskcache import Cache
from jinja2 import Template
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.contrib.concurrent import thread_map

cache = Cache("results/diskcache/text2sql_ict")

SINGLE_DB_TPL = Template(
    """
### Complete sqlite SQL query only and with no explanation
### Sqlite SQL tables, with their properties:
#
# {% for db in databases %}{% for table in db.tables %}{{ table.name }}({% for column in table.columns %}"{{ column }}"{% if not loop.last %}, {% endif %}{% endfor %})
# {% endfor %}{% if not loop.last %}
# {% endif %}{% endfor %}
### {{ question }}
SELECT
"""
)

MULTI_DB_TPL = Template(
    """
### Complete sqlite SQL query only and with no explanation
### Sqlite SQL databases, with their tables and properties:
#
# {% for db in databases %}{{ db.name }}
# {% for table in db.tables %}{{ table.name }}({% for column in table.columns %}"{{ column }}"{% if not loop.last %}, {% endif %}{% endfor %})
# {% endfor %}{% if not loop.last %}
# {% endif %}{% endfor %}
### {{ question }}
SELECT
"""
)

COT_TPL = Template(
    """
Based on the provided natural language question, find the database that can best answer this question from the list schemas below. Only output the corresponding database schema number in the [id] format, without any additional information.

Question: {{ question }}

Sqlite SQL databases, with their tables and properties:

{% for database in databases %}[{{ loop.index0 }}] {{ database.name }}
{% for table in database.tables %}{{ table.name }}({% for column in table.columns %}"{{ column }}"{% if not loop.last %}, {% endif %}{% endfor %})
{% endfor %}{% if not loop.last %}

{% endif %}{% endfor %}
"""
)

DEFAULT_CLIENT = OpenAI()


@cache.memoize(name="chat_complete")
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
def chat_complete(
    messages,
    model,
    client=DEFAULT_CLIENT,
    **kwargs,
):
    response = client.chat.completions.create(messages=messages, model=model, **kwargs)
    return response


def text2sql(
    instance: dict,
    model: str = "gpt-3.5-turbo",
    chain_of_thought: bool = False,
) -> str:
    question = instance["question"]
    schemas = instance["schemas"]
    examples = instance["examples"]
    if chain_of_thought:
        messages = [
            {
                "role": "user",
                "content": COT_TPL.render(
                    question=question,
                    databases=schemas,
                ),
            }
        ]
        response = chat_complete(
            messages=messages, model=model, seed=42, temperature=0.0, max_tokens=1000
        )
        content = response.choices[0].message.content
        idx = re.search(r"\[(\d+)\]", content)
        if idx:
            idx = int(idx.group(1))
            idx = max(idx, 0)
            idx = min(idx, len(schemas) - 1)
        else:
            idx = 0

        schemas = schemas[idx : idx + 1]

    messages = []
    for exp in examples or []:
        tpl = SINGLE_DB_TPL if len(schemas) == 1 else MULTI_DB_TPL
        prompt = tpl.render(question=exp["question"], databases=exp["schemas"])
        messages.append({"role": "user", "content": prompt})
        messages.append(
            {"role": "assistant", "content": exp["sql"].removeprefix("SELECT")}
        )

    tpl = SINGLE_DB_TPL if len(schemas) == 1 else MULTI_DB_TPL
    prompt = tpl.render(question=question, databases=schemas)
    messages.append({"role": "user", "content": prompt})
    response = chat_complete(
        messages=messages, model=model, seed=42, temperature=0.0, max_tokens=1000
    )
    content = response.choices[0].message.content
    joined = " ".join(map(str.strip, content.split("\n")))
    sql = f"SELECT {joined}"

    return sql


# fmt: off
if __name__ == "__main__":
    model = "gpt-3.5-turbo"
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
            "name": "voting",
            "tables": [
                {"name": "area_code_state", "columns": ["area_code", "state"]},
                {"name": "contestants", "columns": ["contestant_number", "contestant_name"]},
                {"name": "votes", "columns": ["vote_id", "phone_number", "state", "contestant_number", "created"]},
            ]
        },
    ]

    instances = [
        {
            "question": question,
            "schemas": schemas,
        } for _ in range(2)
    ]
    sqls = thread_map(
        text2sql,
        instances,
        [model] * 2,
        [False, True],
    )
    print(sqls)

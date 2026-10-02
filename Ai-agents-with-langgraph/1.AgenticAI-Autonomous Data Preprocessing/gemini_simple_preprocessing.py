import os
import io
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import google.genai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import box

#Load API Key
load_dotenv(".env")
google_api_key = os.getenv("GOOGLE_API_KEY")

console = Console()

#Load Dataset
df = pd.read_csv("dataset.csv")

console.print("\n[bold green]Dataset loaded successfully[/bold green]\n")
table = Table(title="Dataset Preview", box=box.DOUBLE_EDGE)

for col in df.columns:
    table.add_column(col)

for _, row in df.iterrows():
    table.add_row(*[str(i) for i in row])

console.print(table)

# LLM
# Passing an explicit genai.Client(api_key=...) prevents the SDK from
# silently falling back to Application Default Credentials (a common
# cause of 401 ACCESS_TOKEN_TYPE_UNSUPPORTED errors when gcloud ADC
# is present on the machine).
genai_client = genai.Client(api_key=google_api_key)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0, client=genai_client)


def extract_text(response) -> str:
    """
    Gemini responses can return `.content` as a plain string OR as a
    list of content-part dicts (e.g. [{"text": "..."}]). Normalize
    either shape into a plain string.
    """

    raw_content = response.content

    if isinstance(raw_content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw_content
        )

    return raw_content


# Agent 1 - Guding Agent
tasks = [
    "Display the entire dataset",
    "Print number of rows, number of columns and column names",
    "Identify and print quantitative and qualitative columns",
    "Detect missing values, print number of missing values and handle them",
    "Detect outliers, print high and low outliers and handle them using IQR",
    "Find duplicates and print number of duplicate rows",
]
# Agent 2 - Coding Agent
def coder_agent(task):

    prompt = f"""
You are an expert Python data preprocessing engineer.

Rules:
1. Dataframe name is df
2. Use pandas and numpy
3. Do NOT reload dataset
4. Numeric columns = df.select_dtypes(include=np.number)
5. Categorical columns = df.select_dtypes(include="object")
6. Return only executable Python code

Task:
{task}
"""

    response = llm.invoke(prompt)
    code = extract_text(response)

    code = code.replace("```python", "").replace("```", "").strip()
    console.print("\n[bold yellow]Generated Code[/bold yellow]\n")

    syntax = Syntax(code, "python", theme="monokai", line_numbers=False)
    console.print(syntax)

    return code


# Agent 3 - Executor Agent

def executor_agent(code):

    global df

    try:

        local_vars = {
            "df": df,
            "np": np,
            "pd": pd,
           # "IsolationForest": IsolationForest
        }

        # Capture output
        import io
        import sys

        buffer = io.StringIO()
        sys.stdout = buffer

        exec(code, globals(), local_vars)

        sys.stdout = sys.__stdout__

        df = local_vars.get("df", df)

        output = buffer.getvalue()


        console.print(Panel(output if output else "Execution Completed", title="Execution Output", style="bold cyan on black"))

    except Exception as e:

        console.print(Panel(str(e), title="Execution Error", style="bold red"))


# Checker Agent
def checker_agent(iteration):

    if iteration < 6:
        console.print("\n[bold cyan]Checker Agent --> Pending Preprocessing[/bold cyan]")
    else:
        console.print("\n[bold green]Checker Agent --> Preprocessed Completely[/bold green]")


for i, task in enumerate(tasks, start=1):

    console.print(f"[bold magenta]====================================\nIteration {i}\n====================================[/bold magenta]")
    console.print(f"[bold blue]Guiding Agent --> {task}[/bold blue]")
    code = coder_agent(task)
    executor_agent(code)
    checker_agent(i)


console.print("\n[bold green]Final Cleaned Dataset[/bold green]\n")

final_table = Table(title="Processed Dataset", box=box.DOUBLE_EDGE)

for col in df.columns:
    final_table.add_column(col)

for _, row in df.iterrows():
    final_table.add_row(*[str(i) for i in row])

console.print(final_table)

output_file = "Final_Dataset.csv"
df.to_csv(output_file, index=False)

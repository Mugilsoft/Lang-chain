# =========================================================
# AUTONOMOUS PREPROCESSING MULTI-AGENT SYSTEM
# WITH DEBUGGER AGENT (GEMINI VERSION)
# =========================================================

import os
import io
import sys
import traceback
import warnings
import logging

import pandas as pd
import numpy as np

from typing import (
    TypedDict,
    List,
    Optional
)

from dotenv import load_dotenv

import google.genai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import (
    StateGraph,
    END
)

from rich.console import Console

from rich.panel import Panel

from rich.syntax import Syntax

from rich.table import Table

from rich import box


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv(".env")

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)

DATASET_PATH = "dataset.csv"

OUTPUT_DATASET = "Final_Dataset.csv"


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

console = Console()


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(DATASET_PATH)

console.print(

    "\n[bold green]"
    "Dataset Loaded Successfully"
    "[/bold green]\n"
)


# =========================================================
# DATASET PREVIEW
# =========================================================

preview_table = Table(

    title="Dataset Preview",

    box=box.DOUBLE_EDGE
)

for col in df.columns:

    preview_table.add_column(col)

for _, row in df.head().iterrows():

    preview_table.add_row(
        *[str(i) for i in row]
    )

console.print(preview_table)


# =========================================================
# LLM
# =========================================================
# Passing an explicit genai.Client(api_key=...) prevents the
# SDK from silently falling back to Application Default
# Credentials (a common cause of 401 ACCESS_TOKEN_TYPE_UNSUPPORTED
# errors when gcloud ADC is present on the machine).

genai_client = genai.Client(
    api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(

    model="gemini-3.5-flash",

    temperature=0,

    client=genai_client
)


# =========================================================
# HELPER: NORMALIZE LLM RESPONSE CONTENT
# =========================================================
# Gemini responses can return `.content` as a plain string OR
# as a list of content-part dicts (e.g. [{"text": "..."}]).
# This normalizes either shape into a plain string.

def extract_text(response) -> str:

    raw_content = response.content

    if isinstance(raw_content, list):

        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw_content
        )

    return raw_content


# =========================================================
# PREPROCESSING TASKS
# =========================================================

PREPROCESSING_TASKS = [

    "Display dataset summary",

    "Print rows, columns and column names",

    "Identify numerical and categorical columns",

    "Handle missing values",

    "Handle outliers using IQR only for numeric columns",

    "Find duplicate rows"
]


# =========================================================
# STATE
# =========================================================

class PreprocessingState(TypedDict):

    df: pd.DataFrame

    tasks: List[str]

    task_index: int

    current_task: str

    generated_code: str

    preprocessing_output: str

    preprocessing_error: Optional[str]

    retry_count: int

    max_retries: int

    debug_history: List[str]

    last_error: str


# =========================================================
# PLANNER AGENT
# =========================================================

def planner_agent(state):

    task_index = state["task_index"]

    task = state["tasks"][task_index]

    console.print(

        f"\n[bold cyan]"
        f"PREPROCESSING TASK "
        f"{task_index + 1}"
        f"[/bold cyan]"
    )

    console.print(

        f"[yellow]{task}[/yellow]"
    )

    return {

        "current_task": task
    }


# =========================================================
# CODER AGENT
# =========================================================

def coder_agent(state):

    task = state["current_task"]

    prompt = f"""
You are an expert Python preprocessing engineer.

Rules:
1. Dataframe name is df
2. Use pandas and numpy only
3. Do NOT reload dataset
4. Return executable Python code only
5. Print meaningful output
6. Update df if preprocessing is applied

Task:
{task}
"""

    response = llm.invoke(prompt)

    code = extract_text(response).strip()

    code = code.replace(
        "```python",
        ""
    )

    code = code.replace(
        "```",
        ""
    )

    console.print(

        "\n[bold green]"
        "GENERATED CODE"
        "[/bold green]\n"
    )

    syntax = Syntax(

        code,

        "python",

        theme="monokai"
    )

    console.print(syntax)

    return {

        "generated_code": code
    }


# =========================================================
# EXECUTOR AGENT
# =========================================================

def executor_agent(state):

    code = state["generated_code"]

    dataframe = state["df"]

    local_vars = {

        "df": dataframe.copy(),

        "pd": pd,

        "np": np
    }

    buffer = io.StringIO()

    try:

        sys.stdout = buffer

        exec(code, {}, local_vars)

        sys.stdout = sys.__stdout__

        updated_df = local_vars.get(
            "df",
            dataframe
        )

        output = buffer.getvalue()

        console.print(

            Panel(

                output if output else "Execution Completed",

                title="Execution Output",

                style="bold green"
            )
        )

        return {

            "df": updated_df,

            "preprocessing_output": output,

            "preprocessing_error": None,

            "last_error": ""
        }

    except Exception:

        sys.stdout = sys.__stdout__

        error_message = traceback.format_exc()

        console.print(

            Panel(

                error_message,

                title="Execution Failed",

                style="bold red"
            )
        )

        return {

            "preprocessing_error": error_message,

            "last_error": error_message
        }


# =========================================================
# DEBUGGER AGENT
# =========================================================

def debugger_agent(state):

    console.print(

        "\n[bold red]"
        "DEBUGGER AGENT ACTIVATED"
        "[/bold red]"
    )

    task = state["current_task"]

    original_code = state["generated_code"]

    error_message = state["last_error"]

    debug_prompt = f"""
You are a senior Python debugging engineer.

Task:
{task}

Original Code:
{original_code}

Execution Error:
{error_message}

Rules:
1. Fix the error
2. Preserve original preprocessing logic
3. Dataframe name must remain df
4. Use pandas and numpy only
5. Return ONLY executable Python code
6. Do not explain anything
7. Handle numeric-only operations correctly
"""

    response = llm.invoke(debug_prompt)

    fixed_code = extract_text(response).strip()

    fixed_code = fixed_code.replace(
        "```python",
        ""
    )

    fixed_code = fixed_code.replace(
        "```",
        ""
    )

    console.print(

        "\n[bold yellow]"
        "FIXED CODE GENERATED"
        "[/bold yellow]\n"
    )

    syntax = Syntax(

        fixed_code,

        "python",

        theme="monokai"
    )

    console.print(syntax)

    debug_history = state["debug_history"]

    debug_history.append(error_message)

    return {

        "generated_code": fixed_code,

        "retry_count":
        state["retry_count"] + 1,

        "debug_history": debug_history
    }


# =========================================================
# EXECUTION ROUTER
# =========================================================

def execution_router(state):

    # SUCCESS
    if state["preprocessing_error"] is None:

        return "checker_agent"

    # MAX RETRIES REACHED
    if state["retry_count"] >= state["max_retries"]:

        return END

    # SEND TO DEBUGGER
    return "debugger_agent"


# =========================================================
# CHECKER AGENT
# =========================================================

def checker_agent(state):

    console.print(

        "\n[bold green]"
        "TASK COMPLETED SUCCESSFULLY"
        "[/bold green]"
    )

    return {}


# =========================================================
# NEXT TASK AGENT
# =========================================================

def next_task_agent(state):

    return {

        "task_index":
        state["task_index"] + 1,

        "retry_count": 0
    }


# =========================================================
# PREPROCESSING ROUTER
# =========================================================

def preprocessing_router(state):

    next_index = state["task_index"] + 1

    if next_index >= len(state["tasks"]):

        return END

    return "planner_agent"


# =========================================================
# BUILD GRAPH
# =========================================================

builder = StateGraph(
    PreprocessingState
)


# =========================================================
# ADD NODES
# =========================================================

builder.add_node(
    "planner_agent",
    planner_agent
)

builder.add_node(
    "coder_agent",
    coder_agent
)

builder.add_node(
    "executor_agent",
    executor_agent
)

builder.add_node(
    "debugger_agent",
    debugger_agent
)

builder.add_node(
    "checker_agent",
    checker_agent
)

builder.add_node(
    "next_task_agent",
    next_task_agent
)


# =========================================================
# ENTRY POINT
# =========================================================

builder.set_entry_point(
    "planner_agent"
)


# =========================================================
# MAIN FLOW
# =========================================================

builder.add_edge(
    "planner_agent",
    "coder_agent"
)

builder.add_edge(
    "coder_agent",
    "executor_agent"
)


# =========================================================
# EXECUTION ROUTING
# =========================================================

builder.add_conditional_edges(

    "executor_agent",

    execution_router,

    {

        "checker_agent":
        "checker_agent",

        "debugger_agent":
        "debugger_agent",

        END: END
    }
)


# =========================================================
# DEBUG LOOP
# =========================================================

builder.add_edge(
    "debugger_agent",
    "executor_agent"
)


# =========================================================
# CONTINUE TASKS
# =========================================================

builder.add_edge(
    "checker_agent",
    "next_task_agent"
)

builder.add_conditional_edges(

    "next_task_agent",

    preprocessing_router,

    {

        "planner_agent":
        "planner_agent",

        END: END
    }
)


# =========================================================
# COMPILE GRAPH
# =========================================================

graph = builder.compile()


# =========================================================
# VISUALIZE GRAPH
# =========================================================

from IPython.display import (
    Image,
    display
)

graph_png = (
    graph.get_graph()
    .draw_mermaid_png()
)

display(Image(graph_png))

print(
    graph.get_graph().draw_ascii()
)
# Save image
with open("langgraph.png", "wb") as f:
    f.write(graph_png)

print("Graph image saved successfully!")

# Display image
display(Image(graph_png))

# =========================================================
# RUN GRAPH
# =========================================================

final_state = graph.invoke({

    "df": df,

    "tasks": PREPROCESSING_TASKS,

    "task_index": 0,

    "current_task": "",

    "generated_code": "",

    "preprocessing_output": "",

    "preprocessing_error": None,

    "retry_count": 0,

    "max_retries": 3,

    "debug_history": [],

    "last_error": ""

},
config={
    "recursion_limit": 100
})


# =========================================================
# SAVE FINAL DATASET
# =========================================================

final_state["df"].to_csv(

    OUTPUT_DATASET,

    index=False
)

console.print(

    "\n[bold green]"
    "AUTONOMOUS PREPROCESSING PIPELINE COMPLETED"
    "[/bold green]"
)

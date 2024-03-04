from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from io import StringIO


# Function to extract tables from markdown content
def extract_markdown_tables(md_content):
    tables = []
    current_table = []
    in_table = False
    lines = md_content.split("\n")

    for line in lines:
        # Check for table start/end
        if re.match(r"\|.*\|", line):
            in_table = True
            current_table.append(line)
        elif in_table:
            # Table has ended
            tables.append("\n".join(current_table))
            current_table = []
            in_table = False

    # Add the last table if it wasn't added
    if in_table:
        tables.append("\n".join(current_table))

    return tables


# Function to extract and shorten the operation name if it's too long
def shorten_name(x):
    return x.split("/")[0].strip() if "/" in x else re.sub(r"\s*\([^)]*\)", "", x)


def plot(file_content, operation):
    # Extracting tables from the file content
    tables_md = extract_markdown_tables(file_content)

    # Convert markdown tables to Pandas DataFrames
    dataframes = []
    for table_md in tables_md:
        df = pd.read_table(StringIO(table_md), sep="|")
        df.columns = df.columns.str.strip()  # Clean column names
        df = df.apply(
            lambda x: x.str.strip() if x.dtype == "object" else x
        )  # Clean cell values
        df.dropna(axis=1, how="all", inplace=True)  # Drop columns with all NaN values
        dataframes.append(df)

    # Extracting table titles and operation values for each table
    title_pattern = r"\*\*(.*?)\s*\("
    titles = re.findall(title_pattern, file_content)

    # Preparing data for the visualization
    mlx_gpu_values = []
    mps_values = []
    cuda_values = []
    table_titles = []

    for i, df in enumerate(dataframes):
        if "Operation" in df.columns and operation in df["Operation"].values:
            op_row = df[df["Operation"] == operation]

            title = titles[i] if i < len(titles) else f"Table {i+1}"
            table_titles.append(shorten_name(title))

            # Extract mlx_gpu, mps, and cuda values
            mlx_gpu_val = (
                op_row["mlx_gpu"].values[0] if "mlx_gpu" in df.columns else np.nan
            )
            mps_val = op_row["mps"].values[0] if "mps" in df.columns else np.nan
            cuda_val = op_row["cuda"].values[0] if "cuda" in df.columns else np.nan

            mlx_gpu_values.append(pd.to_numeric(mlx_gpu_val, errors="coerce"))
            mps_values.append(pd.to_numeric(mps_val, errors="coerce"))
            cuda_values.append(pd.to_numeric(cuda_val, errors="coerce"))

    plt.figure(figsize=(12, 6))

    # Colors for mlx_gpu, mps, and cuda
    mlx_gpu_color = "skyblue"
    mps_color = "salmon"
    cuda_color = "lightgreen"

    bar_width = 0.25
    indices = np.arange(len(table_titles))

    # Plot each set of bars
    for i in indices:
        # Plot mlx_gpu and annotate
        bar_mlx_gpu = plt.bar(
            i - bar_width,
            mlx_gpu_values[i],
            bar_width,
            color=mlx_gpu_color,
            label="mlx_gpu" if i == 0 else "",
        )
        if not np.isnan(mlx_gpu_values[i]):
            plt.annotate(
                f"{mlx_gpu_values[i]:.2f}",
                (
                    bar_mlx_gpu[0].get_x() + bar_mlx_gpu[0].get_width() / 2,
                    bar_mlx_gpu[0].get_height(),
                ),
                ha="center",
                va="bottom",
            )

        # Plot mps and annotate
        bar_mps = plt.bar(
            i, mps_values[i], bar_width, color=mps_color, label="mps" if i == 0 else ""
        )
        if not np.isnan(mps_values[i]):
            plt.annotate(
                f"{mps_values[i]:.2f}",
                (
                    bar_mps[0].get_x() + bar_mps[0].get_width() / 2,
                    bar_mps[0].get_height(),
                ),
                ha="center",
                va="bottom",
            )

        # Plot cuda and annotate
        bar_cuda = plt.bar(
            i + bar_width,
            cuda_values[i],
            bar_width,
            color=cuda_color,
            label="cuda" if i == 0 else "",
        )
        if not np.isnan(cuda_values[i]):
            plt.annotate(
                f"{cuda_values[i]:.2f}",
                (
                    bar_cuda[0].get_x() + bar_cuda[0].get_width() / 2,
                    bar_cuda[0].get_height(),
                ),
                ha="center",
                va="bottom",
            )

    plt.xlabel("Chips/GPUs")
    plt.ylabel("Runtime (ms)")
    plt.title(f'"{operation}" benchmark')
    plt.xticks(indices, table_titles, rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    op_chart_table_titles_path = f"plot_{operation}.png"
    plt.savefig(op_chart_table_titles_path)
    plt.close()


if __name__ == "__main__":
    current_directory = Path(__file__).parent
    parent_directory = current_directory.parent
    file_path = f"{parent_directory}/benchmarks/average_benchmark.md"

    with open(file_path, "r") as file:
        file_content = file.read()

    for op in [
        "Linear",
        "Concat",
        "MatMul",
        "Softmax",
        "Conv2d",
        "BCE",
        "Sort",
        "Sigmoid",
    ]:
        plot(file_content, operation=op)

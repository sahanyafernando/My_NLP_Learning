import copy
import json
import pathlib


BASE_DIR = pathlib.Path(__file__).resolve().parent
SOURCE_NOTEBOOK = BASE_DIR / "Project_01_Public_Response_Analysis.ipynb"


def load_notebook():
    return json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))


def clone_cells(cells):
    return [copy.deepcopy(cell) for cell in cells]


def build_markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def build_save_cell():
    source = [
        "import pickle, pathlib\n",
        "# Update this path if your project lives elsewhere in Drive\n",
        "artifacts_root = pathlib.Path(\"/content/drive/MyDrive/My_NLP_Learning/Project_01_Public_Responce_Analysis\")\n",
        "artifacts_dir = artifacts_root / \"artifacts\"\n",
        "artifacts_dir.mkdir(exist_ok=True)\n",
        "with open(artifacts_dir / \"preprocessing_outputs.pkl\", \"wb\") as f:\n",
        "    pickle.dump({\n",
        "        \"df\": df,\n",
        "        \"one_hot_vectorizer\": one_hot_vectorizer,\n",
        "        \"one_hot_matrix\": one_hot_matrix,\n",
        "        \"bow_vectorizer\": bow_vectorizer,\n",
        "        \"bow_matrix\": bow_matrix,\n",
        "        \"tfidf_vectorizer\": tfidf_vectorizer,\n",
        "        \"tfidf_matrix\": tfidf_matrix,\n",
        "        \"cooccurrence_vectorizer\": cooccurrence_vectorizer,\n",
        "        \"cooccurrence_matrix\": cooccurrence_matrix\n",
        "    }, f)\n",
        "print(\"Saved preprocessing outputs to artifacts/preprocessing_outputs.pkl\")\n",
    ]
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


def build_load_cell():
    source = [
        "import pickle, pathlib\n",
        "# Update this path if your project lives elsewhere in Drive\n",
        "artifacts_root = pathlib.Path(\"/content/drive/MyDrive/My_NLP_Learning/Project_01_Public_Responce_Analysis\")\n",
        "artifacts_path = artifacts_root / \"artifacts/preprocessing_outputs.pkl\"\n",
        "if artifacts_path.exists():\n",
        "    with open(artifacts_path, \"rb\") as f:\n",
        "        artifacts = pickle.load(f)\n",
        "    df = artifacts[\"df\"]\n",
        "    one_hot_vectorizer = artifacts[\"one_hot_vectorizer\"]\n",
        "    one_hot_matrix = artifacts[\"one_hot_matrix\"]\n",
        "    bow_vectorizer = artifacts[\"bow_vectorizer\"]\n",
        "    bow_matrix = artifacts[\"bow_matrix\"]\n",
        "    tfidf_vectorizer = artifacts[\"tfidf_vectorizer\"]\n",
        "    tfidf_matrix = artifacts[\"tfidf_matrix\"]\n",
        "    cooccurrence_vectorizer = artifacts[\"cooccurrence_vectorizer\"]\n",
        "    cooccurrence_matrix = artifacts[\"cooccurrence_matrix\"]\n",
        "    print(\"Loaded preprocessing outputs from artifacts/preprocessing_outputs.pkl\")\n",
        "else:\n",
        "    print(\"Run 01_data_loading_and_preprocessing.ipynb to generate artifacts first.\")\n",
    ]
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


def save_notebook(cells, filename, metadata, nbformat, nbformat_minor):
    notebook = {
        "cells": cells,
        "metadata": metadata,
        "nbformat": nbformat,
        "nbformat_minor": nbformat_minor,
    }
    (BASE_DIR / filename).write_text(
        json.dumps(notebook, indent=2), encoding="utf-8"
    )


def main():
    notebook = load_notebook()
    metadata = notebook.get("metadata", {})
    nbformat = notebook.get("nbformat", 4)
    nbformat_minor = notebook.get("nbformat_minor", 2)
    cells = notebook["cells"]

    # Notebook 01: data loading, preprocessing, analysis
    nb1_cells = clone_cells(cells[0:22])
    nb1_cells.append(
        build_markdown_cell(
            "## Save preprocessing artifacts\n"
            "Run this cell after the preprocessing steps to enable downstream notebooks "
            "to reload the prepared data."
        )
    )
    nb1_cells.append(build_save_cell())
    save_notebook(
        nb1_cells,
        "01_data_loading_and_preprocessing.ipynb",
        metadata,
        nbformat,
        nbformat_minor,
    )

    load_markdown = build_markdown_cell(
        "## Load preprocessing artifacts\n"
        "Loads outputs saved by `01_data_loading_and_preprocessing.ipynb`. "
        "Run that notebook first if this file is missing."
    )
    load_cell = build_load_cell()

    # Notebook 02: embeddings and topic modeling
    nb2_cells = clone_cells([cells[0]]) + [load_markdown, load_cell]
    nb2_cells.extend(clone_cells(cells[22:49]))
    save_notebook(
        nb2_cells,
        "02_embeddings_and_topic_modeling.ipynb",
        metadata,
        nbformat,
        nbformat_minor,
    )

    # Notebook 03: clustering and sentiment scoring
    nb3_cells = clone_cells([cells[0]]) + [clone_cells([load_markdown])[0], clone_cells([load_cell])[0]]
    nb3_cells.extend(clone_cells(cells[49:55]))
    save_notebook(
        nb3_cells,
        "03_clustering_and_sentiment.ipynb",
        metadata,
        nbformat,
        nbformat_minor,
    )


if __name__ == "__main__":
    main()


{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ad7b6cTykwWs",
        "outputId": "d678e479-1516-4210-8a26-9a9d0dcd1f23"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting gradio\n",
            "  Downloading gradio-4.44.1-py3-none-any.whl.metadata (15 kB)\n",
            "Collecting aiofiles<24.0,>=22.0 (from gradio)\n",
            "  Downloading aiofiles-23.2.1-py3-none-any.whl.metadata (9.7 kB)\n",
            "Requirement already satisfied: anyio<5.0,>=3.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (3.7.1)\n",
            "Collecting fastapi<1.0 (from gradio)\n",
            "  Downloading fastapi-0.115.0-py3-none-any.whl.metadata (27 kB)\n",
            "Collecting ffmpy (from gradio)\n",
            "  Downloading ffmpy-0.4.0-py3-none-any.whl.metadata (2.9 kB)\n",
            "Collecting gradio-client==1.3.0 (from gradio)\n",
            "  Downloading gradio_client-1.3.0-py3-none-any.whl.metadata (7.1 kB)\n",
            "Collecting httpx>=0.24.1 (from gradio)\n",
            "  Downloading httpx-0.27.2-py3-none-any.whl.metadata (7.1 kB)\n",
            "Requirement already satisfied: huggingface-hub>=0.19.3 in /usr/local/lib/python3.10/dist-packages (from gradio) (0.24.7)\n",
            "Requirement already satisfied: importlib-resources<7.0,>=1.3 in /usr/local/lib/python3.10/dist-packages (from gradio) (6.4.5)\n",
            "Requirement already satisfied: jinja2<4.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (3.1.4)\n",
            "Requirement already satisfied: markupsafe~=2.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (2.1.5)\n",
            "Requirement already satisfied: matplotlib~=3.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (3.7.1)\n",
            "Requirement already satisfied: numpy<3.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (1.26.4)\n",
            "Collecting orjson~=3.0 (from gradio)\n",
            "  Downloading orjson-3.10.7-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (50 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m50.4/50.4 kB\u001b[0m \u001b[31m3.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from gradio) (24.1)\n",
            "Requirement already satisfied: pandas<3.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (2.2.2)\n",
            "Requirement already satisfied: pillow<11.0,>=8.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (10.4.0)\n",
            "Requirement already satisfied: pydantic>=2.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (2.9.2)\n",
            "Collecting pydub (from gradio)\n",
            "  Downloading pydub-0.25.1-py2.py3-none-any.whl.metadata (1.4 kB)\n",
            "Collecting python-multipart>=0.0.9 (from gradio)\n",
            "  Downloading python_multipart-0.0.12-py3-none-any.whl.metadata (1.9 kB)\n",
            "Requirement already satisfied: pyyaml<7.0,>=5.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (6.0.2)\n",
            "Collecting ruff>=0.2.2 (from gradio)\n",
            "  Downloading ruff-0.6.9-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (25 kB)\n",
            "Collecting semantic-version~=2.0 (from gradio)\n",
            "  Downloading semantic_version-2.10.0-py2.py3-none-any.whl.metadata (9.7 kB)\n",
            "Collecting tomlkit==0.12.0 (from gradio)\n",
            "  Downloading tomlkit-0.12.0-py3-none-any.whl.metadata (2.7 kB)\n",
            "Requirement already satisfied: typer<1.0,>=0.12 in /usr/local/lib/python3.10/dist-packages (from gradio) (0.12.5)\n",
            "Requirement already satisfied: typing-extensions~=4.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (4.12.2)\n",
            "Requirement already satisfied: urllib3~=2.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (2.2.3)\n",
            "Collecting uvicorn>=0.14.0 (from gradio)\n",
            "  Downloading uvicorn-0.31.0-py3-none-any.whl.metadata (6.6 kB)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from gradio-client==1.3.0->gradio) (2024.6.1)\n",
            "Collecting websockets<13.0,>=10.0 (from gradio-client==1.3.0->gradio)\n",
            "  Downloading websockets-12.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)\n",
            "Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.10/dist-packages (from anyio<5.0,>=3.0->gradio) (3.10)\n",
            "Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.10/dist-packages (from anyio<5.0,>=3.0->gradio) (1.3.1)\n",
            "Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<5.0,>=3.0->gradio) (1.2.2)\n",
            "Collecting starlette<0.39.0,>=0.37.2 (from fastapi<1.0->gradio)\n",
            "  Downloading starlette-0.38.6-py3-none-any.whl.metadata (6.0 kB)\n",
            "Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx>=0.24.1->gradio) (2024.8.30)\n",
            "Collecting httpcore==1.* (from httpx>=0.24.1->gradio)\n",
            "  Downloading httpcore-1.0.6-py3-none-any.whl.metadata (21 kB)\n",
            "Collecting h11<0.15,>=0.13 (from httpcore==1.*->httpx>=0.24.1->gradio)\n",
            "  Downloading h11-0.14.0-py3-none-any.whl.metadata (8.2 kB)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.19.3->gradio) (3.16.1)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.19.3->gradio) (2.32.3)\n",
            "Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.19.3->gradio) (4.66.5)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib~=3.0->gradio) (1.3.0)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib~=3.0->gradio) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib~=3.0->gradio) (4.54.1)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib~=3.0->gradio) (1.4.7)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib~=3.0->gradio) (3.1.4)\n",
            "Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib~=3.0->gradio) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3.0,>=1.0->gradio) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas<3.0,>=1.0->gradio) (2024.2)\n",
            "Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.10/dist-packages (from pydantic>=2.0->gradio) (0.7.0)\n",
            "Requirement already satisfied: pydantic-core==2.23.4 in /usr/local/lib/python3.10/dist-packages (from pydantic>=2.0->gradio) (2.23.4)\n",
            "Requirement already satisfied: click>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0,>=0.12->gradio) (8.1.7)\n",
            "Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0,>=0.12->gradio) (1.5.4)\n",
            "Requirement already satisfied: rich>=10.11.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0,>=0.12->gradio) (13.8.1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib~=3.0->gradio) (1.16.0)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (2.18.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub>=0.19.3->gradio) (3.3.2)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0,>=0.12->gradio) (0.1.2)\n",
            "Downloading gradio-4.44.1-py3-none-any.whl (18.1 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m18.1/18.1 MB\u001b[0m \u001b[31m58.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading gradio_client-1.3.0-py3-none-any.whl (318 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m318.7/318.7 kB\u001b[0m \u001b[31m19.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading tomlkit-0.12.0-py3-none-any.whl (37 kB)\n",
            "Downloading aiofiles-23.2.1-py3-none-any.whl (15 kB)\n",
            "Downloading fastapi-0.115.0-py3-none-any.whl (94 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m94.6/94.6 kB\u001b[0m \u001b[31m5.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading httpx-0.27.2-py3-none-any.whl (76 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m76.4/76.4 kB\u001b[0m \u001b[31m5.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading httpcore-1.0.6-py3-none-any.whl (78 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m78.0/78.0 kB\u001b[0m \u001b[31m5.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading orjson-3.10.7-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (141 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m141.9/141.9 kB\u001b[0m \u001b[31m10.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading python_multipart-0.0.12-py3-none-any.whl (23 kB)\n",
            "Downloading ruff-0.6.9-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (10.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m10.9/10.9 MB\u001b[0m \u001b[31m74.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading semantic_version-2.10.0-py2.py3-none-any.whl (15 kB)\n",
            "Downloading uvicorn-0.31.0-py3-none-any.whl (63 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m63.7/63.7 kB\u001b[0m \u001b[31m4.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading ffmpy-0.4.0-py3-none-any.whl (5.8 kB)\n",
            "Downloading pydub-0.25.1-py2.py3-none-any.whl (32 kB)\n",
            "Downloading h11-0.14.0-py3-none-any.whl (58 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m58.3/58.3 kB\u001b[0m \u001b[31m4.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading starlette-0.38.6-py3-none-any.whl (71 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m71.5/71.5 kB\u001b[0m \u001b[31m5.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading websockets-12.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (130 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m130.2/130.2 kB\u001b[0m \u001b[31m9.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: pydub, websockets, tomlkit, semantic-version, ruff, python-multipart, orjson, h11, ffmpy, aiofiles, uvicorn, starlette, httpcore, httpx, fastapi, gradio-client, gradio\n",
            "Successfully installed aiofiles-23.2.1 fastapi-0.115.0 ffmpy-0.4.0 gradio-4.44.1 gradio-client-1.3.0 h11-0.14.0 httpcore-1.0.6 httpx-0.27.2 orjson-3.10.7 pydub-0.25.1 python-multipart-0.0.12 ruff-0.6.9 semantic-version-2.10.0 starlette-0.38.6 tomlkit-0.12.0 uvicorn-0.31.0 websockets-12.0\n"
          ]
        }
      ],
      "source": [
        "!pip install gradio"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data = !kaggle datasets download -d shayanfazeli/heartbeat"
      ],
      "metadata": {
        "id": "W3hAe2kWla50"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import gradio as gr\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import zipfile\n",
        "import joblib"
      ],
      "metadata": {
        "id": "JjZpwLcSlUUF"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "with zipfile.ZipFile('heartbeat.zip', 'r') as zip_ref:\n",
        "    zip_ref.extractall('data')"
      ],
      "metadata": {
        "id": "zdHLdFdYlbo9"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_data = pd.read_csv(r\"/content/data/mitbih_test.csv\")\n",
        "\n",
        "test_row = test_data.iloc[0, :-1]  # Get the first row of test data without the 'Target' column\n",
        "\n",
        "np.savetxt(\"/content/test_ecg.txt\", test_row.values, delimiter=',')"
      ],
      "metadata": {
        "id": "S8LYjJffkxe9"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def predict_ecg_class_from_file(ecg_file):\n",
        "    try:\n",
        "        ecg_data = np.loadtxt(ecg_file.name)\n",
        "\n",
        "        if ecg_data.shape[0] == 187:\n",
        "            ecg_data = np.append(ecg_data, [0])\n",
        "\n",
        "        if ecg_data.shape[0] != 188:\n",
        "            return \"Error: The file should contain exactly 188 features.\"\n",
        "\n",
        "        ecg_data = ecg_data.reshape(1, -1)\n",
        "\n",
        "        pca = joblib.load('/content/pca_transform.pkl')\n",
        "        ecg_data_pca = pca.transform(ecg_data)\n",
        "\n",
        "        classifier = joblib.load('/content/Binary classifier_random_forest_model.pkl')\n",
        "        binary_prediction = classifier.predict(ecg_data_pca)\n",
        "\n",
        "        class_mapping = {\n",
        "            0: \"Normal 'N'\",\n",
        "            1: \"Supra-ventricular premature 'S'\",\n",
        "            2: \"Ventricular escape 'V'\",\n",
        "            3: \"Fusion of ventricular and normal 'F'\",\n",
        "            4: \"Fusion of paced and normal 'Q'\"\n",
        "        }\n",
        "\n",
        "        if binary_prediction[0] != 0:\n",
        "            sub_classifier = joblib.load('/content/sub-classifier_random_forest_model.pkl')\n",
        "            sub_prediction = sub_classifier.predict(ecg_data_pca)\n",
        "\n",
        "            return f\"Predicted Class: {class_mapping[sub_prediction[0]]}\"\n",
        "        else:\n",
        "\n",
        "            return f\"Predicted Class: {class_mapping[binary_prediction[0]]}\"\n",
        "\n",
        "    except Exception as e:\n",
        "        return f\"Error processing file: {e}\"\n"
      ],
      "metadata": {
        "id": "AbAphN58lLli"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "interface = gr.Interface(\n",
        "    fn=predict_ecg_class_from_file,\n",
        "    inputs=gr.File(label=\"Upload ECG .txt File\"),\n",
        "    outputs=\"text\",\n",
        "    title=\"ECG Signal Classifier\",\n",
        "    description=\"Upload a .txt file containing the 188 features of the ECG signal to predict its class.\"\n",
        ")\n",
        "\n",
        "interface.launch(debug=True)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 660
        },
        "id": "zHU36KOvlSZ2",
        "outputId": "d0cec985-ed8d-462d-9d9d-4a5006089dab"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Setting queue=True in a Colab notebook requires sharing enabled. Setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).\n",
            "\n",
            "Colab notebook detected. This cell will run indefinitely so that you can see errors and logs. To turn off, set debug=False in launch().\n",
            "Running on public URL: https://18cdbf335368c7a1e9.gradio.live\n",
            "\n",
            "This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<div><iframe src=\"https://18cdbf335368c7a1e9.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but PCA was fitted with feature names\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    }
  ]
}

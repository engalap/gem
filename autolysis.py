# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "requests",
#   "openai",
#   "matplotlib",
#   "seaborn",
#   "python-dotenv",
# ]
# ///

import json
import pandas as pd
import requests
import sys
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from dotenv import load_dotenv
load_dotenv("./.secret")
import base64
from io import BytesIO
from PIL import Image

openai.api_key = os.getenv("OPENAI_API_KEY")
Api_key = os.getenv("OPENAI_API_KEY")
if Api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
# e = requests.get("https://aiproxy.sanand.workers.dev/openai/v1/models", headers= {
#     "Authorization": f"Bearer {Api_key}"
# }).json()


def analyze_csv(file_path):

    df = pd.read_csv(file_path)
    column_info = df.dtypes

    headers = {
        "Authorization": f"Bearer {Api_key}",
        "Content-Type": "application/json"
    }
    function_call_multiple = {
        "name": "analyze_csv",
        "description": "Analyze a CSV file and suggest appropriate analysis techniques",
        "parameters": {
            "type": "object",
            "properties": {
                "file_info": {
                    "type": "object",
                    "properties": {
                        "file_name": {"type": "string"},
                        "column_info": {"type": "object"}
                    }
                },
                "suggested_analyses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "technique": {"type": "string"},
                            "columns": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                # "Distance":{
                #     "type": "array",
                #     "items": {
                #         "type": "object",
                #         "properties": {
                #             "technique": {"type": "string"},
                #             "columns": {"type": "array", "items": {"type": "string"}}
                #         }
                #     }
                # }

            }
        }
    }
    analysis_names = ["Descriptive Statistics", "Correlation Analysis", "Distribution Analysis", "Comparative Analysis", "Time Series Analysis", "Top-N Analysis", "Categorical Analysis","Geographic Analysis"]

    data = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": f"Analyze this CSV file with columns {list(df.columns)} and datatypes {column_info.to_dict()}. Suggest appropriate analysis techniques from {analysis_names} and specify which columns to use for each technique."
        }],
        "functions": [function_call_multiple],
        "function_call": {"name": "analyze_csv"}
    }

    response = requests.post(openai.api_base, headers=headers, json=data)

    response_json = response.json()
    print(response_json) #Print the response to inspect

    # Error handling
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
    # suggested_analyses = json.loads(response.choices[0].message.function_call.arguments)["suggested_analyses"]

    return response_json
    



# Load the DataFrame
def plots(folder_name, df,json_data):
    analyses_data = json.loads(json_data['choices'][0]['message']['function_call']['arguments'])['suggested_analyses']
    charts = []
    for analysis in analyses_data:
        technique = analysis['technique']
        columns = analysis['columns']
        
        print(f"Performing {technique} on columns: {columns}")

        if technique == "Correlation Analysis":
            correlation_matrix = df[columns].corr()
            print(correlation_matrix)
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
            plt.savefig(f"{folder_name}/Correlation Matrix.png")
            plt.title('Correlation Matrix')
            charts.append(f'{folder_name}/Correlation Matrix.png')
            
            plt.show()
        elif technique == "Distribution Analysis":
            for col in columns:
                plt.figure(figsize=(8, 6))
                sns.histplot(df[col], kde=True)
                plt.title(f'Distribution_of_{col}')
                plt.savefig(f"{folder_name}/Distribution_of_{col}.png")
                charts.append(f"{folder_name}/Distribution_of_{col}.png")
                plt.show()
        elif technique == "Comparative Analysis":
            # Example: Boxplot comparison
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=columns[1], y=columns[0], data=df)  # Assuming 'authors' and 'average_rating'
            plt.title(f'Comparative Analysis of {columns[0]} by {columns[1]}')
            plt.savefig(f"{folder_name}/Comparative_analysis{columns[0]}_{columns[1]}.png")
            charts.append(f"{folder_name}/Comparative_analysis{columns[0]}_{columns[1]}.png")
            plt.show()
        elif technique == "Time Series Analysis":
            # Example: Line plot over time (assuming 'original_publication_year')
            plt.figure(figsize=(10, 6))
            plt.plot(df.groupby(columns[0])[columns[1]].mean().index, df.groupby(columns[0])[columns[1]].mean().values)
            plt.xlabel(columns[0])
            plt.ylabel(columns[1])
            plt.title('Time Series Analysis')
            plt.savefig(f"{folder_name}/Time_series.png")
            charts.append(f"{folder_name}/Time_series.png")
            plt.show()

        elif technique == "Top-N Analysis":
            top_n = 10 #Example: Top 10
            print(df.sort_values(by=columns[1], ascending=False).head(top_n)[columns])

        elif technique == "Categorical Analysis":
            for col in columns:
                print(df[col].value_counts())
                plt.figure(figsize=(8, 6))
                df[col].value_counts().plot(kind='bar')
                plt.title(f'Categorical Analysis of {col}')
                plt.savefig(f"{folder_name}/Categorical_analysis_{col}.png")
                charts.append(f"{folder_name}/Categorical_analysis_{col}.png")
                plt.show()
        return charts



def resize_and_encode_image(image_path, size=(32, 32)):
    try:
        img = Image.open(image_path)
        img = img.resize(size)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def image_analysis(charts):
    image_messages = []
    for image in charts:
          image_data = resize_and_encode_image(image)
          if image_data:
              image_messages.append({"type": "image_url", "image_url": {"url" : f"data:image/png;base64,{image_data}"},})
    return image_messages




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <csv_filename>")
        sys.exit(1)  # Exit with an error code

    csv_filename = sys.argv[1]
    print(csv_filename)
    csv_filename = os.path.basename(csv_filename)

    folder_name = os.path.splitext(csv_filename)[0]

    e = analyze_csv(csv_filename)
    c = pd.read_csv(csv_filename)
    charts = plots(folder_name,c,e)
    ef = image_analysis(charts)
    messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "What insight is learned about the data through this analysis shown in the images"
            },
            ],
        },
    ]   
    messages[0]['content'].extend(ef)
    openai.api_key = Api_key
    openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"


    headers = {
    "Authorization": f"Bearer {Api_key}",
    "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 300,
        }

    response = requests.post(openai.api_base, headers=headers, json=data)
    ww = response.json()

    try:
        df = pd.read_csv(csv_filename)
        # Now you can work with the DataFrame 'df'
    except FileNotFoundError:
        print(f"Error: File not found: {csv_filename}")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: Could not parse CSV file: {csv_filename}")
        sys.exit(1)
    
    
    print(folder_name)
    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    readme_path = os.path.join(folder_name, "README.md")

    with open(readme_path, 'w') as f:
        f.write("# Automated Analysis Report\n\n")
        f.write(f"## File Analyzed: {csv_filename}\n\n")
        f.write("### Summary of Analysis\n")
        f.write(ww['choices'][0]['message']['content'] + "\n\n")
    
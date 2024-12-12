import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDQyNjNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.ECv6hw2LCl3m7Ahp5hyOFkoZyV4wgucZ6Zz6nA8J8Yo"

def analyze_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    
    
    # Get column names and types
    column_info = df.dtypes
    
    # Prepare data for ChatGPT-4
    file_info = {
        "file_name": file_path,
        "column_info": column_info.to_dict()
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
                }
            }
        }
    }

    # Send data to ChatGPT-4 with structured function calling

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Analyze this CSV file with columns {list(df.columns)} and datatypes {column_info.to_dict()}. Suggest appropriate analysis techniques and specify which columns to use for each technique."
        }],
        functions=[function_call_multiple],
        function_call={"name": "analyze_csv"}
    )
    print(response)

    # Extract the suggested analysis from the response
    suggested_analyses = json.loads(response.choices[0].message.function_call.arguments)["suggested_analyses"]
    
    # Run the suggested analysis based on the structured response
    analysis_results = {}
    charts = []
    
    for analysis in suggested_analyses:
        technique = analysis["technique"]
        columns = analysis["columns"]
        
        if technique == "summary_statistics":
            analysis_results["summary_stats"] = df[columns].describe()
        
        elif technique == "correlation_matrix":
            analysis_results["correlation_matrix"] = df[columns].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[columns].corr(), annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.savefig('correlation_matrix.png')
            charts.append('correlation_matrix.png')
    
        if "missing values" in suggested_analyses:
            analysis_results["missing_values"] = df.isnull().sum()
    
        if "correlation matrix" in suggested_analyses:
            analysis_results["correlation_matrix"] = df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.savefig('correlation_matrix.png')
            charts.append('correlation_matrix.png')
    
        if "outliers" in suggested_analyses:
            from scipy.stats import zscore
            z_scores = df.apply(zscore)
            analysis_results["outliers"] = (z_scores.abs() > 3).sum()
    
        if "clustering" in suggested_analyses:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(df.select_dtypes(include=[float, int]).dropna())
            analysis_results["clusters"] = kmeans.labels_
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=df.select_dtypes(include=[float, int]).columns[0], 
                            y=df.select_dtypes(include=[float, int]).columns[1], 
                            hue=kmeans.labels_, palette='viridis')
            plt.title('KMeans Clustering')
            plt.savefig('kmeans_clustering.png')
            charts.append('kmeans_clustering.png')
    
        if "hierarchy detection" in suggested_analyses:
            from sklearn.cluster import AgglomerativeClustering
            agglomerative = AgglomerativeClustering(n_clusters=3)
            hierarchy_labels = agglomerative.fit_predict(df.select_dtypes(include=[float, int]).dropna())
            analysis_results["hierarchy"] = hierarchy_labels
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=df.select_dtypes(include=[float, int]).columns[0], 
                            y=df.select_dtypes(include=[float, int]).columns[1], 
                            hue=hierarchy_labels, palette='viridis')
            plt.title('Hierarchical Clustering')
            plt.savefig('hierarchical_clustering.png')
            charts.append('hierarchical_clustering.png')
    
    # Write results to README.md
    with open('README.md', 'w') as f:
        f.write("# Automated Analysis Report\n\n")
        f.write(f"## File Analyzed: {file_path}\n\n")
        f.write("### Summary of Analysis\n")
        f.write(suggested_analyses + "\n\n")
        for key, value in analysis_results.items():
            f.write(f"### {key.replace('_', ' ').title()}\n")
            f.write(value.to_string() + "\n\n")
        for chart in charts:
            f.write(f"![{chart}]({chart})\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <csv_filename>")
        sys.exit(1)  # Exit with an error code

    csv_filename = sys.argv[1]
    e = analyze_csv(csv_filename)

    try:
        df = pd.read_csv(csv_filename)
        # Now you can work with the DataFrame 'df'
    except FileNotFoundError:
        print(f"Error: File not found: {csv_filename}")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: Could not parse CSV file: {csv_filename}")
        sys.exit(1)


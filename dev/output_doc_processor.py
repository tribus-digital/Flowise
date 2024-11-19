import os
import json

# Set the input and output directories
input_directory = "outputs/"
output_directory = "outputs/processed/"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Loop through all JSON files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".json"):
        json_path = os.path.join(input_directory, filename)

        # Read the JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Derive the title from the filename (without extension)
        title = os.path.splitext(filename)[0]

        # Start constructing the Markdown content
        md_content = f"### {title}\n\n"

        for item in data:
            question = item.get("question", "No question provided.")
            response = item.get("response", "No response provided.")
            contexts = item.get("contexts", [])

            # State the question
            md_content += f"# {question}\n\n"

            # Add the response
            md_content += response + "\n\n---\n\n"

            md_content += "<details>\n"  # Start a collapsible section (most modern Markdown parsers support this)
            md_content += f"  <summary>RAG Context</summary>\n\n"
            # Context Information
            md_content += f"**Total Context Items:** {len(contexts)}\n\n"

            for context in contexts:
                context_title = context.get("title", "No title")
                context_url = context.get("url", "No URL")
                context_date = context.get("date", "No date")
                context_score = context.get("relevance", "No score")

                md_content += f"- **{context_title}**\n"
                md_content += f"  - **URL:** {context_url}\n"
                md_content += f"  - **Last-Modified:** {context_date}\n"
                md_content += f"  - **Score:** {context_score}\n\n"
            md_content += "</details>\n\n"
            md_content += "---\n\n---\n\n---\n\n"

        # Define the output file paths
        base_filename = os.path.splitext(filename)[0]
        md_output_path = os.path.join(output_directory, f"{base_filename}.md")

        # Write the Markdown content to a .md file
        with open(md_output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"Processed {filename}")

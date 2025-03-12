# Novel Summarizer Notebook

## Overview
The **Novel Summarizer Notebook** is designed to process and summarize chapters of *Gone with the Wind* from a PDF file. It extracts chapter-wise content, generates one-sentence summaries using OpenAI's API, identifies curse words, and saves the results in a structured CSV file.

## Features
- **PDF Parsing**: Extracts chapter-wise text from *Gone with the Wind*.
- **Chapter Summarization**: Uses OpenAI's GPT-3.5-turbo to generate concise one-sentence summaries for each chapter.
- **Curse Word Detection**: Identifies offensive language within the text using AI-powered analysis.
- **Data Export**: Saves the extracted summaries and curse word counts to a CSV file.
- **Visualization**: Displays the processed data in a tabular format.

## Installation
### Prerequisites
Ensure you have Python installed along with the necessary dependencies.

### Required Libraries
Install dependencies using pip:
```bash
pip install pymupdf openai python-dotenv pandas
```

## Setup
### Obtain OpenAI API Key
1. Sign up at [OpenAI](https://openai.com/) and generate an API key.
2. Create a `.env` file in the working directory and add your key:
   ```ini
   OPENAI_API_KEY=your_api_key_here
   ```

### Prepare the PDF File
- Place *Gone with the Wind* in the working directory with the name `gonewiththewind.pdf`.

## Usage
### Initialize the Summarizer
```python
summarizer = NovelSummarizer("gonewiththewind.pdf")
```

### Extract Chapters
```python
summarizer.extract_chapters()
```

### Summarize All Chapters
```python
results = summarizer.summarize_all_chapters()
```

### Save Summaries to CSV
```python
summary_df = summarizer.save_summaries_csv(results, "gone_with_the_wind_summaries.csv")
display(summary_df)
```

## Output
- A CSV file `gone_with_the_wind_summaries.csv` containing:
  - Chapter Number
  - One-sentence summary
  - Count of curse words per chapter
- A DataFrame displaying the summarized results within the notebook.

## Error Handling
- If the PDF fails to load, an error message will be displayed.
- If OpenAI's API call fails, a fallback mechanism provides default curse words.
- Summarization errors are handled gracefully with informative error messages.

## License
This project is for educational purposes. Modify and distribute as needed.

---
For any issues, feel free to contribute or report them in the repository!


# %% 
# Novel Summarizer Notebook
# This notebook summarizes "Gone with the Wind" chapter by chapter

# Import necessary libraries
import os
import re
import csv
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import pandas as pd
from IPython.display import display, Markdown

# Load environment variables (for OpenAI API key)
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# %% [markdown]
# ## Novel Summarizer Class

# %%
class NovelSummarizer:
    def __init__(self, pdf_path: str):
        """
        Initialize the novel summarizer with the path to the PDF.
        
        Args:
            pdf_path: Path to the PDF file to summarize
        """
        self.pdf_path = pdf_path
        self.doc = None
        self.chapters = []
        self.curse_words = set()  # Will be populated from the text
        
    def load_document(self) -> None:
        """Load the PDF document."""
        try:
            self.doc = fitz.open(self.pdf_path)
            display(Markdown(f"Successfully loaded '{self.pdf_path}' with {len(self.doc)} pages."))
        except Exception as e:
            display(Markdown(f"**Error loading PDF:** {e}"))
            raise
    
    def extract_chapters(self) -> List[Dict]:
        """
        Extract chapters from the PDF document.
        
        Returns:
            List of dictionaries containing chapter number, title, and text
        """
        if not self.doc:
            self.load_document()
        
        chapters = []
        chapter_pattern = re.compile(r'CHAPTER\s+([IVXLCDM]+|[0-9]+)', re.IGNORECASE)
        
        current_chapter = None
        current_text = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            
            # Check if this page starts a new chapter
            match = chapter_pattern.search(text)
            
            if match and (page_num == 0 or chapter_pattern.search(text).start() < 100):
                # If we were collecting a previous chapter, save it
                if current_chapter:
                    chapters.append({
                        "number": current_chapter,
                        "text": "\n".join(current_text)
                    })
                
                # Start a new chapter
                current_chapter = match.group(1)
                current_text = [text]
            elif current_chapter:
                current_text.append(text)
        
        # Add the last chapter
        if current_chapter:
            chapters.append({
                "number": current_chapter,
                "text": "\n".join(current_text)
            })
        
        self.chapters = chapters
        display(Markdown(f"Extracted {len(chapters)} chapters from the document."))
        
        # Extract curse words from the novel
        self.extract_curse_words()
        
        return chapters
    
    def extract_curse_words(self) -> None:
        """
        Extract curse words from the novel using OpenAI's API.
        This populates the self.curse_words set.
        """
        # Combine a sample of text from multiple chapters
        sample_text = ""
        for chapter in self.chapters[:5]:  # Use first 5 chapters as a sample
            sample_text += chapter["text"][:2000] + "\n\n"  # Take first 2000 chars from each
        
        try:
            response = openai.OpenAI().chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a literary analyst helping to identify curse words and offensive language in historical novels."},
                    {"role": "user", "content": f"Based on this sample from 'Gone with the Wind', identify all curse words, profanity, and offensive language used in the novel. Return ONLY a comma-separated list of the actual words, with no explanations or commentary:\n\n{sample_text}"}
                ],
                max_tokens=200,
                temperature=0.5
            )
            
            curse_words_text = response.choices[0].message.content.strip()
            # Split by commas and clean up each word
            curse_words = [word.strip().lower() for word in curse_words_text.split(',')]
            # Filter out empty strings
            self.curse_words = set(word for word in curse_words if word)
            
            display(Markdown(f"Identified {len(self.curse_words)} curse words in the novel: {', '.join(self.curse_words)}"))
            
        except Exception as e:
            display(Markdown(f"**Error extracting curse words:** {e}"))
            # Fallback to some common words that might be in the novel
            self.curse_words = {"damn", "hell", "devil", "god's nightgown"}
    
    def count_curse_words(self, text: str) -> int:
        """
        Count the number of curse words in a text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Number of curse words found
        """
        if not self.curse_words:
            self.extract_curse_words()
            
        text = text.lower()
        count = 0
        
        for word in self.curse_words:
            # Use word boundary to match whole words only
            pattern = r'\b' + re.escape(word) + r'\b'
            count += len(re.findall(pattern, text))
            
        return count
    
    def summarize_chapter_one_sentence(self, chapter_text: str) -> str:
        """
        Summarize a single chapter in one sentence using OpenAI's API.
        
        Args:
            chapter_text: The text of the chapter to summarize
            
        Returns:
            A one-sentence summary of the chapter
        """
        try:
            # Truncate very long chapters to avoid token limits
            if len(chapter_text) > 12000:
                chapter_text = chapter_text[:12000] + "..."
            
            response = openai.OpenAI().chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a literary assistant that creates concise one-sentence chapter summaries."},
                    {"role": "user", "content": f"Please summarize the following chapter from 'Gone with the Wind' in exactly ONE SENTENCE. Focus on the most important plot development or character moment:\n\n{chapter_text}"}
                ],
                max_tokens=100,
                temperature=0.7
            )
            summary = response.choices[0].message.content.strip()
            
            # Ensure it's just one sentence by taking the first sentence if multiple are returned
            if '.' in summary:
                first_sentence = summary.split('.')[0].strip() + '.'
                return first_sentence
            return summary
            
        except Exception as e:
            display(Markdown(f"**Error summarizing chapter:** {e}"))
            return "Error generating summary."
    
    def summarize_all_chapters(self) -> Dict[str, Dict]:
        """
        Summarize all chapters in the document with one-sentence summaries and count curse words.
        
        Returns:
            Dictionary mapping chapter numbers to dictionaries containing summary and curse word count
        """
        if not self.chapters:
            self.extract_chapters()
        
        results = {}
        
        for i, chapter in enumerate(self.chapters):
            display(Markdown(f"Summarizing Chapter {chapter['number']} ({i+1}/{len(self.chapters)})..."))
            summary = self.summarize_chapter_one_sentence(chapter['text'])
            curse_count = self.count_curse_words(chapter['text'])
            results[chapter['number']] = {
                "summary": summary,
                "curse_count": curse_count
            }
        
        return results
    
    def save_summaries_csv(self, results: Dict[str, Dict], output_path: str) -> None:
        """
        Save the chapter summaries and curse word counts to a CSV file.
        
        Args:
            results: Dictionary mapping chapter numbers to dictionaries with summary and curse count
            output_path: Path to save the CSV file
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Chapter', 'Summary', 'Curse Word Count'])
            
            # Write chapter summaries in order
            for chapter_num in sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else x):
                writer.writerow([
                    f"Chapter {chapter_num}", 
                    results[chapter_num]["summary"], 
                    results[chapter_num]["curse_count"]
                ])
        
        display(Markdown(f"Summaries saved to {output_path}"))
        
        # Also display the summaries as a DataFrame
        df = pd.DataFrame(
            [[f"Chapter {k}", v["summary"], v["curse_count"]] for k, v in results.items()], 
            columns=['Chapter', 'Summary', 'Curse Word Count']
        )
        return df

# %% [markdown]
# ## Run the Summarizer

# %%
# Initialize the summarizer
summarizer = NovelSummarizer("gonewiththewind.pdf")

# %% [markdown]
# ### Extract and Summarize Chapters

# %%
# Extract chapters
summarizer.extract_chapters()

# %%
# Summarize all chapters and count curse words
results = summarizer.summarize_all_chapters()

# %% [markdown]
# ### Save and Display Results

# %%
# Save to CSV and display as DataFrame
summary_df = summarizer.save_summaries_csv(results, "gone_with_the_wind_summaries.csv")
display(summary_df)




# %%

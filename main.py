#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime
import fitz  # PyMuPDF
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import torch

class PDFAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the PDF analyzer with a sentence transformer model"""
        self.model = SentenceTransformer(model_name)
        # For better performance on CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract all text from a PDF file with page numbers"""
        text_by_page = {}
        try:
            # Try with PyMuPDF first
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                text_by_page[page_num] = doc[page_num].get_text()
        except Exception as e:
            # Fallback to pdfplumber
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text_by_page[page_num] = page.extract_text() or ""
            except Exception as e2:
                print(f"Error extracting text from {pdf_path}: {str(e2)}")
        
        return text_by_page
    
    def find_title(self, text_by_page):
        """Try to find document title from the first page"""
        if 0 not in text_by_page:
            return "Document Overview"
            
        first_page = text_by_page[0]
        lines = first_page.split('\n')
        
        # Try to identify title - usually one of the first few lines
        # that is centered, bold, or in a larger font
        if lines:
            # Simplistic approach: take first non-empty line that's not too long
            for line in lines:
                cleaned = line.strip()
                if cleaned and len(cleaned) < 100:
                    return cleaned
                    
        return "Document Overview"
    
    def extract_sections(self, pdf_path):
        """
        Extract sections from a PDF based on natural text boundaries.
        Instead of trying to identify actual headings, we'll divide text into
        logical chunks that can be analyzed for relevance.
        """
        # Get text by page
        text_by_page = self.extract_text_from_pdf(pdf_path)
        
        # Find document title
        title = self.find_title(text_by_page)
        
        # For this challenge, we'll use a simpler approach - divide each page into sections
        sections = []
        
        # Add title section
        title_section = {
            "title": title,
            "level": "title",
            "page": 0,
            "content": ""
        }
        sections.append(title_section)
        
        # Process each page
        for page_num, page_text in text_by_page.items():
            if not page_text.strip():
                continue
                
            # Break page into paragraphs or chunks
            paragraphs = [p for p in page_text.split('\n\n') if p.strip()]
            
            # Process each paragraph
            for para_idx, paragraph in enumerate(paragraphs):
                # If paragraph is too short, skip or combine
                if len(paragraph) < 50:
                    continue
                
                # Extract potential section title (first sentence or line)
                lines = paragraph.strip().split('\n')
                potential_title = lines[0] if lines else ""
                
                # Limit title length
                if len(potential_title) > 100:
                    words = potential_title.split()
                    potential_title = " ".join(words[:10]) + "..."
                
                # Create section
                section = {
                    "title": potential_title,
                    "level": "h1" if para_idx == 0 else "h2",
                    "page": page_num,
                    "content": paragraph
                }
                
                sections.append(section)
        
        return sections
    
    def extract_sections_with_content(self, pdf_path):
        """
        Extract sections with their content using our simplified approach
        This function works for any domain, not specific to travel or any other topic.
        """
        # Use our new extract_sections method that already includes content
        return self.extract_sections(pdf_path)
    
    def _extract_content_between_headings(self, doc, start_page, start_heading, end_page, end_heading):
        """Helper method to extract content between two headings"""
        content = ""
        
        try:
            # Extract content from start page
            page = doc[start_page]
            text = page.get_text()
            
            # Find starting position after heading
            start_idx = text.find(start_heading)
            if start_idx >= 0:
                start_idx += len(start_heading)
                
                # If both headings are on the same page
                if start_page == end_page:
                    end_idx = text.find(end_heading, start_idx)
                    if end_idx >= 0:
                        content = text[start_idx:end_idx].strip()
                else:
                    # Add content from first page after heading
                    content = text[start_idx:].strip()
                    
                    # Add content from intermediate pages
                    for page_num in range(start_page + 1, end_page):
                        page = doc[page_num]
                        content += "\n" + page.get_text()
                    
                    # Add content from last page up to next heading
                    page = doc[end_page]
                    text = page.get_text()
                    end_idx = text.find(end_heading)
                    
                    if end_idx >= 0:
                        content += "\n" + text[:end_idx].strip()
        except Exception as e:
            # Fallback to pdfplumber if there's an error
            try:
                with pdfplumber.open(doc.name) as pdf:
                    if start_page == end_page:
                        # Both headings on same page
                        page = pdf.pages[start_page]
                        text = page.extract_text()
                        start_idx = text.find(start_heading) + len(start_heading)
                        end_idx = text.find(end_heading, start_idx)
                        
                        if start_idx >= 0 and end_idx >= 0:
                            content = text[start_idx:end_idx].strip()
                    else:
                        # Headings on different pages
                        # Extract from start page after heading
                        page = pdf.pages[start_page]
                        text = page.extract_text()
                        start_idx = text.find(start_heading) + len(start_heading)
                        
                        if start_idx >= 0:
                            content = text[start_idx:].strip()
                            
                        # Extract from middle pages
                        for page_num in range(start_page + 1, end_page):
                            content += "\n" + pdf.pages[page_num].extract_text()
                            
                        # Extract from end page up to next heading
                        page = pdf.pages[end_page]
                        text = page.extract_text()
                        end_idx = text.find(end_heading)
                        
                        if end_idx >= 0:
                            content += "\n" + text[:end_idx].strip()
            except:
                pass
        
        # Clean up the content
        content = re.sub(r'\s+', ' ', content).strip()
        return content
    
    def _extract_content_after_heading(self, doc, page_num, heading):
        """Helper method to extract content after a heading to the end of the document"""
        content = ""
        
        try:
            # Extract content from heading page
            page = doc[page_num]
            text = page.get_text()
            
            # Find start position after heading
            start_idx = text.find(heading) + len(heading)
            
            if start_idx >= 0:
                # Add content from first page after heading
                content = text[start_idx:].strip()
                
                # Add content from remaining pages
                for p_num in range(page_num + 1, len(doc)):
                    page = doc[p_num]
                    content += "\n" + page.get_text()
        except Exception as e:
            # Fallback to pdfplumber if there's an error
            try:
                with pdfplumber.open(doc.name) as pdf:
                    # Extract from heading page
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    start_idx = text.find(heading) + len(heading)
                    
                    if start_idx >= 0:
                        content = text[start_idx:].strip()
                        
                    # Extract from remaining pages
                    for p_num in range(page_num + 1, len(pdf.pages)):
                        content += "\n" + pdf.pages[p_num].extract_text()
            except:
                pass
        
        # Clean up the content
        content = re.sub(r'\s+', ' ', content).strip()
        return content
    
    def rank_sections_by_relevance(self, sections, persona, job_to_be_done, top_n=5):
        """
        Rank sections by relevance to the persona and job to be done
        Returns the top N most relevant sections
        """
        if not sections:
            return []
        
        # Create queries for different aspects of the search
        main_query = f"{persona} needs to {job_to_be_done}"
        persona_query = f"Information relevant for a {persona}"
        task_query = f"How to {job_to_be_done}"
        
        # Encode queries
        main_embedding = self.model.encode(main_query)
        persona_embedding = self.model.encode(persona_query)
        task_embedding = self.model.encode(task_query)
        
        # Calculate relevance scores for each section
        scores = []
        for section in sections:
            # Skip sections with no content or very short content
            if not section["content"] or len(section["content"]) < 20:
                continue
                
            # Prepare text for encoding - use title and beginning of content
            title = section["title"]
            # Take a sample of content (beginning, middle, and end)
            content_length = len(section["content"])
            
            if content_length <= 1000:
                content_sample = section["content"]
            else:
                begin = section["content"][:600]
                middle_start = max(0, content_length // 2 - 200)
                middle = section["content"][middle_start:middle_start + 400]
                end = section["content"][-200:]
                content_sample = begin + " " + middle + " " + end
            
            text_to_encode = f"{title}: {content_sample}"
            
            # Encode section text
            try:
                section_embedding = self.model.encode(text_to_encode)
                
                # Calculate cosine similarity for different queries
                main_similarity = np.dot(main_embedding, section_embedding) / (np.linalg.norm(main_embedding) * np.linalg.norm(section_embedding))
                persona_similarity = np.dot(persona_embedding, section_embedding) / (np.linalg.norm(persona_embedding) * np.linalg.norm(section_embedding))
                task_similarity = np.dot(task_embedding, section_embedding) / (np.linalg.norm(task_embedding) * np.linalg.norm(section_embedding))
                
                # Weight the similarities (main query has highest weight)
                weighted_similarity = main_similarity * 0.6 + persona_similarity * 0.2 + task_similarity * 0.2
                
                # Add a bonus for sections with relevant keywords in title
                title_lower = title.lower()
                keywords = set(job_to_be_done.lower().split())
                keyword_matches = sum(1 for kw in keywords if kw in title_lower and len(kw) > 3)
                title_bonus = min(0.1, keyword_matches * 0.02)
                
                final_score = weighted_similarity + title_bonus
                scores.append((section, final_score))
            except Exception as e:
                print(f"Error encoding section: {str(e)}")
        
        # Sort by relevance score
        ranked_sections = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Return top N sections or all if fewer than N
        return [section for section, score in ranked_sections[:top_n]]
    
    def extract_subsections(self, sections, persona, job_to_be_done):
        """
        Extract the most relevant text from each section using enhanced techniques
        """
        if not sections:
            return []
            
        # Create more specific queries to target different aspects
        main_query = f"{persona} needs to {job_to_be_done}"
        specific_query = f"Important information for {job_to_be_done}"
        detail_query = f"Details about {job_to_be_done} for {persona}"
        
        # Encode queries
        query_embeddings = {
            "main": self.model.encode(main_query),
            "specific": self.model.encode(specific_query),
            "detail": self.model.encode(detail_query)
        }
        
        subsections = []
        for section in sections:
            # Skip sections with no content
            if not section["content"] or len(section["content"]) < 20:
                continue
                
            # Break content into semantic chunks (paragraphs or sentences)
            # First try to split by paragraphs
            paragraphs = [p for p in section["content"].split('\n\n') if len(p.strip()) > 10]
            
            # If there aren't enough paragraphs, try sentence splitting
            if len(paragraphs) <= 1:
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', section["content"]) if len(s.strip()) > 10]
                
                # Group sentences into meaningful chunks
                chunks = []
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 500:  # Keep chunks under reasonable size
                        current_chunk += " " + sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                paragraphs = chunks if chunks else [section["content"]]
            
            # Encode each paragraph/chunk
            try:
                paragraph_embeddings = self.model.encode(paragraphs)
                
                # Calculate combined similarity for each paragraph/chunk
                similarities = []
                for i, emb in enumerate(paragraph_embeddings):
                    # Calculate similarity for each query type
                    main_sim = np.dot(query_embeddings["main"], emb) / (np.linalg.norm(query_embeddings["main"]) * np.linalg.norm(emb))
                    specific_sim = np.dot(query_embeddings["specific"], emb) / (np.linalg.norm(query_embeddings["specific"]) * np.linalg.norm(emb))
                    detail_sim = np.dot(query_embeddings["detail"], emb) / (np.linalg.norm(query_embeddings["detail"]) * np.linalg.norm(emb))
                    
                    # Calculate weighted average
                    combined_sim = main_sim * 0.5 + specific_sim * 0.3 + detail_sim * 0.2
                    
                    # Add content length as a factor (favor reasonably sized content)
                    content_length_factor = min(1.0, len(paragraphs[i]) / 500)
                    
                    # Final score
                    final_score = combined_sim * content_length_factor
                    similarities.append((i, final_score))
                
                # Get the most relevant paragraph/chunk
                most_relevant = sorted(similarities, key=lambda x: x[1], reverse=True)
                
                if most_relevant:
                    best_idx = most_relevant[0][0]
                    
                    # Check if the content is substantive enough
                    content = paragraphs[best_idx]
                    if len(content) < 50:  # If too short, combine with next best
                        if len(most_relevant) > 1:
                            next_best_idx = most_relevant[1][0]
                            content += " " + paragraphs[next_best_idx]
                    
                    subsection = {
                        "document": section.get("document", ""),
                        "refined_text": content.strip(),
                        "page_number": section["page"] + 1  # Convert to 1-indexed
                    }
                    subsections.append(subsection)
            except Exception as e:
                print(f"Error processing subsection: {str(e)}")
                
        return subsections
    
    def analyze_collection(self, input_json, pdf_dir):
        """
        Analyze a collection of PDFs based on input JSON
        """
        # Extract metadata
        documents = input_json.get("documents", [])
        persona = input_json.get("persona", {}).get("role", "")
        job_to_be_done = input_json.get("job_to_be_done", {}).get("task", "")
        
        all_sections = []
        doc_map = {}
        
        # Process each document
        for doc_info in documents:
            filename = doc_info.get("filename", "")
            title = doc_info.get("title", "")
            
            pdf_path = os.path.join(pdf_dir, filename)
            if not os.path.exists(pdf_path):
                print(f"Warning: File {pdf_path} not found")
                continue
                
            # Extract sections
            sections = self.extract_sections_with_content(pdf_path)
            
            # Add document info to each section
            for section in sections:
                section["document"] = filename
                section["document_title"] = title
                
            all_sections.extend(sections)
            doc_map[filename] = title
        
        # Rank sections
        ranked_sections = self.rank_sections_by_relevance(all_sections, persona, job_to_be_done)
        
        # Extract subsections
        subsections = self.extract_subsections(ranked_sections, persona, job_to_be_done)
        
        # Prepare output JSON
        output = {
            "metadata": {
                "input_documents": [doc.get("filename", "") for doc in documents],
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": subsections
        }
        
        # Add extracted sections with proper format
        for i, section in enumerate(ranked_sections):
            extracted = {
                "document": section["document"],
                "section_title": section["title"],
                "importance_rank": i + 1,
                "page_number": section["page"] + 1  # Convert to 1-indexed
            }
            output["extracted_sections"].append(extracted)
            
        return output

def main():
    parser = argparse.ArgumentParser(description='PDF Collection Analyzer')
    parser.add_argument('--input', required=True, help='Path to the input JSON file')
    parser.add_argument('--output', required=True, help='Path to the output JSON file')
    args = parser.parse_args()
    
    # Create PDF analyzer
    analyzer = PDFAnalyzer()
    
    # Load input JSON
    with open(args.input, 'r', encoding='utf-8') as f:
        input_json = json.load(f)
    
    # Determine PDF directory from input path
    pdf_dir = os.path.join(os.path.dirname(args.input), "PDFs")
    
    # Analyze collection
    output = analyzer.analyze_collection(input_json, pdf_dir)
    
    # Write output JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    
    print(f"Analysis complete. Output written to {args.output}")

if __name__ == "__main__":
    main()

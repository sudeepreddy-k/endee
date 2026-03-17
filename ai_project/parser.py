import PyPDF2
import re
import json

def parse_ipc_pdf(pdf_path):
    sections = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            full_text_list = []
            # Skip first 14 pages (TOC and arrangement of sections)
            for i in range(14, len(reader.pages)):
                page = reader.pages[i]
                # Append newline to ensure page boundaries don't merge words accidentally
                full_text_list.append(page.extract_text() + "\n")
            
            text = "\n".join(full_text_list)
            
            # --- GLOBAL CLEANING ---
            # 1. Remove common PDF headers/footers
            text = re.sub(r'THE INDIAN PENAL CODE', '', text)
            text = re.sub(r'Page \d+ of \d+', '', text)
            
            # 2. Remove standalone page numbers (digits on their own line)
            text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
            
            # 3. Remove footnote underscore lines (often ___ at end of page)
            text = re.sub(r'\n\s*_{3,}\s*\n', '\n', text)
            
            # 4. Remove Chapter and Title headers that interrupt flow (e.g., CHAPTER XVII, OF THEFT)
            # We look for lines that are primarily uppercase and likely headings
            text = re.sub(r'\n\s*(?:CHAPTER\s+[IVXLCDM]+|OF\s+[A-Z\s]{5,})\n', '\n', text)
            
            # 5. Aggressive Footnote Block Removal
            # Footnotes start with digit+dot and common legalese, and usually end at next footnote or section start
            footnote_pattern = r'\n\s*\d+\.\s+(?:Subs\.|Ins\.|The words|Certain words|The word|Vide|Rep\.|Clause|As to|Amendment|Substituted|Inserted).*?(?=\n\s*(?:\d+\.\s+|\d+[A-Z]*\.\s*[A-Z]|\Z))'
            text = re.sub(footnote_pattern, '\n', text, flags=re.IGNORECASE | re.DOTALL)

            # --- SECTION EXTRACTION ---
            # Pattern: (Number) . (Title) (Separator) (Description)
            # We use a STRICT lookahead: The next section MUST have a number AND a separator (dot-dash or colon)
            # Separator matches dash variants (.— , . — , .--) or a single colon (:)
            # The lazy (.*?) will now correctly capture through footnotes and page breaks until a REAL new section
            section_pattern = r'(?:^|\n)\s*(\d+[A-Z]*)\.\s*([A-Z].*?)\s*([\.\s\u2014\u2013:-]{2,}|:)\s*(.*?)(?=\n\s*\d+[A-Z]*\.\s*[A-Z].*?[\.\s\u2014\u2013:-]{2,}|(?:\n\s*CHAPTER\s+[IVXLCDM]+)|\Z)'
            
            matches = re.finditer(section_pattern, text, re.DOTALL)
            
            for match in matches:
                section_num = match.group(1).strip()
                title = match.group(2).strip()
                description = match.group(4).strip()
                
                # Title cleanup (remove excess spaces and internal mid-title page numbers)
                title = re.sub(r'\s+', ' ', title).strip()
                
                # Description cleanup
                # Remove isolated numbers representing page breaks
                description = re.sub(r'\n\s*\d+\s*\n', ' ', description)
                # Remove any leftover footnote lines missed by global match
                description = re.sub(r'\n\s*\d+\.\s+(?:Subs\.|Ins\.|The words|Certain word|Vide|Rep\.|Clause|Amendment).*?\n', '\n', description, flags=re.IGNORECASE)
                
                # Single pass whitespace collapse
                description = re.sub(r'\s+', ' ', description).strip()
                
                # Minimal validation (length check and ensuring section_num is numeric-ish)
                if len(description) > 20: 
                    sections.append({
                        "section": f"IPC {section_num}",
                        "title": title,
                        "description": description
                    })
            
            # De-duplicate by section number
            unique_sections = []
            seen = set()
            for s in sections:
                if s['section'] not in seen:
                    unique_sections.append(s)
                    seen.add(s['section'])
            
            print(f"Extracted {len(unique_sections)} unique sections.")
            return unique_sections
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        import traceback
        traceback.print_exc()
        return []

def save_to_ipc_data(sections, output_path):
    try:
        content = '"""\nIndian Penal Code (IPC) sections dataset extracted from PDF.\n"""\n\n'
        content += "IPC_SECTIONS = " + json.dumps(sections, indent=4)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully saved {len(sections)} sections to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    pdf_file = 'ipc.pdf'
    output_file = 'ipc_data.py'
    extracted_sections = parse_ipc_pdf(pdf_file)
    if extracted_sections:
        save_to_ipc_data(extracted_sections, output_file)
    else:
        print("No sections extracted. Check parser regex or PDF content.")

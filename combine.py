#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import List

def count_japanese_chars(text: str) -> int:
    """Count Japanese characters (Kanji, Hiragana, Katakana) in text"""
    # Unicode ranges for Japanese characters
    hiragana_range = (0x3040, 0x309F)
    katakana_range = (0x30A0, 0x30FF)
    kanji_range = (0x4E00, 0x9FAF)
    # Extended ranges for less common characters
    kanji_ext_range1 = (0x3400, 0x4DBF)
    kanji_ext_range2 = (0x20000, 0x2A6DF)
    
    count = 0
    for char in text:
        code = ord(char)
        if (hiragana_range[0] <= code <= hiragana_range[1] or
            katakana_range[0] <= code <= katakana_range[1] or
            kanji_range[0] <= code <= kanji_range[1] or
            kanji_ext_range1[0] <= code <= kanji_ext_range1[1] or
            kanji_ext_range2[0] <= code <= kanji_ext_range2[1]):
            count += 1
    
    return count

def filter_lines_by_japanese(text: str, min_japanese: int = 3) -> str:
    """Filter out lines with fewer than min_japanese Japanese characters"""
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        if count_japanese_chars(line) >= min_japanese:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def combine_markdown_files(folder_path: str, output_file: str, min_japanese: int = 3):
    """Combine all Markdown files in a folder into a single text file"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return False
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return False
    
    # Find all Markdown files
    md_files = list(folder.glob('*.md'))
    
    if not md_files:
        print(f"No Markdown files found in '{folder_path}'.")
        return False
    
    print(f"Found {len(md_files)} Markdown files:")
    for md_file in md_files:
        print(f"  - {md_file.name}")
    
    # Process each file
    combined_text = ""
    
    for md_file in sorted(md_files):
        print(f"Processing {md_file.name}...")
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Filter lines by Japanese character count
            filtered_text = filter_lines_by_japanese(content, min_japanese)
            
            # Add file separator
            if combined_text:
                combined_text += "\n\n" + "="*50 + "\n\n"
            
            combined_text += filtered_text
            
        except Exception as e:
            print(f"Error processing {md_file.name}: {e}")
            continue
    
    # Write to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        print(f"Successfully created '{output_file}' with {len(combined_text)} characters.")
        return True
    
    except Exception as e:
        print(f"Error writing to output file: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python combine_md.py <folder_path> <output_file> [min_japanese]")
        print("Example: python combine_md.py ./docs combined.txt 3")
        return
    
    folder_path = sys.argv[1]
    output_file = sys.argv[2]
    min_japanese = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    print(f"Combining Markdown files from '{folder_path}' into '{output_file}'")
    print(f"Filtering lines with fewer than {min_japanese} Japanese characters")
    
    success = combine_markdown_files(folder_path, output_file, min_japanese)
    
    if success:
        print("Operation completed successfully.")
    else:
        print("Operation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Convert leaky_mcqs_new.jsonl to DP-FUSION compatible input.json format

This script transforms MCQ data with private entity annotations into the format
expected by DP-FUSION_simple.py, mapping entity offsets from chunk-level to line-level.
"""

import json
import argparse
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import sys

def find_line_for_position(line_boundaries: List[Tuple[int, int]], position: int) -> int:
    """
    Find which line a character position belongs to.

    Args:
        line_boundaries: List of (start, end) character positions for each line
        position: Character position to locate

    Returns:
        Line index (0-based)
    """
    for i, (start, end) in enumerate(line_boundaries):
        if start <= position < end:
            return i
    # If position is at the very end, return last line
    if position == line_boundaries[-1][1]:
        return len(line_boundaries) - 1
    return -1  # Should not happen with valid data

def convert_entry(mcq_entry: Dict[str, Any], preserve_mcq_data: bool = False) -> Dict[str, Any]:
    """
    Convert a single MCQ entry to DP-FUSION format.

    Args:
        mcq_entry: Dictionary containing MCQ data with private entity annotations
        preserve_mcq_data: Whether to preserve MCQ metadata in output

    Returns:
        Dictionary in DP-FUSION input format
    """
    # Get the raw text with newlines preserved
    chunk_text_raw = mcq_entry.get("chunk_text_raw", "")

    # Split into lines for the passage array
    lines = chunk_text_raw.split('\n')

    # Calculate line boundaries (character positions)
    line_boundaries = []
    current_pos = 0
    for line in lines:
        start = current_pos
        end = current_pos + len(line)
        line_boundaries.append((start, end))
        current_pos = end + 1  # +1 for the newline character

    # Initialize private_entities array (one empty list per line)
    private_entities = [[] for _ in lines]

    # Process each private entity
    chunk_private_spans = mcq_entry.get("chunk_private_spans", [])

    for entity_data in chunk_private_spans:
        # Extract entity information
        entity_text = entity_data.get("text", "")
        entity_type = entity_data.get("label", "MISC")

        # Get chunk-level segments (may have multiple segments)
        chunk_segments = entity_data.get("chunk_segments", [])

        # Process each segment (though typically there's just one)
        for segment in chunk_segments:
            if len(segment) >= 2:
                chunk_start, chunk_end = segment[0], segment[1]

                # Find which line this entity belongs to (based on start position)
                line_idx = find_line_for_position(line_boundaries, chunk_start)

                if line_idx >= 0 and line_idx < len(lines):
                    # Calculate offset within the line
                    line_start = line_boundaries[line_idx][0]

                    # Adjust offsets relative to the line start
                    line_relative_start = chunk_start - line_start
                    line_relative_end = chunk_end - line_start

                    # Create entity object in DP-FUSION format
                    entity_obj = {
                        "text": entity_text,
                        "type": entity_type,
                        "offset": [line_relative_start, line_relative_end]
                    }

                    # Add to the appropriate line's entity list
                    private_entities[line_idx].append(entity_obj)

    # Sort entities within each line by start position
    for line_entities in private_entities:
        line_entities.sort(key=lambda x: x["offset"][0])

    # Create the output dictionary
    output = {
        "passage": lines,
        "private_entities": private_entities
    }

    # Optionally preserve MCQ metadata
    if preserve_mcq_data:
        metadata = {
            "doc_id": mcq_entry.get("doc_id", ""),
            "question": mcq_entry.get("question", ""),
            "options": mcq_entry.get("options", []),
            "answer_index": mcq_entry.get("answer_index", -1),
            "answer_text": mcq_entry.get("answer_text", ""),
            "target_label": mcq_entry.get("target_label", ""),
            "chunk_private_ratio": mcq_entry.get("chunk_private_ratio", 0.0)
        }
        output["metadata"] = metadata

    return output

def main():
    parser = argparse.ArgumentParser(
        description="Convert leaky_mcqs_new.jsonl to DP-FUSION compatible format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="leaky_mcqs_new.jsonl",
        help="Input JSONL file path (default: leaky_mcqs_new.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="input_converted.json",
        help="Output JSON file path (default: input_converted.json)"
    )
    parser.add_argument(
        "--preserve-mcq-data",
        action="store_true",
        help="Preserve MCQ metadata (question, options, answer) in output"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to process (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )

    args = parser.parse_args()

    # Load and process the JSONL file
    output_data = []
    error_count = 0

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            # Apply limit if specified
            if args.limit:
                lines = lines[:args.limit]

            # Process each line
            for i, line in enumerate(tqdm(lines, desc="Converting entries")):
                try:
                    # Parse JSON from line
                    mcq_entry = json.loads(line.strip())

                    # Convert to DP-FUSION format
                    converted_entry = convert_entry(mcq_entry, args.preserve_mcq_data)
                    output_data.append(converted_entry)

                    if args.verbose and i < 3:  # Show first 3 entries as examples
                        print(f"\n=== Entry {i} ===")
                        print(f"Original chunks: {len(mcq_entry.get('chunk_private_spans', []))} private entities")
                        print(f"Converted: {sum(len(entities) for entities in converted_entry['private_entities'])} entities across {len(converted_entry['passage'])} lines")

                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"Error parsing JSON at line {i+1}: {e}", file=sys.stderr)
                except Exception as e:
                    error_count += 1
                    print(f"Error processing entry at line {i+1}: {e}", file=sys.stderr)
                    if args.verbose:
                        import traceback
                        traceback.print_exc()

        # Save the converted data
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Print summary
        print(f"\n✓ Conversion complete!")
        print(f"  - Processed: {len(output_data)} entries")
        if error_count > 0:
            print(f"  - Errors: {error_count} entries skipped")
        print(f"  - Output saved to: {args.output}")

        # Show sample of first entry if verbose
        if args.verbose and output_data:
            print("\n=== Sample Output (First Entry) ===")
            first_entry = output_data[0]
            print(f"Passage lines: {len(first_entry['passage'])}")
            print(f"First line: {first_entry['passage'][0][:100]}..." if first_entry['passage'] else "No passage")
            total_entities = sum(len(entities) for entities in first_entry['private_entities'])
            print(f"Total private entities: {total_entities}")

            # Show first few entities
            for i, line_entities in enumerate(first_entry['private_entities'][:3]):
                if line_entities:
                    print(f"  Line {i}: {len(line_entities)} entities")
                    for entity in line_entities[:2]:  # Show first 2 entities per line
                        print(f"    - {entity['type']}: '{entity['text']}' at {entity['offset']}")

    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
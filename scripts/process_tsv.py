#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# What it does:
# This script extracts data from a delimited file. For each column, it writes
# the data into a separate output file. It primarily assumes the file is a
# standard Tab-Separated Values (TSV) file.
#
# How it works:
# The script reads the input file line by line. For each line, it first tries
# to split the line into columns using a tab character as the delimiter. If
# this results in only one column, it applies a fallback logic and tries to
# split the line by sequences of two or more spaces. This makes the script
# flexible enough to handle non-standard files where spaces are used as
# delimiters instead of tabs.
#
# How to use it:
# You run this script from your command line, calling it "process_tsv.py".
# You must provide the path to the input file and a space-separated list of
# paths for the output files.
#
# Example (for a 3-column file):
# python process_tsv.py --tsv input.tsv --cols column1.txt column2.txt column3.txt
# python ./scripts/process_tsv.py --tsv "./data/corpus/parallel-phrase.tsv" --cols "./tmp/temp-en.txt" "./tmp/temp-zo.txt"
# python ./scripts/process_tsv.py --tsv "./data/corpus/parallel-phrase.tsv" --cols "./data/parallel_base/phrase.en" "./data/parallel_base/phrase.zo"
#
# Arguments:
#   --tsv: The path to the source file. (Required)
#   --cols: A list of output file paths. The number of paths must match the
#           number of columns. (Required)
# -----------------------------------------------------------------------------

import argparse
import re
import sys
from contextlib import ExitStack

def extract_columns(tsv_file_path, output_file_paths):
    """
    Reads a delimited file and writes each column to a separate output file.
    It primarily tries to split by tabs, but falls back to splitting by
    two or more spaces to handle non-standard files.

    Args:
        tsv_file_path (str): The path to the input file.
        output_file_paths (list): A list of paths for the output files.
    """
    num_outputs = len(output_file_paths)
    print(f"Reading data from: {tsv_file_path}")
    print(f"Expecting {num_outputs} columns.")
    print(f"Writing columns to: {', '.join(output_file_paths)}")

    try:
        # ExitStack is used to safely manage a dynamic number of open files.
        # It ensures all files are closed automatically, even if errors occur.
        with ExitStack() as stack:
            # Open the input file for reading
            infile = stack.enter_context(open(tsv_file_path, 'r', encoding='utf-8'))

            # Open all the output files for writing
            outfiles = [stack.enter_context(open(path, 'w', encoding='utf-8')) for path in output_file_paths]

            # We read the file line-by-line to implement custom delimiter logic,
            # instead of using the more rigid csv module.
            for i, line in enumerate(infile):
                # Clean up leading/trailing whitespace from the line
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                # --- NEW DELIMITER LOGIC ---
                # First, try to split by the standard tab delimiter.
                # This ensures the script still works perfectly for proper TSV files.
                row = line.split('\t')

                # If splitting by tab results in only one column, it's likely not a
                # real TSV row. Let's try our fallback for the user's format.
                if len(row) == 1:
                    # Fallback: Split by two or more consecutive whitespace characters.
                    # This handles cases where multiple spaces are used as a delimiter.
                    row = re.split(r'\s{2,}', line)
                # --- END NEW LOGIC ---

                # Now, check if we have the expected number of columns after splitting.
                if len(row) != num_outputs:
                    print(
                        f"Warning: Skipping malformed row {i+1} in {tsv_file_path}. "
                        f"Expected {num_outputs} columns but found {len(row)}.",
                        file=sys.stderr
                    )
                    # Print the problematic line to help with debugging
                    print(f"--> Problematic line content: '{line}'", file=sys.stderr)
                    continue

                # Write each column's data to the corresponding output file
                for col_index, column_data in enumerate(row):
                    # We also strip each individual column to remove extra whitespace
                    outfiles[col_index].write(column_data.strip() + '\n')

        print("\nExtraction complete!")

    except FileNotFoundError:
        print(f"Error: The file '{tsv_file_path}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """
    Main function to parse command-line arguments and run the script.
    """
    parser = argparse.ArgumentParser(
        description="Extracts columns from a TSV or space-delimited file into separate files.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--tsv',
        required=True,
        help="Path to the input TSV or space-delimited file."
    )
    parser.add_argument(
        '--cols',
        required=True,
        nargs='+',  # '+' means 1 or more arguments will be gathered into a list
        help="A space-separated list of output file paths, one for each column."
    )

    args = parser.parse_args()

    extract_columns(args.tsv, args.cols)

if __name__ == '__main__':
    main()

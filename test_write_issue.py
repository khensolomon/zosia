import os

# Define the target directory using the current working directory as base
# This will resolve to C:\dev\zosia\data\tokenizers
target_dir = os.path.join(os.getcwd(), 'data', 'tokenizers')
test_file_path = os.path.join(target_dir, 'diagnose_write_issue.txt')

print(f"--- Starting File Write Diagnostic ---")
print(f"Current Working Directory: {os.getcwd()}")
print(f"Target directory: {target_dir}")
print(f"Test file path: {test_file_path}")

try:
    # 1. Ensure the directory exists
    print(f"Step 1: Ensuring directory '{target_dir}' exists...")
    os.makedirs(target_dir, exist_ok=True)
    print("Step 1: Directory ensured to exist.")

    # 2. Attempt to write the file
    print(f"Step 2: Attempting to write file '{test_file_path}'...")
    with open(test_file_path, 'w') as f:
        f.write("This file is created to diagnose write issues.")
    print("Step 2: File successfully written!")
    print(f"--- DIAGNOSTIC RESULT: SUCCESS ---")
    print(f"Please check '{target_dir}' for 'diagnose_write_issue.txt'.")

except Exception as e:
    # 3. Catch and print any error
    print(f"--- DIAGNOSTIC RESULT: FAILED ---")
    print(f"An error occurred during file writing:")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {e}")
    print(f"Please review the error message for clues.")

print(f"--- Diagnostic Complete ---")
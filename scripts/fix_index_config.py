#!/usr/bin/env python3
"""
Script to replace IndexConfig() constructor calls with IndexConfig.load() calls in test files.
"""

import re
from pathlib import Path
import sys


def fix_index_config_calls(file_path: Path) -> bool:
    """Fix IndexConfig constructor calls in a single file.
    
    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        content = file_path.read_text()
        original_content = content
        
        # Pattern to match IndexConfig(...) calls
        # This handles multi-line calls and various parameter patterns
        pattern = r'IndexConfig\s*\('
        
        # Find all matches and replace them
        def replace_match(match):
            return 'IndexConfig.load('
        
        content = re.sub(pattern, replace_match, content)
        
        if content != original_content:
            file_path.write_text(content)
            print(f"✅ Fixed: {file_path}")
            return True
        else:
            print(f"⏭️  No changes: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix all test files."""
    tests_dir = Path("tests")
    
    if not tests_dir.exists():
        print("❌ tests/ directory not found")
        sys.exit(1)
    
    # Find all Python test files
    test_files = list(tests_dir.glob("*.py"))
    
    if not test_files:
        print("❌ No Python test files found")
        sys.exit(1)
    
    print(f"🔍 Found {len(test_files)} test files")
    
    modified_count = 0
    for test_file in test_files:
        if fix_index_config_calls(test_file):
            modified_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   Files processed: {len(test_files)}")
    print(f"   Files modified: {modified_count}")
    print(f"   Files unchanged: {len(test_files) - modified_count}")


if __name__ == "__main__":
    main()
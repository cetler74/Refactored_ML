#!/usr/bin/env python
"""Fix all indentation issues in the manager.py file by rewriting it properly"""

import re

def main():
    try:
        with open('app/exchange/manager.py.bak', 'r') as f:
            lines = f.readlines()
        
        # Create a new list of lines with proper indentation
        fixed_lines = []
        in_method = False
        current_method = ""
        current_indent = 0
        indent_stack = []
        
        for i, line in enumerate(lines):
            # Check for method declarations
            if re.match(r'^\s*async def \w+', line) or re.match(r'^\s*def \w+', line):
                in_method = True
                current_method = re.search(r'def (\w+)', line).group(1)
                current_indent = line.index('def')
                fixed_lines.append(line)
                continue
            
            # Inside a method
            if in_method:
                # Check if the line is indented properly
                stripped = line.lstrip()
                if not stripped:  # Empty line
                    fixed_lines.append(line)
                    continue
                
                # Calculate the expected indentation
                if stripped.startswith('try:') or stripped.startswith('if ') or stripped.startswith('for ') or stripped.startswith('while ') or stripped.startswith('else:') or stripped.startswith('elif '):
                    expected_indent = current_indent + 4
                    indent_stack.append(expected_indent)
                elif stripped.startswith('except ') or stripped.startswith('finally:'):
                    # Match the indentation of the corresponding try
                    if indent_stack:
                        expected_indent = indent_stack[-1]
                    else:
                        expected_indent = current_indent + 4
                elif stripped.startswith('}') or stripped.startswith(']') or stripped.startswith(')'):
                    # For closing brackets, match the indentation of the opening
                    if indent_stack:
                        indent_stack.pop()
                        expected_indent = indent_stack[-1] if indent_stack else current_indent + 4
                    else:
                        expected_indent = current_indent + 4
                else:
                    # Regular line - use the current indentation level
                    expected_indent = indent_stack[-1] if indent_stack else current_indent + 4
                
                # Fix the indentation
                actual_indent = len(line) - len(line.lstrip())
                if actual_indent != expected_indent:
                    fixed_line = ' ' * expected_indent + stripped
                else:
                    fixed_line = line
                
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        # Write the fixed file
        with open('app/exchange/manager.py', 'w') as f:
            f.writelines(fixed_lines)
        
        print("Fixed indentation in manager.py")
        
    except Exception as e:
        print(f"Error fixing indentation: {str(e)}")
        # Use a different approach - manually fix the most problematic sections
        
        with open('app/exchange/manager.py.bak', 'r') as f:
            content = f.read()
        
        # Fix indentation
        fixes = [
            # fix initialize method
            (r'async def initialize\(self\):[^\n]*\n[^\n]*\n[^\n]*\n\s+credentials', 
             'async def initialize(self):\n        """Initialize connections to configured exchanges."""\n        # Get credentials from settings\n        credentials'),
            
            # fix fetch_ticker method
            (r'if exchange_id not in self\.exchanges:[^\n]*\n[^\n]*\n\s+return \{\}', 
             'if exchange_id not in self.exchanges:\n            logger.error(f"Exchange {exchange_id} not initialized")\n            return {}'),
            
            # fix fetch_usdc_pairs method
            (r'if exchange_id not in self\.exchanges:[^\n]*\n[^\n]*\n\s+return \[\]', 
             'if exchange_id not in self.exchanges:\n            logger.error(f"Exchange {exchange_id} not initialized")\n            return []'),
            
            # fix close method
            (r'if hasattr\(exchange, \'close\'\):[^\n]*\n\s+await exchange\.close\(\)', 
             'if hasattr(exchange, \'close\'):\n                        await exchange.close()'),
            
            # fix direct API close
            (r'for name, api in self\.direct_apis\.items\(\):[^\n]*\n\s+try:[^\n]*\n\s+if hasattr\(api, \'close\'\):',
             'for name, api in self.direct_apis.items():\n                    try:\n                        if hasattr(api, \'close\'):'),
            
            # fix max_retries indentation
            (r'if hasattr\(exchange, \'load_markets\'\):[^\n]*\n\s+max_retries = 3',
             'if hasattr(exchange, \'load_markets\'):\n                    max_retries = 3'),
        ]
        
        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content)
        
        with open('app/exchange/manager.py', 'w') as f:
            f.write(content)
        
        print("Applied specific indentation fixes to manager.py")

if __name__ == "__main__":
    main() 
"""
Improved Duplicate Route Finder - Finds duplicate function names
"""

import re


def find_all_route_functions(filepath='app.py'):
    """Find all Flask route functions and their line numbers"""

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    functions = {}
    current_decorators = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track decorators
        if stripped.startswith('@'):
            current_decorators.append((i, stripped))
        # Found a function definition
        elif stripped.startswith('def '):
            func_match = re.search(r'def (\w+)\(', stripped)
            if func_match:
                func_name = func_match.group(1)

                # Check if this is a route function (has @app.route decorator)
                has_route = any('@app.route' in dec[1] for dec in current_decorators)

                if has_route:
                    if func_name not in functions:
                        functions[func_name] = []

                    # Get the route path
                    route_path = 'unknown'
                    for dec_line, dec_text in current_decorators:
                        if '@app.route' in dec_text:
                            route_match = re.search(r"@app\.route\(['\"]([^'\"]+)", dec_text)
                            if route_match:
                                route_path = route_match.group(1)
                                break

                    functions[func_name].append({
                        'line': i,
                        'route': route_path,
                        'decorators': current_decorators.copy()
                    })

                # Reset decorators after function definition
                current_decorators = []
        # Reset decorators on blank lines or non-decorator lines
        elif not stripped.startswith('@') and not stripped.startswith('#') and stripped:
            if not stripped.startswith('def '):
                current_decorators = []

    return functions


def find_duplicates(functions):
    """Find duplicate function names"""
    duplicates = {}

    for func_name, occurrences in functions.items():
        if len(occurrences) > 1:
            duplicates[func_name] = occurrences

    return duplicates


def print_duplicates(duplicates):
    """Print duplicates in a clear format"""

    if not duplicates:
        print("✅ No duplicate function names found!")
        return False

    print(f"\n⚠️  Found {len(duplicates)} duplicate function name(s):\n")
    print("=" * 70)

    for func_name, occurrences in duplicates.items():
        print(f"\n🔴 Function: {func_name}")
        print(f"   Defined {len(occurrences)} times:")

        for i, occ in enumerate(occurrences, 1):
            print(f"\n   {i}. Line {occ['line']}: {occ['route']}")
            for dec_line, dec_text in occ['decorators']:
                print(f"      {dec_text}")

        print("-" * 70)

    return True


def comment_out_duplicates(duplicates, filepath='app.py'):
    """Comment out duplicate functions (keeping the first occurrence)"""

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Create backup
    backup_file = 'app.py.backup'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"💾 Backup created: {backup_file}\n")

    lines_to_comment = set()

    for func_name, occurrences in duplicates.items():
        # Keep the first occurrence, comment out the rest
        for occ in occurrences[1:]:  # Skip first, comment rest
            func_line = occ['line'] - 1

            # Find start (decorators)
            start_line = func_line
            for dec_line, _ in reversed(occ['decorators']):
                if dec_line < occ['line']:
                    start_line = dec_line - 1

            # Find end of function
            end_line = func_line
            indent = len(lines[func_line]) - len(lines[func_line].lstrip())

            for i in range(func_line + 1, len(lines)):
                curr_line = lines[i].rstrip()
                if curr_line:  # Non-empty line
                    curr_indent = len(lines[i]) - len(lines[i].lstrip())
                    # Function ends when we find a line with same or less indentation
                    if curr_indent <= indent and not curr_line.strip().startswith('#'):
                        end_line = i - 1
                        break
            else:
                end_line = len(lines) - 1

            # Add all lines in this range to comment out
            for i in range(start_line, end_line + 1):
                lines_to_comment.add(i)

            print(f"🔧 Commenting out duplicate '{func_name}' at line {occ['line']}")
            print(f"   (Lines {start_line + 1} to {end_line + 1})")

    # Comment out the lines
    for i in lines_to_comment:
        if i < len(lines) and not lines[i].strip().startswith('#'):
            # Preserve indentation
            indent = len(lines[i]) - len(lines[i].lstrip())
            lines[i] = ' ' * indent + '# DUPLICATE - ' + lines[i].lstrip()

    # Write modified file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"\n✅ Duplicates commented out in {filepath}")
    print(f"✅ Original saved as {backup_file}")


def main():
    print("=" * 70)
    print("Flask Duplicate Function Name Finder")
    print("=" * 70)

    try:
        # Find all route functions
        functions = find_all_route_functions('app.py')

        # Find duplicates
        duplicates = find_duplicates(functions)

        # Print results
        has_duplicates = print_duplicates(duplicates)

        if has_duplicates:
            print("\n" + "=" * 70)
            print("💡 RECOMMENDATION:")
            print("=" * 70)
            print("\nOption 1 (Automatic): Comment out duplicate definitions")
            print("Option 2 (Manual): Rename duplicate functions by adding suffixes")
            print("\nExample manual fix:")
            print("  def get_notifications()     → def get_notifications_api()")
            print("  def mark_notification_read() → def mark_notification_read_api()")

            response = input("\nAutomatically comment out duplicates? (y/n): ")

            if response.lower() == 'y':
                comment_out_duplicates(duplicates)
                print("\n✅ Done! You can now run: python app.py")
            else:
                print("\nPlease fix manually using the information above.")
        else:
            print("\n✅ Your app.py is ready to run!")

    except FileNotFoundError:
        print("\n❌ Error: app.py not found")
        print("Make sure you're in the backend directory")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
import json

with open("tags.json", 'r') as f:
    data = [json.loads(x) for x in f.readlines()]

unique_files = set()
for item in data:
    unique_files.add(item['rel_fname'])

unique_files = ['root/'+x for x in list(unique_files)]
# print(unique_files)



from collections import defaultdict

def build_file_tree_with_relations(file_list):
    # Initialize the root of the tree
    tree = defaultdict(dict)
    relations = list()
    all_directories = set()


    for file in file_list:
        # Split the file path into components
        parts = file.split('/')
        current_level = tree
        parent = None

        # Traverse the path and build the tree structure
        for i, part in enumerate(parts):
            if part not in current_level:
                current_level[part] = {}

            # Record the relation
            if parent is not None:
                all_directories.add(parent)
                relation_type = "folder" if i < len(parts) - 1 else "file"
                relations.append(f'"{parent}" contains {relation_type} "{part}"')

            parent = part
            current_level = current_level[part]

    return tree, relations, list(all_directories)


# Example usage

file_tree, relations, all_directories = build_file_tree_with_relations(list(unique_files))
# [print(x) for x in relations]
# [print(x) for x in all_directories]
# [print(k, v) for k, v in file_tree.items()]


# All def links
defs = list()
defs_dict = dict()
for item in data:
    if item['kind'] == "ref": continue
    """
    This thing also provides those relations
    file contains function
    file contains class
    class contains method
    """
    defs.append((
        item['fname'], item['name'], item['info']
    ))
    defs_dict[
        (item['fname'], item['name'].split()[-1])
    ] = len(defs_dict)

    print(defs[-1][:-1], len(defs_dict)-1)

    if item['category'] == "function" and '.' not in item['name']:
        relations.append(f"{item['rel_fname']} file contains function {item['name']}")
    
    elif item['category'] == "class":
        relations.append(f"{item['rel_fname']} file contains class {item['name']}")
        for method in item['info'].split('\n'):
            relations.append(f"{item['name']} class contains method {method}")

# [print(x) for x in relations]




# resolving ref function calls

import ast
import os

"""
def parse_imports(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())

    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            for alias in node.names:
                imports[alias.asname or alias.name] = f"{module}.{alias.name}"

    return imports

def find_imported_function_location(file_path, function_name):
    # Parse the imports in the file
    imports = parse_imports(file_path)

    # Get the base directory of the file
    base_dir = os.path.dirname(file_path)

    # Check if the function is in the imports dictionary
    # if function_name == "create_objects_array":
    #     print()
    #     print(base_dir, function_name, imports[function_name])
    #     print()
    if function_name in imports:
        import_path = imports[function_name]
        # Resolve the import path to a file path
        parts = (parts := import_path.split('.'))[len(parts)-2:-1]
        resolved_path = os.path.join(base_dir, *parts) + '.py'
        return resolved_path
    else:
        return ""

def get_local_path(file_path, function_name):
    for item in data:
        if item['kind'] == 'def' and item['fname'] == file_path and item['name'].split('.')[-1] == function_name:
            return file_path
    return ""
    
def resolve_reference(file_path, function_name):
    if location := get_local_path(file_path, function_name) or find_imported_function_location(file_path, function_name):
        # return location
        return defs_dict.get(
            (location, function_name), location
        )
    return -1
    """


for item in data:
    if item['kind'] == "def": continue
    ref = resolve_reference(item['fname'], item['name'])
    print(item['rel_fname'], item['name'], ref)
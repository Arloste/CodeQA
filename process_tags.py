import json

with open("tags.json", 'r') as f:
    data = [
        json.loads(x) for x in f.readlines()
    ]

# list unique files

unique_files = set()
for item in data:
    unique_files.add(item['rel_fname'])

unique_files = ["root/"+x for x in list(unique_files)]
unique_files

nodes_csv = {"root": 0}
edges_csv = set()

def extract_relationships(file_paths):
    """
    Extracts unique relationships from a list of file paths.

    Args:
    - file_paths (list of str): A list of file paths.

    Returns:
    - list of str: A list of relationships in the format "folder contains folder" or "folder contains file".
    """
    relationships = set()

    for path in file_paths:
        # Split the path into components
        parts = path.split('/')

        # Generate relationships
        for i in range(1, len(parts)):
            start = 1 if i > 1 else 0
            # Construct the current folder path
            current_folder = '/'.join(parts[start:i])
            # Construct the next part (either a folder or a file)
            next_part = '/'.join(parts[1:i+1])
            
            for folder in [current_folder, next_part]:
                if folder not in nodes_csv:
                    nodes_csv[folder] = len(nodes_csv)

            relation = "folder" if i < len(parts) - 1 else "file"
            relationships.add(f"{current_folder} contains {relation} {next_part}")
            edges_csv.add((
                nodes_csv[current_folder], f"contains {relation}", nodes_csv[next_part]
            ))

    return list(relationships)

relationships = extract_relationships(unique_files)

# All def links
defs = list()
defs_dict = dict()

for item in data:
    if item['kind'] == "ref": continue
    """f
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

    loc = (item['rel_fname'], item['name'])
    if loc not in nodes_csv:
        nodes_csv[loc] = len(nodes_csv)

    if item['category'] == "function" and '.' not in item['name']:
        edges_csv.add((
            nodes_csv[item['rel_fname']], "file contains function", nodes_csv[loc]
        ))
    
    elif item['category'] == "class":
        edges_csv.add((
            nodes_csv[item['rel_fname']], "file contains class", nodes_csv[loc]
        ))
        for method in item['info'].split('\n'):
            method_loc = (item['rel_fname'], method)
            if method_loc not in nodes_csv:
                nodes_csv[method_loc] = len(nodes_csv)
            edges_csv.add((
                nodes_csv[loc], "class contains method", nodes_csv[method_loc]
            ))



import ast
import os

def resolve_reference(name, file_path, project_root):
    """
    Resolves the reference of a function, class, or method to its definition.

    Args:
    - name (str): The name of the function, class, or method to resolve.
    - file_path (str): The relative file path where the name is referenced, starting from the project root.
    - project_root (str): The root directory of the project.

    Returns:
    - str: The file path of the definition, or an empty string if it's a built-in or external module.
    """

    # Parse the file to analyze imports and definitions
    with open(os.path.join(project_root, file_path), 'r') as file:
        tree = ast.parse(file.read())

    # Track imports
    imports = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            for alias in node.names:
                imports[alias.asname or alias.name] = f"{module}.{alias.name}"

    # Check if the name is imported
    if name in imports:
        import_path = imports[name]
        if '.' in import_path:
            # Handle nested imports
            module_parts = import_path.split('.')
            module_file = os.path.join(project_root, *module_parts[:-1], f"{module_parts[-2]}.py")
            if os.path.exists(module_file):
                return module_file
        else:
            # Handle top-level imports
            module_file = os.path.join(project_root, f"{import_path}.py")
            if os.path.exists(module_file):
                return module_file

    # Search the project directory for the definition
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                module_file_path = os.path.join(root, file)
                with open(module_file_path, 'r') as f:
                    module_tree = ast.parse(f.read())
                    for node in ast.walk(module_tree):
                        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
                            if node.name == name:
                                return module_file_path

    # If the name is not found, assume it's an external module
    return "No location detected"

project_root = 'test_input/codegraph-main'

current_ref = None
for i, item in enumerate(data):
    if item['kind'] == "def":
        current_ref = item.copy()
        continue

    name = item['name']
    file_path = item['rel_fname']

    definition_location = resolve_reference(name, file_path, project_root)
    definition_location = definition_location[len(project_root)+1:]

    location_id = nodes_csv.get((definition_location, name), -1)
    
    if location_id == -1: continue

    edges_csv.add((
        nodes_csv[(current_ref['rel_fname'], current_ref['name'])],
        "references", location_id
    ))


for item in data:
    loc = (item['rel_fname'], item['name'])
    node_id = nodes_csv.get(loc, -1)
    if node_id == -1 or item['kind'] == "ref": continue
    del nodes_csv[loc]
    nodes_csv[node_id] = f"# location: {item['rel_fname']}\n{item['info']}"

nodes_csv_items = nodes_csv.copy().items()
for k, v in nodes_csv_items:
    if type(v) == int:
        del nodes_csv[k]
        nodes_csv[v] = k


nodes = [(k, v) for k, v in nodes_csv.items()]
nodes = sorted(nodes, key=lambda l:l[0])

with open("test_input/output/nodes.csv", 'w') as f:
    f.write("id,name\n")
    for item in nodes:
        data = item[1].replace("\n", '\\n')
        f.write(f"{item[0]},{data}\n")

with open("test_input/output/edges.csv", 'w') as f:
    f.write("id_head,type,id_tail\n")
    for item in edges_csv:
        f.write(f"{item[0]},{item[1]},{item[2]}\n")
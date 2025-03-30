import json
from tqdm import tqdm
import os

TAGS_FILE_PATH = "tags.json"
REPO_PATH = "input_repository"
# OUTPUT_FOLDER_PATH = "G-Retriever/dataset" # TODO check path correctness # PROD output folder
OUTPUT_FOLDER_PATH = '.' # dev output folder

with open(TAGS_FILE_PATH, 'r') as f:
    data = [
        json.loads(x) for x in f.readlines()
    ]

# list unique files

unique_files = set()
print("Stage 1: getting all unique files")
for item in data:
    unique_files.add(item['rel_fname'])


# list all plain text files
# (because RepoGraph lists only .py files)

plain_text_file_path = REPO_PATH + '/'
file_types_to_consider = ["txt", 'md', 'yml']
plain_text_files = list()

def walk_dir(path, ind=0):
    for name in os.listdir(path):
        if os.path.isdir(path + name + '/'):
            walk_dir(path + name + '/', ind+4)
        else:
            ext = name.split('.')[-1]
            if ext in file_types_to_consider:
                try:
                    with open(path + name, 'r', encoding="utf-8") as f:
                        contents = f.read()
                        full_path = path + name
                    plain_text_files.append(
                        (full_path[len(plain_text_file_path):], contents)
                    )
                except:
                    print("Problem with file: ", path + name)
                

walk_dir(plain_text_file_path)
plain_text_dict = {v[0]: v[1] for v in plain_text_files}

for item in plain_text_files:
    unique_files.add(item[0])


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

    print("Stage 2: creating the file structure")

    for path in tqdm(file_paths):
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
            relationships.add(f"{current_folder} folder contains {relation} {next_part}")
            edges_csv.add((
                nodes_csv[current_folder], f"folder contains {relation}", nodes_csv[next_part]
            ))

    return list(relationships)

relationships = extract_relationships(unique_files)

# All def links
defs = list()
defs_dict = dict()

print("Stage 3: parsing definitions")
for item in tqdm(data):
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


print("Stage 4: resolving references")
import ast
import os

traversed_files_dict = {}

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
    imports = {}
    if (project_root, file_path) not in traversed_files_dict:
        
        try:
            # Track imports
            

            with open(os.path.join(project_root, file_path), 'r') as file:
                tree = ast.parse(file.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[alias.asname or alias.name] = alias.name
                elif isinstance(node, ast.ImportFrom):
                    module = node.module
                    for alias in node.names:
                        imports[alias.asname or alias.name] = f"{module}.{alias.name}"
        except: pass
        traversed_files_dict[(project_root, file_path)] = imports

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

    # Search all defined functions and search for the definition
    for item in defs:
        if name in item[1][:60]:
            return item[0]

    # If the name is not found, assume it's an external module
    return "No location detected"



current_ref = None
for i, item in enumerate(tqdm(data)):
    if item['kind'] == "def":
        current_ref = item.copy()
        continue

    name = item['name']
    file_path = item['rel_fname']

    definition_location = resolve_reference(name, file_path, REPO_PATH)
    definition_location = '/'.join(definition_location.split('/')[1:])

    
    location_id = nodes_csv.get((definition_location, name), -1)
    
    if location_id == -1: continue
    try:
        edges_csv.add((
            nodes_csv[(current_ref['rel_fname'], current_ref['name'])],
            "references", location_id
        ))
    except: pass

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

nodes_csv_items = nodes_csv.copy().items()
for node_id, node_location in nodes_csv_items:
    if text := plain_text_dict.get(node_location, ""):
        nodes_csv[node_id] = f"{node_location} file contents:\n{text}"


print("Step 4.1: Update file information")
file2contents = {}

for file_name in tqdm(unique_files):
    if not file_name.endswith(".py"): continue
    file_name = '/'.join(file_name.split('/')[1:])

    with open(f"{REPO_PATH}/" + file_name, 'r') as f:
        contents = f.readlines()
    
    lines_to_remove = set()
    for tag in data:
        if tag["rel_fname"] == file_name and type(tag["line"])  == list and len(tag["line"]) == 2 and tag["kind"] == "def":
            start, end = tag["line"]
            [lines_to_remove.add(x) for x in range(start, end+1)]
    contents = [contents[x] for x in range(len(contents)) if x+1 not in lines_to_remove]
    file2contents[file_name] = ''.join(contents)

nodes_csv_items = nodes_csv.copy().items()
for node_id, node_location in nodes_csv_items:
    if text := file2contents.get(node_location, ""):
        nodes_csv[node_id] = f"#{node_location}'s text outside functions:\n{text}"

# Adding class textual information that is outside the class methods
# this is also not captured by RepoGraph
print("Step 4.2: Update class information")

class2contents = {}

for i, item in tqdm(enumerate(data), total=len(data)):
    if item['category'] == "class" and item['kind'] == 'def':
        methods = set(item['info'].split('\n'))
        class_def_span = set(range(*item['line']))
        
        lines_to_exclude = set()
        for sub_item in data[i:]:
            if sub_item['name'] in methods and sub_item['kind'] == 'def':
                try:
                    start, end = sub_item['line']
                    [lines_to_exclude.add(x) for x in range(start, end+1)]
                except: print("Could not read the lines")
        lines_left = class_def_span  - lines_to_exclude

        with open(f"{REPO_PATH}/" + item['rel_fname'], 'r') as f:
            contents = f.readlines()
        contents = [contents[x-1] for x in sorted(list(lines_left))]
        class2contents[(item['rel_fname'], item['name'])] = ''.join(contents)

for node_id, node_text in nodes_csv.items():
    for (class_file_name, class_name), class_text in class2contents.items():
        if node_text.count(' ') == 2 and class_file_name in node_text and class_name in node_text:
            class_location, *class_methods = node_text.split('\n')
            text = f"{class_location}\n{class_text}\n#This class contains functions: {class_methods}"
            nodes_csv[node_id] = text
            break

print("Stage 5: writing nodes to files")
# Sorting the nodes so that their IDs in the ascending order

nodes = [(k, v) for k, v in nodes_csv.items()]
nodes = sorted(nodes, key=lambda l:l[0])

with open(f"{OUTPUT_FOLDER_PATH}/nodes.csv", 'w') as f:
    f.write("id,name\n")
    for item in nodes:
        try:
            data = item[1].replace("\n", '\\n')
            f.write(f"{item[0]},{data}\n")
        except:
            pass

with open(f"{OUTPUT_FOLDER_PATH}/edges.csv", 'w') as f:
    f.write("id_head,type,id_tail\n")
    for item in edges_csv:
        f.write(f"{item[0]},{item[1]},{item[2]}\n")


import importlib.util
import sys

# Path to the module
folder_name = 'G-Retriever'
module_name = 'src/utils/lm_modeling'
module_path = f"./{folder_name}/{module_name}.py"

# Load the module
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

with open(f"{OUTPUT_FOLDER_PATH}/edges.csv") as f:
    edges = f.readlines()[1:]
    edges = [x.strip().split(',') for x in edges]

with open(f"{OUTPUT_FOLDER_PATH}/nodes.csv") as f:
    nodes = f.readlines()[1:]
    nodes = [x.strip().split(',') for x in nodes]
    node_ids = [x[0] for x in nodes]
    nodes_texts = [','.join(x[1:]) for x in nodes]

# Removing edges with missing nodes
nodes_set = set(node_ids)
edges = [e for e in edges if e[0] in nodes_set and e[2] in nodes_set]
heads, relation_types, tails = zip(*edges)

# Remapping nodes (due to Torch Geometric issue)
node_mapping = dict()
for i, node_id in enumerate(node_ids):
    node_mapping[node_id] = str(i)

node_ids = [node_mapping[n] for n in node_ids]
heads = [node_mapping[x] for x in heads]
tails = [node_mapping[x] for x in tails]

# Writing updated nodes and edges to files
with open(f"{OUTPUT_FOLDER_PATH}/edges.csv", 'w') as f:
    f.write("id_head,type,id_tail\n")
    for h, r, t in zip(heads, relation_types, tails):
        f.write(f"{h},{r},{t}\n")

with open(f"{OUTPUT_FOLDER_PATH}/nodes.csv", 'w') as f:
    f.write("id,name\n")
    for idx, node_name in zip(node_ids, nodes_texts):
        f.write(f"{idx},{node_name}\n")


# default sentence-transformer name
llm_module_name = "sbert"

model, tokenizer, device = module.load_model[llm_module_name]()
text2embedding = module.load_text2embedding[llm_module_name]


from tqdm.notebook import tqdm
import torch
from torch_geometric.data.data import Data
import pandas as pd


# Process nodes
print("Encoding graph nodes...")
x = text2embedding(model, tokenizer, device, nodes_texts)

# Process relation types
print("Encoding graph edges...")
e = text2embedding(model, tokenizer, device, relation_types)

# Create edge index tensor
edge_index = torch.LongTensor([
    pd.Series(heads).astype(int), pd.Series(tails).astype(int)
])

# Create graph data structure
data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes_texts))

# Save the graph data
torch.save(data, f"{OUTPUT_FOLDER_PATH}/graph.pt")
# creates a file with all function definitions and references
python RepoGraph/repograph/construct_graph.py test_input/test_repo/

# output location
mkdir test_input/output

# creates a repository graph with dependencies
python process_tags.py

# cleanup
rm tags.json
rm graph.pkl
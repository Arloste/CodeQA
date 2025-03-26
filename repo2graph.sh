# creates a file with all function definitions and references
python RepoGraph/repograph/construct_graph.py test_repo

# creates a repository graph with dependencies
python process_tags.py

# cleanup
rm tags.json
rm graph.pkl
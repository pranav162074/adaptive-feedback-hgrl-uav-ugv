test_nodes.txt		- 400 nodes of a 20*20 square map in the form "id x y"
testAssignment.txt	- 50499 random pairs of nodes ids from test_nodes.txt
test_edgesX.txt		- pairs of adjacent nodes ids from test_nodes.txt forming edges
			- X = 0 - tree
			- X = 20 - full graph
			- created starting at full graph and repeatedly erasing edges until a tree remains

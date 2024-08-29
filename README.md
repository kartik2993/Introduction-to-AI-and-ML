There are 10 Maze example problem in .txt format.

Depth first search is the search algorithm where we always explore the deepest node in the frontier. We keep going deeper and deeper through our search tree. And then if we hit a dead end, we back up and we try something else instead.It may find the shortest path but not sure each time.

Breadth first search, which behaves very similarly to depth first search with one difference. Instead of always exploring the deepest node in the search tree the way the depth first search does, breadth first search is always going to explore the shallowest node in the frontier.

Greedy Best-First Search, often abbreviated GBFS,
is a search algorithm that, instead of expanding the deepest node,
like DFS, or the shallowest node, like BFS,
this algorithm is always going to expand the node
that it thinks is closest to the goal. It doesn't always estimate the optimal or shortest path.

A star search is going to solve this problem by,
instead of just considering the heuristic,
also considering how long it took us to get to any particular state.
So the distinction is greedy best-first search, if I am in a state
right now, the only thing I care about is
what is the estimated distance, the heuristic value, between me
and the goal.
Whereas A star search will take into consideration
two pieces of information.
It'll take into consideration, how far do I estimate I am from the goal,
but also how far did I have to travel in order to get here?
Because that is relevant, too.
So we'll search algorithms by expanding the node with the lowest
value of g(n) plus h(n).

# Search methods

import search

a_b = search.GPSProblem('A', 'B', search.romania)
a_t = search.GPSProblem('A', 'T', search.romania)
o_e = search.GPSProblem('O', 'E', search.romania)
t_n = search.GPSProblem('T', 'N', search.romania)
t_e = search.GPSProblem('T', 'E', search.romania)

#print search.breadth_first_graph_search(ab).path()
#print search.depth_first_graph_search(ab).path()

#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()

print "\nRuta A -> B"
print "Busqueda por ramificacion y acotacion", search.search_ramification(a_b).path()
print 'Busqueda Informada:', search.search_heuristic(a_b).path()

print "\nRuta A -> T"
print "Busqueda por ramificacion y acotacion", search.search_ramification(a_t).path()
print 'Busqueda Informada:', search.search_heuristic(a_t).path()

print "\nRuta O -> E"
print "Busqueda por ramificacion y acotacion", search.search_ramification(o_e).path()
print 'Busqueda Informada:', search.search_heuristic(o_e).path()

print "\nRuta T -> N"
print "Busqueda por ramificacion y acotacion", search.search_ramification(t_e).path()
print 'Busqueda Informada:', search.search_heuristic(t_e).path()




#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450

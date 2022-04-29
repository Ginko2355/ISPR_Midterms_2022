# Implement a Bayesian Network (BN) comprising at least 10 nodes, all with binomial
# or multinomial distribution. Represent the BN with the data structures that you deem
# appropriate and in the programming language that you prefer. The BN should model some
# problem/process of your choice, so you are also free to define the topology according
# to your prior knowledge (just be ready to justify your choices). For instance, you can
# define a BN to represent a COVID diagnosis through a certain number of events/exams/symptoms:
# e.g. Cough, Cold, Fever, Breathing problems, Swab Exam, etc.
#
# Or you can model your daily routine: Wakeup, Coffee, Toilet, Study, Lunch, etc.
# Once you have modelled the BN, also plug in the necessary local conditional probability
# tables. You can set the values of the probabilities following your own intuition on the
# problem (ie no need to learn them from data). Then run some episoded of Ancestral Sampling
# on the BN and discuss the results.
#
# The assignment needs to be fully implemented by you, without using BN libraries.
# Add a vertex to the dictionary


# Ogni volta che aggiungo un nodo devo chiedere se ha qualche arco entrante da qualche
# altro nodo GIA' NEL GRAFO, così che posso chiedere la tabella delle probabilità
# di una giusta dimensione
def add_node(node, probability_table):
    global bn_vertices
    global vertices_no
    if node in bn_vertices:
        print("Vertex ", node, " already exists.")
    else:
        entering_edges = 0
        while True:
            print("There are any arcs entering into ", node, " between the existing nodes?")
            choice = input("Y/N: ")
            if str.upper(choice) == "Y" and len(bn_vertices) > 0:
                print("List of nodes: ", bn_vertices.keys())
                entering_node = input("Write the node that have an entering arc into n: ")
                if entering_node not in bn_vertices:
                    print("Node ", entering_node, " not exists.")
                else:
                    add_edge(entering_node, node)

                    entering_edges += 1
            elif len(bn_vertices) == 0:
                print("Not enough nodes")
                break
            else:
                break

        vertices_no = vertices_no + 1
        # VEDERE SE LA PROBABILITY TABLE E' DELLE DIMENSIONI GIUSTE
        # SE FACCIO BINOMIAL DISTRIBUTIONS LA DIM DEVE ESSERE 2^N x 2
        bn_vertices[node] = probability_table

# Add an edge between vertex v1 and v2 with edge weight e
def add_edge(n1, n2):
    global bn_vertices
    # Check if vertex v1 is a valid vertex
    if n1 not in bn_vertices:
        print("Vertex ", n1, " does not exist.")
    # Check if vertex v2 is a valid vertex
    elif n2 not in bn_vertices:
        print("Vertex ", n2, " does not exist.")
    else:
        bn_vertices[n1].append(n2)

# Print the graph
def print_graph():
    global bn_vertices
    for vertex in bn_vertices:
        for edges in bn_vertices[vertex]:
            print(vertex, " -> ", edges[0], " edge weight: ", edges[1])

# driver code
bn_vertices = {}
if __name__ == '__main__':
    # stores the number of vertices in the graph
    vertices_no = 0
    add_node(1)
    add_node(2)
    add_node(3)
    add_node(4)
    # Add the edges between the vertices by specifying
    # the from and to vertex along with the edge weights.
    add_edge(1, 2, 1)
    add_edge(1, 3, 1)
    add_edge(2, 3, 3)
    add_edge(3, 4, 4)
    add_edge(4, 1, 5)
    print_graph()
    # Reminder: the second element of each list inside the dictionary
    # denotes the edge weight.
    print("Internal representation: ", bn_vertices)
    print("Midterm 2 Assignment 4 Alessandro Bucci")
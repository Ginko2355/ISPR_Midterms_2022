
class Graph(object):
    # Initialize the graph
    def __init__(self, number_of_nodes):
        self.nodes_indexes = dict()
        self.nodes_ids = dict()
        self.node_cpt = dict()
        self.node_probabilities = dict()
        self.configuration = dict()
        self.adjMatrix = []
        for i in range(number_of_nodes):
            self.adjMatrix.append([0 for i in range(number_of_nodes)])
        self.size = number_of_nodes
    # Add node
    def add_node(self,node, cpt=None):
        if len(self.nodes_indexes) >= self.size:
            print("Maximum number of nodes already reached,", node, "can't be added")
        else:
            self.nodes_indexes[node] = len(self.nodes_indexes)
            self.nodes_ids[node] = len(self.nodes_ids)+1
            self.node_cpt[node] = cpt
    # Add edges
    def add_edge(self, node1, node2, directed=False):
        if node1 not in self.nodes_indexes.keys():
            print(node1, " not exist")
        elif node2 not in self.nodes_indexes.keys():
            print(node2, " not exist")
        else:
            if node1 == node2:
                print(node1, " and ", node2, " are the same node")
            id_node_1 = self.nodes_indexes[node1]
            id_node_2 = self.nodes_indexes[node2]

            self.adjMatrix[id_node_1][id_node_2] = 1
            if not directed:
                self.adjMatrix[id_node_2][id_node_1] = 1

    # Remove edges
    def remove_edge(self, node1, node2, directed = False):
        if self.adjMatrix[node1][node2] == 0:
            print("No edge between", node1, "and", node2)
            return
        id_node_1 = self.nodes_indexes[node1]
        id_node_2 = self.nodes_indexes[node2]

        self.adjMatrix[id_node_1][id_node_2] = 0
        if not directed:
            self.adjMatrix[id_node_2][id_node_1] = 0


    def __len__(self):
        return self.size

    # Print the matrix
    def print_matrix(self):
        print("Matrix: ")
        for row in self.adjMatrix[:len(self.nodes_ids)]:
            print(row[:len(self.nodes_ids)])

    def print_nodes(self):
        print("Nodes:  ")
        for node, nindex in self.nodes_indexes.items():
            print(node, ":", str(nindex))

    @staticmethod
    def array_is_zero(a):
        return not np.any(a)

    def topological_order(self):
        tmp_matrix = np.array(self.adjMatrix)[0:len(self.nodes_ids),0:len(self.nodes_ids)]
        topological_index_list = list()
        topological_order= list()
        while not Graph.array_is_zero(tmp_matrix):
            tmp_list = topological_index_list.copy()
            for id in topological_index_list:
                for row_id in range(len(tmp_matrix)):
                    if row_id == id:
                        tmp_matrix[row_id] = np.zeros(np.shape(tmp_matrix[row_id]))

            for row_id, row in enumerate(tmp_matrix.transpose()):
                if Graph.array_is_zero(row) and row_id not in topological_index_list:
                    topological_index_list.append(row_id)

            if tmp_list == topological_index_list:
                 print("The graph is not acyclic")
                 return

        for node_index in topological_index_list:
            topological_order.append(self.get_node_from_id(node_index+1))

        return topological_order

    def get_node_from_id(self, id):
        if id<=0:
            print("The id starts from 1")
            return
        return list(self.nodes_ids.keys())[list(self.nodes_ids.values()).index(id)]

    def get_node_from_index(self, index):
        return list(self.nodes_indexes.keys())[list(self.nodes_indexes.values()).index(index)]

    def print_nodes_id(self):
        print("Nodes:  ")
        for node, nid in self.nodes_ids.items():
            print(node, ":", str(nid))

    def load_CPT_test(self):
        CPT = dict()
        #First cpt 
        CPT[frozenset([1,2])]= [0.95,0.05]
        CPT[frozenset([1,-2])]= [0.94,0.06]
        CPT[frozenset([-1,2])]= [0.29,0.71]
        CPT[frozenset([-1,-2])]= [0.001,0.999]
        self.node_cpt['Alarm']=CPT.copy()
        CPT.clear()

        CPT[frozenset([])]= [0.001,0.999]
        self.node_cpt['Burglary'] = CPT.copy()
        CPT.clear()

        CPT[frozenset([])]= [0.002,0.998]
        self.node_cpt['Earthquake'] = CPT.copy()
        CPT.clear()

        CPT[frozenset([3])] = [0.90, 0.10]
        CPT[frozenset([-3])] = [0.05, 0.95]
        self.node_cpt['JohnCalls'] = CPT.copy()
        CPT.clear()

        CPT[frozenset([3])] = [0.70, 0.30]
        CPT[frozenset([-3])] = [0.01, 0.99]
        self.node_cpt['MaryCalls'] = CPT.copy()
        CPT.clear()

    def load_CPT(self):
        CPT = dict()
        PLN = "Party last night"
        SL = "Studying late"
        GTBL = "Going to bed late"
        AIS = "Alarm is set"
        WUE = "Wake up early"
        HB = "Have breakfast"
        S = "Studying"
        HL = "Have lunch"
        HDAH ="Have dinner at home"
        HDWF = "Have dinner with friends"

        CPT[frozenset([])] = [0.5, 0.5]
        self.node_cpt[PLN] = CPT.copy()
        CPT.clear()

        CPT[frozenset([])] = [0.5, 0.5]
        self.node_cpt[SL] = CPT.copy()
        CPT.clear()

        CPT[frozenset([])] = [0.8, 0.2]
        self.node_cpt[AIS] = CPT.copy()
        CPT.clear()

        CPT[frozenset([self.nodes_ids[PLN],self.nodes_ids[SL]])]= [0.99,0.01]
        CPT[frozenset([self.nodes_ids[PLN],-self.nodes_ids[SL]])]= [0.7,0.3]
        CPT[frozenset([-self.nodes_ids[PLN],self.nodes_ids[SL]])]= [0.5,0.5]
        CPT[frozenset([-self.nodes_ids[PLN],-self.nodes_ids[SL]])]= [0.01,0.99]
        self.node_cpt[GTBL]=CPT.copy()
        CPT.clear()

        CPT[frozenset([self.nodes_ids[AIS], self.nodes_ids[GTBL]])] = [0.6, 0.4]
        CPT[frozenset([self.nodes_ids[AIS], -self.nodes_ids[GTBL]])] = [0.9, 0.1]
        CPT[frozenset([-self.nodes_ids[AIS], self.nodes_ids[GTBL]])] = [0.2, 0.8]
        CPT[frozenset([-self.nodes_ids[AIS], -self.nodes_ids[GTBL]])] = [0.7, 0.3]
        self.node_cpt[WUE] = CPT.copy()
        CPT.clear()

        CPT[frozenset([self.nodes_ids[WUE]])] = [0.9, 0.1]
        CPT[frozenset([-self.nodes_ids[WUE]])] = [0.3, 0.7]
        self.node_cpt[HB] = CPT.copy()
        CPT.clear()

        CPT[frozenset([self.nodes_ids[WUE], self.nodes_ids[HB]])] = [0.9, 0.1]
        CPT[frozenset([self.nodes_ids[WUE], -self.nodes_ids[HB]])] = [0.7, 0.3]
        CPT[frozenset([-self.nodes_ids[WUE], self.nodes_ids[HB]])] = [0.5, 0.5]
        CPT[frozenset([-self.nodes_ids[WUE], -self.nodes_ids[HB]])] = [0.3, 0.7]
        self.node_cpt[S] = CPT.copy()
        CPT.clear()

        CPT[frozenset([self.nodes_ids[HB], self.nodes_ids[S]])] = [0.7, 0.3]
        CPT[frozenset([self.nodes_ids[HB], -self.nodes_ids[S]])] = [0.5, 0.5]
        CPT[frozenset([-self.nodes_ids[HB], self.nodes_ids[S]])] = [1., 0.]
        CPT[frozenset([-self.nodes_ids[HB], -self.nodes_ids[S]])] = [0.9, 0.1]
        self.node_cpt[HL] = CPT.copy()
        CPT.clear()

        CPT[frozenset([self.nodes_ids[S], self.nodes_ids[HL]])] = [0.5, 0.5]
        CPT[frozenset([self.nodes_ids[S], -self.nodes_ids[HL]])] = [0.8, 0.2]
        CPT[frozenset([-self.nodes_ids[S], self.nodes_ids[HL]])] = [0.3, 0.7]
        CPT[frozenset([-self.nodes_ids[S], -self.nodes_ids[HL]])] = [0.99, 0.01]
        self.node_cpt[HDAH] = CPT.copy()
        CPT.clear()

        CPT[frozenset([self.nodes_ids[HDAH],self.nodes_ids[S], self.nodes_ids[HL]])] = [0., 1.]
        CPT[frozenset([self.nodes_ids[HDAH],self.nodes_ids[S], -self.nodes_ids[HL]])] = [0., 1.]
        CPT[frozenset([self.nodes_ids[HDAH],-self.nodes_ids[S], self.nodes_ids[HL]])] = [0., 1.]
        CPT[frozenset([self.nodes_ids[HDAH],-self.nodes_ids[S], -self.nodes_ids[HL]])] = [0., 1.]
        CPT[frozenset([-self.nodes_ids[HDAH],self.nodes_ids[S], self.nodes_ids[HL]])] = [0.7, 0.3]
        CPT[frozenset([-self.nodes_ids[HDAH],self.nodes_ids[S], -self.nodes_ids[HL]])] = [0.9, 0.1]
        CPT[frozenset([-self.nodes_ids[HDAH],-self.nodes_ids[S], self.nodes_ids[HL]])] = [0.2, 0.8]
        CPT[frozenset([-self.nodes_ids[HDAH],-self.nodes_ids[S], -self.nodes_ids[HL]])] = [0.99, 0.01]
        self.node_cpt[HDWF] = CPT.copy()
        CPT.clear()


    def parent_list(self,node):
        parent_list = list()
        tmp_matrix = np.array(self.adjMatrix)
        node_index = self.nodes_indexes[node]

        for parent_index, arc in enumerate(tmp_matrix.transpose()[node_index]):
            if arc:
                parent_list.append(self.get_node_from_index(parent_index))

        return parent_list

    def get_node_probability(self, node):
        parent_list = self.parent_list(node)
        parents_id_set = set()

        if node not in self.node_cpt.keys():
            print("Errore")
            return

        for parent in parent_list:
            if parent not in self.node_cpt.keys():
                print("Errore")
                return

            if self.configuration[parent]:
                parents_id_set.add(self.nodes_ids[parent])
            else:
                parents_id_set.add(-self.nodes_ids[parent])

        if self.configuration[node]:
             return self.node_cpt[node][frozenset(parents_id_set)][0]
        else:
             return self.node_cpt[node][frozenset(parents_id_set)][1]

    def get_node_probabilities(self, node):
        parent_list = self.parent_list(node)
        parents_id_set = set()

        if node not in self.node_cpt.keys():
            print("Errore")
            return

        for parent in parent_list:
            if parent not in self.node_cpt.keys():
                print("Errore")
                return

            if self.configuration[parent]:
                parents_id_set.add(self.nodes_ids[parent])
            else:
                parents_id_set.add(-self.nodes_ids[parent])

        return self.node_cpt[node][frozenset(parents_id_set)]

    def set_configuration(self,**kwargs):
        self.configuration = kwargs.copy()

    def joint_probability(self):
        topological_order = self.topological_order()
        joint_probability = 1.

        for node in topological_order:
            joint_probability *= self.get_node_probability(node)

        return joint_probability

    def ancestral_sampling(self, iterations):
        topological_order = self.topological_order()
        distribution = dict()
        repetition_for_each_sample = dict()

        for i in range(iterations):
            self.configuration.clear()
            for node in topological_order:
                p_node = self.get_node_probabilities(node)
                sampled_node = np.random.choice([True,False],1,p=p_node)[0]
                self.configuration[node] = sampled_node
            if tuple((self.configuration.items())) not in distribution:
                distribution[tuple((self.configuration.items()))] = self.joint_probability()
                repetition_for_each_sample[tuple((self.configuration.items()))] = 0
            else:
                repetition_for_each_sample[tuple((self.configuration.items()))] += 1

        return distribution,repetition_for_each_sample


def test():
    g = Graph(10)

    g.add_node("Burglary")
    g.add_node("Earthquake")
    g.add_node("Alarm")
    g.add_node("JohnCalls")
    g.add_node("MaryCalls")

    g.add_edge("Burglary", "Alarm", True)
    g.add_edge("Earthquake", "Alarm", True)
    g.add_edge("Alarm", "JohnCalls", True)
    g.add_edge("Alarm", "MaryCalls", True)
    g.print_nodes_id()
    g.print_matrix()

    kwargs={"Burglary" : False, "Earthquake" : False, "Alarm" : False,
             "JohnCalls" : False, "MaryCalls" : True}
    #g.set_configuration(**kwargs)
    g.load_CPT()
    #print("Joint probability: ",g.joint_probability())



def main():
    g = Graph(10)
    g.add_node("Party last night") #1
    g.add_node("Studying late") #2
    g.add_node("Going to bed late") #3
    g.add_node("Alarm is set") #4
    g.add_node("Wake up early") #5
    g.add_node("Have breakfast") #6
    g.add_node("Studying") #7
    g.add_node("Have lunch") #8
    g.add_node("Have dinner at home") #9
    g.add_node("Have dinner with friends") #10

    g.add_edge("Party last night", "Going to bed late", directed=True)
    g.add_edge("Studying late", "Going to bed late", directed=True)
    g.add_edge("Alarm is set", "Wake up early", directed=True)
    g.add_edge("Going to bed late", "Wake up early", directed=True)

    g.add_edge("Wake up early", "Have breakfast", directed=True)
    g.add_edge("Wake up early", "Studying", directed=True)

    g.add_edge("Have breakfast", "Studying", directed=True)
    g.add_edge("Have breakfast", "Have lunch", directed=True)

    g.add_edge("Studying", "Have lunch", directed=True)
    g.add_edge("Studying", "Have dinner with friends", directed=True)
    g.add_edge("Studying", "Have dinner at home", directed=True)

    g.add_edge("Have lunch", "Have dinner at home", directed=True)
    g.add_edge("Have lunch", "Have dinner with friends", directed=True)

    g.add_edge("Have dinner at home", "Have dinner with friends", directed=True)

    g.print_nodes_id()
    g.print_matrix()
    g.load_CPT()

    distr, repet = g.ancestral_sampling(1000)
    print("Max probability encountered: ", distr[max(distr,key=distr.get)])
    print("With the combination: \n" + '\n'.join(map(str, max(distr,key=distr.get))))
    print("\n")
    print("Min probability encountered: ", distr[min(distr,key=distr.get)])
    print("With the combination: \n" + '\n'.join(map(str, min(distr,key=distr.get))))




if __name__ == '__main__':
    main()
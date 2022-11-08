import sys
from typing import Optional, Union, List
parentdir = '/home/jerzy/Documents/GitHub/GitHub/FEDOT'
bamtdir = '/home/jerzy/Documents/GitHub/GitHub/BAMT'
sys.path.insert(0, parentdir)
sys.path.insert(0, bamtdir)

from math import ceil
from pgmpy.estimators import K2Score
from pgmpy.models import BayesianNetwork
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.objective.objective_eval import ObjectiveEvaluate
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from examples.divided_bn import DividedBN
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
import bamt.Preprocessors as pp
import bamt.Networks as Nets
from sklearn import preprocessing
import random
import pandas as pd
from copy import deepcopy
from fedot.core.dag.graph import Graph
import time


class CustomGraphModel(Graph):

    def __init__(self, nodes: Optional[Union[OptNode, List[OptNode]]] = None):
        super().__init__(nodes)
        self.unique_pipeline_id = 1


class CustomGraphNode(OptNode):
    def __str__(self):
        return self.content["name"]


# задаем метрику
def custom_metric(graph: CustomGraphModel, data: pd.DataFrame):
    score = 0
    graph_nx, labels = graph_structure_as_nx_graph(graph)
    struct = []
    for pair in graph_nx.edges():
        l1 = str(labels[pair[0]])
        l2 = str(labels[pair[1]])
        struct.append([l1, l2])

    global local_edges

    bn_model = BayesianNetwork(struct)
    bn_model.add_nodes_from(data.columns)
    # bn_model.add_edges_from(local_edges)

    score = K2Score(data).score(bn_model)
    return [-score]

# задаем кроссовер (обмен ребрами)
def custom_crossover_exchange_edges(graph_first: OptGraph, graph_second: OptGraph, max_depth):
    def find_node(graph: OptGraph, node):
        return graph.nodes[dir_of_nodes[node.content['name']]]

    num_cros = 100
    try:
        for _ in range(num_cros):
            old_edges1 = []
            old_edges2 = []
            new_graph_first = deepcopy(graph_first)
            new_graph_second = deepcopy(graph_second)

            edges_1 = new_graph_first.operator.get_all_edges()
            edges_2 = new_graph_second.operator.get_all_edges()
            count = ceil(min(len(edges_1), len(edges_2))/2)
            choice_edges_1 = random.sample(edges_1, count)
            choice_edges_2 = random.sample(edges_2, count)

            for pair in choice_edges_1:
                new_graph_first.operator.disconnect_nodes(pair[0], pair[1], False)
            for pair in choice_edges_2:
                new_graph_second.operator.disconnect_nodes(pair[0], pair[1], False)

            old_edges1 = new_graph_first.operator.get_all_edges()
            old_edges2 = new_graph_second.operator.get_all_edges()

            new_edges_2 = [[find_node(new_graph_second, i[0]), find_node(new_graph_second, i[1])]
                           for i in choice_edges_1]
            new_edges_1 = [[find_node(new_graph_first, i[0]), find_node(new_graph_first, i[1])] for i in choice_edges_2]
            for pair in new_edges_1:
                if pair not in old_edges1:
                    new_graph_first.operator.connect_nodes(pair[0], pair[1])
            for pair in new_edges_2:
                if pair not in old_edges2:
                    new_graph_second.operator.connect_nodes(pair[0], pair[1])

            if old_edges1 != new_graph_first.operator.get_all_edges() or old_edges2 != new_graph_second.operator.get_all_edges():
                break

        if old_edges1 == new_graph_first.operator.get_all_edges() and new_edges_1 != [] and new_edges_1 != None:
            new_graph_first = deepcopy(graph_first)
        if old_edges2 == new_graph_second.operator.get_all_edges() and new_edges_2 != [] and new_edges_2 != None:
            new_graph_second = deepcopy(graph_second)
    except Exception as ex:
        print(ex)
    return new_graph_first, new_graph_second


# задаем три варианта мутации: добавление узла, удаление узла, разворот узла
def custom_mutation_add(graph: OptGraph, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            nodes_not_cycling = (random_node.descriptive_id not in
                                 [n.descriptive_id for n in other_random_node.ordered_subnodes_hierarchy()] and
                                 other_random_node.descriptive_id not in
                                 [n.descriptive_id for n in random_node.ordered_subnodes_hierarchy()])
            if nodes_not_cycling:
                graph.operator.connect_nodes(random_node, other_random_node)
                break

    except Exception as ex:
        graph.log.warn(f'Incorrect connection: {ex}')
    return graph


def custom_mutation_delete(graph: OptGraph, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                graph.operator.disconnect_nodes(other_random_node, random_node, False)
                break
    except Exception as ex:
        print(ex)
    return graph


def custom_mutation_reverse(graph: OptGraph, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                graph.operator.reverse_edge(other_random_node, random_node)
                break
    except Exception as ex:
        print(ex)
    return graph


# задаем правила на запрет дублирующих узлов
def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True

def run_example():

    data = pd.read_csv('examples/data/'+file+'.csv')
    if 'Unnamed: 0' in data.columns:
        data = data.drop(['Unnamed: 0'], axis=1, inplace=True)

    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)


    # initialize divided_bn

    divided_bn = DividedBN(data = data)

    divided_bn.set_local_structures(data, datatype="continuous")

    local_edges = divided_bn.local_structures_edges

    local_nodes = divided_bn.local_structures_nodes

    print('local nodes', local_nodes)

    print('Local edges:', local_edges)


    divided_bn.set_hidden_nodes(data = data)

    print('Hidden nodes:', divided_bn.hidden_nodes)

    bns_info = divided_bn.local_structures_info 

    hidden_df = pd.DataFrame.from_dict(divided_bn.hidden_nodes)

    print('Hidden df:', hidden_df)

    root_nodes = divided_bn.root_nodes
    child_nodes = divided_bn.child_nodes

    print('Root nodes:', root_nodes)
    print('Child nodes:', child_nodes)

    vertices = list(hidden_df.columns)

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, _ = p.apply(hidden_df)

    # словарь: {имя_узла: уникальный_номер_узла}
    global dir_of_nodes
    dir_of_nodes = {hidden_df.columns[i]: i for i in range(len(hidden_df.columns))}

    # правила для байесовских сетей: нет петель, нет циклов, нет повторяющихся узлов
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]

    # задаем для оптимизатора fitness-функцию
    objective = Objective(custom_metric)
    objective_eval = ObjectiveEvaluate(objective, data=discretized_data)
    # инициализация начальной сети (пустая)
    initial = [CustomGraphModel(nodes=[CustomGraphNode(nodes_from=None,
                                                       content={'name': vertex}) for vertex in vertices])]

    requirements = PipelineComposerRequirements(
        primary=vertices,
        secondary=vertices,
        max_arity=100,
        max_depth=100,
        pop_size=pop_size,
        num_of_generations=n_generation,
        crossover_prob=crossover_probability,
        mutation_prob=mutation_probability
    )

    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        selection_types=[SelectionTypesEnum.tournament],
        mutation_types=[custom_mutation_add, custom_mutation_delete, custom_mutation_reverse],
        crossover_types=[custom_crossover_exchange_edges]
    )

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
        rules_for_constraint=rules)

    optimiser = EvoGraphOptimiser(
        graph_generation_params=graph_generation_params,
        parameters=optimiser_parameters,
        requirements=requirements,
        initial_graph=initial,
        objective=objective)

    # запуск оптимизатора
    optimized_graph = optimiser.optimise(objective_eval)[0]
    # вывод полученного графа

    evolutionary_edges = optimized_graph.operator.get_all_edges()

    all_edges = local_edges

    optimized_graph.show()

    print("Evo edges:", evolutionary_edges)

    return all_edges


if __name__ == '__main__':

    # файл с исходными данными (должен лежать в 'examples/data/')
    file = 'arth150'
    # размер популяции
    pop_size = 10
    # количество поколений
    n_generation = 50
    # вероятность кроссовера
    crossover_probability = 0.8
    # вероятность мутации
    mutation_probability = 0.9
    
    start_time = time.time()

    run_example()

    print("--- %s seconds ---" % (time.time() - start_time))

from typing import Optional, Union, Sequence

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.parameters.mutation_prob import AdaptiveMutationProb
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.pipelines.pipeline import Pipeline


class EvoGraphParameterFreeOptimiser(EvoGraphOptimiser):
    """
    Implementation of the parameter-free adaptive evolutionary optimiser
    (population size and genetic operators rates is changing over time).
    For details, see https://ieeexplore.ieee.org/document/9504773
    """

    def __init__(self,
                 objective: Objective,
                 initial_graph: Union[Pipeline, Sequence[Pipeline]],
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 parameters: Optional[GPGraphOptimiserParameters] = None,
                 log: Optional[Log] = None):
        super().__init__(objective, initial_graph, requirements, graph_generation_params, parameters, log)

        self._min_population_size_with_elitism = 7
        if self.parameters.genetic_scheme_type != GeneticSchemeTypesEnum.parameter_free:
            self.log.warn(f'Invalid genetic scheme type was changed to parameter-free. Continue.')
            self.parameters.genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free

        # Define adaptive parameters
        self._mutation_rate = AdaptiveMutationProb()

    def _operators_prob_update(self):
        if not self.generations.is_any_improved:
            mutation_prob = self._mutation_rate.next(self.population)
            crossover_prob = 1. - mutation_prob
            self.requirements.mutation_prob = mutation_prob
            self.requirements.crossover_prob = crossover_prob

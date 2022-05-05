import csv
import itertools
import json
import os
import shutil
from copy import deepcopy
from typing import Any, List, Optional, Union, Sequence

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.serializers import Serializer
from fedot.core.validation.objective import Objective
from fedot.core.visualisation.opt_viz import PipelineEvolutionVisualiser

from fedot.core.optimisers.utils.population_utils import get_metric_position
from fedot.core.repository.quality_metrics_repository import QualityMetricsEnum
from fedot.core.utils import default_fedot_data_dir


class OptHistory:
    """
    Contain history, convert Pipeline to PipelineTemplate, save history to csv
    """

    def __init__(self, objective: Objective = None, save_folder: Optional[str] = None):
        self._objective = objective or Objective([])
        self.individuals: List[List[Individual]] = []
        self.archive_history: List[List[Individual]] = []
        self.save_folder: Optional[str] = save_folder

    def add_to_history(self, individuals: List[Individual]):
        new_inds = deepcopy(individuals)
        self.individuals.append(new_inds)

    def add_to_archive_history(self, individuals: List[Individual]):
        new_inds = deepcopy(individuals)
        self.archive_history.append(new_inds)

    def write_composer_history_to_csv(self, file='history.csv'):
        history_dir = self._get_save_path()
        file = os.path.join(history_dir, file)
        if not os.path.isdir(history_dir):
            os.mkdir(history_dir)
        self._write_header_to_csv(file)
        idx = 0
        adapter = PipelineAdapter()
        for gen_num, gen_inds in enumerate(self.individuals):
            for ind_num, ind in enumerate(gen_inds):
                ind_pipeline_template = adapter.restore_as_template(ind.graph, ind.metadata)
                row = [
                    idx, gen_num, ind.fitness.values,
                    len(ind_pipeline_template.operation_templates), ind_pipeline_template.depth, ind.metadata
                ]
                self._add_history_to_csv(file, row)
                idx += 1

    def _write_header_to_csv(self, f):
        with open(f, 'w', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            metric_str = 'metric'
            if self._objective.is_multi_objective:
                metric_str += 's'
            row = ['index', 'generation', metric_str, 'quantity_of_operations', 'depth', 'metadata']
            writer.writerow(row)

    @staticmethod
    def _add_history_to_csv(f: str, row: List[Any]):
        with open(f, 'a', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(row)

    def save_current_results(self, path: Optional[str] = None):
        if not path:
            path = self._get_save_path()
        if path is not None:
            try:
                last_gen_id = len(self.individuals) - 1
                last_gen = self.individuals[last_gen_id]
                last_gen_history = self.historical_fitness[last_gen_id]
                for individual, ind_fitness in zip(last_gen, last_gen_history):
                    ind_path = os.path.join(path, str(last_gen_id), str(individual.uid))
                    additional_info = \
                        {'fitness_name': self._objective.metric_names,
                         'fitness_value': ind_fitness}
                    PipelineAdapter().restore_as_template(
                        individual.graph, individual.metadata
                    ).export_pipeline(path=ind_path, additional_info=additional_info, datetime_in_path=False)
            except Exception as ex:
                print(ex)

    def save(self, json_file_path: os.PathLike = None) -> Optional[str]:
        if json_file_path is None:
            return json.dumps(self, indent=4, cls=Serializer)
        with open(json_file_path, mode='w') as json_fp:
            json.dump(self, json_fp, indent=4, cls=Serializer)

    @staticmethod
    def load(json_str_or_file_path: Union[str, os.PathLike] = None) -> 'OptHistory':
        try:
            return json.loads(json_str_or_file_path, cls=Serializer)
        except json.JSONDecodeError as exc:
            with open(json_str_or_file_path, mode='r') as json_fp:
                return json.load(json_fp, cls=Serializer)

    def clean_results(self, path: Optional[str] = None):
        if not path and self.save_folder is not None:
            path = os.path.join(default_fedot_data_dir(), self.save_folder)
        if path is not None:
            shutil.rmtree(path, ignore_errors=True)
            os.mkdir(path)

    def show(self, save_path_to_file: str = None):
        """ Visualizes fitness values across generations """

        if self.all_historical_fitness is None:
            return
        viz = PipelineEvolutionVisualiser()
        viz.visualise_fitness_by_generations(self, save_path_to_file=save_path_to_file)

    @property
    def historical_fitness(self) -> Sequence[Sequence[Union[float, Sequence[float]]]]:
        """Return sequence of histories of generations per each metric"""
        if self._objective.is_multi_objective:
            historical_fitness = []
            num_metrics = len(self._objective.metrics)
            for objective_num in range(num_metrics):
                # history of specific objective for each generation
                objective_history = [[ind.fitness.values[objective_num] for ind in generation]
                                     for generation in self.individuals]
                historical_fitness.append(objective_history)
        else:
            historical_fitness = [[pipeline.fitness.value for pipeline in pop] for pop in self.individuals]
        return historical_fitness

    @property
    def all_historical_fitness(self):
        historical_fitness = self.historical_fitness
        if self._objective.is_multi_objective:
            all_historical_fitness = []
            for obj_num in range(len(historical_fitness)):
                all_historical_fitness.append(list(itertools.chain(*historical_fitness[obj_num])))
        else:
            all_historical_fitness = list(itertools.chain(*historical_fitness))
        return all_historical_fitness

    @property
    def all_historical_quality(self):
        if self._objective.is_multi_objective:
            metric_position = get_metric_position(self._objective.metrics, QualityMetricsEnum)
            all_historical_quality = self.all_historical_fitness[metric_position]
        else:
            all_historical_quality = self.all_historical_fitness
        return all_historical_quality

    @property
    def historical_pipelines(self):
        adapter = PipelineAdapter()
        return [
            adapter.restore_as_template(ind.graph, ind.metadata)
            for ind in list(itertools.chain(*self.individuals))
        ]

    def _get_save_path(self):
        if self.save_folder is not None:
            if os.path.sep in self.save_folder:
                # Defined path is full - there is no need to use default dir
                # Create folder if it's not exists
                if os.path.isdir(self.save_folder) is False:
                    os.makedirs(self.save_folder)
                return self.save_folder
            else:
                return os.path.join(default_fedot_data_dir(), self.save_folder)
        return None

from typing import Union, Iterable

from golem.core.optimisers.objective import Objective
from golem.utilities.data_structures import ensure_wrapped_in_sequence

from fedot.core.repository.quality_metrics_repository import MetricType, MetricsRepository, ComplexityMetricsEnum


class MetricsObjective(Objective):
    def __init__(self,
                 metrics: Union[MetricType, Iterable[MetricType]],
                 is_multi_objective: bool = False):
        quality_metrics = {}
        complexity_metrics = {}

        for metric in ensure_wrapped_in_sequence(metrics):
            if callable(metric):
                metric_id = str(metric)
                quality_metrics[metric_id] = metric
            else:
                metric_func = MetricsRepository.metric_by_id(metric)
                if metric_func:
                    if ComplexityMetricsEnum.has_value(metric):
                        complexity_metrics[metric] = metric_func
                    else:
                        quality_metrics[metric] = metric_func
                else:
                    raise ValueError(f'Incorrect metric {metric}')

        super().__init__(quality_metrics, complexity_metrics, is_multi_objective)

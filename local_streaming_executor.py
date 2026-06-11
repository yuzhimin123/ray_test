"""
Local streaming executor for debugging.

Provides a synchronous version of StreamingExecutor that runs in the
calling thread, making it easy to debug and step through scheduling logic.

Usage:
    import ray
    from ray.data._internal.execution.interfaces import ExecutionOptions
    from ray.data.context import DataContext
    from local_streaming_executor import LocalStreamingExecutor

    ray.init()
    ctx = DataContext.get_current()
    executor = LocalStreamingExecutor(ctx)
    output = executor.execute(dag)
    for bundle in output:
        ...
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

from ray.data._internal.execution import create_ranker
from ray.data._internal.execution.interfaces import (
    Executor,
    ExecutionOptions,
    ExecutionResources,
    OutputIterator,
    PhysicalOperator,
    RefBundle,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.resource_manager import ResourceManager
from ray.data._internal.execution.streaming_executor_state import (
    OpState,
    Topology,
    build_streaming_topology,
    format_op_state_summary,
    process_completed_tasks,
    select_operator_to_run,
    update_operator_states,
)
from ray.data._internal.stats import DatasetStats
from ray.data.context import DataContext

logger = logging.getLogger(__name__)


class LocalOutputIterator(OutputIterator):
    """Simple output iterator over a pre-collected list of bundles."""

    def __init__(self, bundles: List[RefBundle]):
        self._bundles = bundles
        self._index = 0

    def get_next(self, output_split_idx: Optional[int] = None) -> RefBundle:
        if self._index >= len(self._bundles):
            raise StopIteration()
        bundle = self._bundles[self._index]
        self._index += 1
        return bundle


class LocalStreamingExecutor(Executor):
    """Synchronous streaming executor for local debugging.

    Mirrors the scheduling algorithm of ``StreamingExecutor`` but runs
    entirely in the calling thread.  This makes it easy to set breakpoints,
    inspect topology state between scheduling steps, and debug without
    threading complexity.

    Key simplifications vs ``StreamingExecutor``:
      - Single-threaded scheduling loop (no ``Thread``)
      - No backpressure policies
      - No cluster or actor autoscalers
      - No execution callbacks
      - No progress manager
      - No stats actor or metrics gauges
      - Infinite resource limits
      - Output collected as a list (not streaming)
    """

    def __init__(self, data_context: DataContext):
        super().__init__(data_context.execution_options)
        self._data_context = data_context
        self._start_time: Optional[float] = None
        self._initial_stats: Optional[DatasetStats] = None
        self._final_stats: Optional[DatasetStats] = None
        self._topology: Optional[Topology] = None
        self._output_node: Optional[PhysicalOperator] = None
        self._resource_manager: Optional[ResourceManager] = None
        self._ranker = create_ranker()
        self._output_bundles: List[RefBundle] = []
        self._shutdown_flag: bool = False

    def execute(
        self,
        dag: PhysicalOperator,
        initial_stats: Optional[DatasetStats] = None,
        callbacks: Optional[List] = None,
    ) -> OutputIterator:
        """Execute the DAG synchronously and return all outputs."""
        self._initial_stats = initial_stats
        self._start_time = time.perf_counter()

        logger.info("LocalStreamingExecutor: building topology ...")
        self._topology = build_streaming_topology(dag, self._options)
        self._output_node = dag

        self._resource_manager = ResourceManager(
            self._topology,
            self._options,
            lambda: ExecutionResources.inf(),
            self._data_context,
        )

        logger.info("LocalStreamingExecutor: starting scheduling loop ...")
        self._scheduling_loop()

        logger.info("LocalStreamingExecutor: collecting outputs ...")
        bundles = self._collect_outputs()
        logger.info(
            "LocalStreamingExecutor: done, collected %d bundles", len(bundles)
        )

        return LocalOutputIterator(bundles)

    # ------------------------------------------------------------------
    # Scheduling loop (synchronous)
    # ------------------------------------------------------------------

    def _scheduling_loop(self) -> None:
        """Run the scheduling loop to completion."""
        while not self._shutdown_flag:
            continue_sched = self._scheduling_loop_step()
            if not continue_sched:
                break

    def _scheduling_loop_step(self) -> bool:
        """One iteration of the scheduling loop.

        Phases (mirroring ``StreamingExecutor._scheduling_loop_step``):
          1. Process completed tasks (``ray.wait``)
          2. Select & dispatch operators
          3. Update operator states (inputs_done / all_inputs_done / finished)
          4. Check completion
        """
        topology = self._topology
        self._resource_manager.update_usages()

        # No backpressure – read all available task outputs each step.
        backpressure_policies: List = []

        # --- Phase 1: process completed tasks ---
        num_errored = process_completed_tasks(
            topology,
            backpressure_policies,
            max_errored_blocks=-1,
        )
        if num_errored:
            logger.warning(
                "LocalStreamingExecutor: %d errored block(s) ignored", num_errored
            )

        self._resource_manager.update_usages()

        # --- Phase 2: dispatch operators ---
        while True:
            op = select_operator_to_run(
                topology,
                self._resource_manager,
                backpressure_policies,
                ensure_liveness=True,
                ranker=self._ranker,
            )
            if op is None:
                break
            topology[op].dispatch_next_task()
            self._resource_manager.update_usages()

        # --- Phase 3: update operator states ---
        update_operator_states(topology)

        # --- Phase 4: check completion ---
        all_completed = all(op.has_completed() for op in topology)
        if all_completed:
            return False

        # If no op can make progress (no active tasks, no pending bundles),
        # drain remaining outputs from the output node and stop.
        if self._is_stuck(topology):
            self._flush_outputs()
            return False

        return True

    # ------------------------------------------------------------------
    # Output collection
    # ------------------------------------------------------------------

    def _flush_outputs(self) -> None:
        """Flush all output-node bundles from OpState queues into
        ``_output_bundles``.  Called when the scheduling loop is stuck
        (i.e. no operator can make further progress)."""
        self._drain_queue(self._output_node)

    def _drain_queue(self, op: PhysicalOperator) -> None:
        """Drain remaining outputs from *op* (and all its transitively
        reachable operators) into ``_output_bundles``."""
        state = self._topology[op]
        # First flush upstream outputs so they are available as inputs.
        for dep in op.input_dependencies:
            dep_state = self._topology.get(dep)
            if dep_state and dep_state.output_queue.has_next():
                self._drain_queue(dep)
        # Now pull the operator's own output and forward to downstream.
        while state.output_queue.has_next():
            bundle = state.output_queue.pop()
            if bundle is not None:
                self._output_bundles.append(bundle)

    def _collect_outputs(self) -> List[RefBundle]:
        """Return all bundles collected during execution.

        By the time this is called the scheduling loop has terminated and
        the ``_output_bundles`` list should already be populated.
        """
        if self._output_bundles:
            return self._output_bundles
        # Fallback: drain the output node directly.
        state = self._topology[self._output_node]
        while state.output_queue.has_next():
            bundle = state.output_queue.pop()
            if bundle is not None:
                self._output_bundles.append(bundle)
        return self._output_bundles

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_stuck(topology: Topology) -> bool:
        """Return True when no operator can make any further progress.

        Progress can happen in three ways:
          1. An operator has active (running) tasks.
          2. An operator has input bundles waiting to be dispatched.
          3. An operator has output that hasn't been pulled into the OpState
             queue yet (``has_next()``).
        When all three are False, the scheduling loop is stuck.
        """
        any_active = any(op.num_active_tasks() > 0 for op in topology)
        any_pending = any(state.has_pending_bundles() for state in topology.values())
        any_output = any(op.has_next() for op in topology)
        return not any_active and not any_pending and not any_output

    def _dump_topology_state(self) -> None:
        """Log current state of every operator for debugging."""
        for i, (op, state) in enumerate(self._topology.items()):
            summary = format_op_state_summary(state, self._resource_manager)
            logger.debug(
                "  [%d] %s  completed=%s  finished=%s  active_tasks=%d  %s",
                i,
                op.name,
                op.has_completed(),
                op.has_execution_finished(),
                op.num_active_tasks(),
                summary,
            )

    def get_topology(self) -> Topology:
        """Return the current topology (useful for test inspection)."""
        return self._topology

    # ------------------------------------------------------------------
    # Executor interface
    # ------------------------------------------------------------------

    def shutdown(self, force: bool = False, exception: Optional[Exception] = None):
        """Shut down all operators and release resources."""
        self._shutdown_flag = True
        if self._topology:
            from ray.data._internal.stats import Timer

            timer = Timer()
            for op in self._topology:
                op.shutdown(timer, force=force)

    def get_stats(self) -> DatasetStats:
        if self._final_stats:
            return self._final_stats
        stats = self._initial_stats or DatasetStats(metadata={}, parent=None)
        for op in self._topology or {}:
            if isinstance(op, InputDataBuffer):
                continue
            builder = stats.child_builder(
                op.name, override_start_time=self._start_time
            )
            stats = builder.build_multioperator(op.get_stats())
        return stats

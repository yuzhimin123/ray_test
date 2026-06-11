"""
Demonstration / test for LocalStreamingExecutor.

Builds a small operator DAG (InputDataBuffer → LimitOperator) and
executes it with the local synchronous executor, verifying outputs.
"""

import logging
import sys

import ray
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.limit_operator import LimitOperator
from ray.data.block import BlockMetadata
from ray.data.context import DataContext

from local_streaming_executor import LocalStreamingExecutor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def make_ref_bundle(data) -> RefBundle:
    """Create a RefBundle containing a single block with the given data.

    Args:
        data: A list of dicts (rows) to store in the block.

    Returns:
        A RefBundle wrapping the data as an ObjectRef.
    """
    import pandas as pd

    df = pd.DataFrame(data)
    block = df.to_dict("records") if hasattr(df, "to_dict") else df
    block_ref = ray.put(df)

    num_rows = len(df)
    size_bytes = len(df) * sum(len(str(v)) for row in data for v in row.values()) or 1
    meta = BlockMetadata(
        num_rows=num_rows,
        size_bytes=size_bytes,
        input_files=None,
        exec_stats=None,
    )
    return RefBundle(
        blocks=[(block_ref, meta)],
        owns_blocks=True,
        schema=None,
    )


def test_single_source():
    """Simple test: InputDataBuffer with two bundles -> LimitOperator.

    Verifies:
      - The scheduling loop terminates
      - The correct number of output bundles is produced
      - Bundles contain the expected data
    """
    users = [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 25},
    ]
    more_users = [
        {"id": 3, "name": "Charlie", "age": 35},
        {"id": 4, "name": "Diana", "age": 28},
    ]

    bundles = [make_ref_bundle(users), make_ref_bundle(more_users)]

    ctx = DataContext.get_current()
    source = InputDataBuffer(ctx, input_data=bundles)
    limit_op = LimitOperator(limit=3, input_op=source, data_context=ctx)

    executor = LocalStreamingExecutor(ctx)
    with executor:
        output = executor.execute(limit_op)
        results = list(output)

    logger.info("Got %d output bundles", len(results))
    for i, b in enumerate(results):
        logger.info("  bundle[%d]: %d blocks, %d rows", i, len(b), b.num_rows())

    assert len(results) == 2, f"Expected 2 output bundles, got {len(results)}"

    total_rows = sum(b.num_rows() or 0 for b in results)
    assert total_rows == 3, f"Expected 3 rows total (limit=3), got {total_rows}"
    logger.info("PASSED: test_single_source")


def test_multi_operator_chain():
    """Chain: InputDataBuffer → LimitOperator (no extra work).

    Exercises the scheduling loop with multiple operators.
    """
    data = [{"x": i} for i in range(10)]
    bundle = make_ref_bundle(data)

    ctx = DataContext.get_current()
    source = InputDataBuffer(ctx, input_data=[bundle])
    limit_op = LimitOperator(limit=5, input_op=source, data_context=ctx)

    executor = LocalStreamingExecutor(ctx)
    with executor:
        output = executor.execute(limit_op)
        results = list(output)

    total_rows = sum(b.num_rows() or 0 for b in results)
    assert total_rows == 5, f"Expected 5 rows (limit=5), got {total_rows}"
    logger.info("PASSED: test_multi_operator_chain")


def test_empty_input():
    """Edge case: InputDataBuffer with no data."""
    ctx = DataContext.get_current()
    source = InputDataBuffer(ctx, input_data=[])
    limit_op = LimitOperator(limit=10, input_op=source, data_context=ctx)

    executor = LocalStreamingExecutor(ctx)
    with executor:
        output = executor.execute(limit_op)
        results = list(output)

    assert len(results) == 0, f"Expected 0 bundles, got {len(results)}"
    logger.info("PASSED: test_empty_input")


def test_inspect_topology():
    """Demonstrate using ``get_topology()`` to inspect execution state."""
    data = [{"v": i} for i in range(5)]
    bundle = make_ref_bundle(data)

    ctx = DataContext.get_current()
    source = InputDataBuffer(ctx, input_data=[bundle])
    limit_op = LimitOperator(limit=3, input_op=source, data_context=ctx)

    executor = LocalStreamingExecutor(ctx)
    with executor:
        output = executor.execute(limit_op)
        _ = list(output)
        topo = executor.get_topology()
        logger.info("Topology has %d operators", len(topo))
        for op, state in topo.items():
            logger.info(
                "  op=%s  completed=%s  finished=%s  outputs=%d",
                op.name,
                op.has_completed(),
                op.has_execution_finished(),
                state.num_completed_tasks,
            )

    logger.info("PASSED: test_inspect_topology")


if __name__ == "__main__":
    ray.init(log_to_driver=False)
    try:
        test_single_source()
        test_multi_operator_chain()
        test_empty_input()
        test_inspect_topology()
        logger.info("All tests PASSED")
    finally:
        ray.shutdown()

"""
Custom in-memory async queue for background image processing.

Why in-memory instead of Redis/RabbitMQ/SQS: the assignment explicitly
allows this ("in-memory queue" is listed as an acceptable option), and it
keeps the whole project runnable with `pip install` + `uvicorn` and no extra
infrastructure. The trade-off — queued jobs are lost on a hard process kill
(not a graceful shutdown, and not a crash after the DB row already exists) —
is mitigated, not eliminated, by two things documented in the README:
  1. On startup we re-enqueue every image still in `pending`/`processing`
     status (see `recover_incomplete_jobs`), so a restart resumes work
     rather than leaving jobs stuck forever.
  2. Job status lives in SQLite, not in the queue, so clients can always
     poll status regardless of what the in-memory queue is doing.

For real production scale (many workers/processes, need for durability
across a full machine failure, replay, dead-letter queues) this module is
exactly the seam where you'd swap in SQS/RabbitMQ/BullMQ — the public
functions here (`enqueue`, `start_workers`) are the only integration point.
"""
import asyncio
import logging

from app import repository
from app.config import MAX_PROCESSING_RETRIES, QUEUE_WORKER_COUNT
from app.pipeline import process_image

logger = logging.getLogger("queue_worker")

_queue: asyncio.Queue[str] = asyncio.Queue()
_worker_tasks: list[asyncio.Task] = []


async def enqueue(image_id: str) -> None:
    await _queue.put(image_id)


async def _worker_loop(worker_index: int) -> None:
    logger.info("worker %d started", worker_index)
    while True:
        image_id = await _queue.get()
        try:
            await asyncio.to_thread(_process_with_retry, image_id)
        except Exception:
            logger.exception("worker %d: unhandled error processing image_id=%s", worker_index, image_id)
        finally:
            _queue.task_done()


def _process_with_retry(image_id: str) -> None:
    """Runs in a worker thread. Retries transient failures a bounded number
    of times before giving up and marking the job `failed`."""
    attempt = 0
    while True:
        try:
            process_image(image_id)
            return
        except Exception as exc:  # noqa: BLE001
            attempt = repository.increment_retry_count(image_id)
            logger.warning("processing image_id=%s failed (attempt %d): %s", image_id, attempt, exc)
            if attempt >= MAX_PROCESSING_RETRIES:
                repository.update_status(image_id, "failed", error_message=f"exhausted retries: {exc}")
                return


async def start_workers(worker_count: int = QUEUE_WORKER_COUNT) -> None:
    for i in range(worker_count):
        _worker_tasks.append(asyncio.create_task(_worker_loop(i)))
    await recover_incomplete_jobs()


async def stop_workers() -> None:
    for task in _worker_tasks:
        task.cancel()
    await asyncio.gather(*_worker_tasks, return_exceptions=True)
    _worker_tasks.clear()


async def recover_incomplete_jobs() -> None:
    """On startup, re-queue anything left in pending/processing from a
    previous run that never reached completed/failed (e.g. process was
    killed mid-job). This is a simple at-least-once recovery strategy —
    a job could theoretically be double-processed if it crashed *after*
    writing results but *before* the status update, which is an accepted
    trade-off documented in the README."""
    incomplete_ids = repository.get_incomplete_image_ids()
    for image_id in incomplete_ids:
        logger.info("recovering incomplete job image_id=%s", image_id)
        await enqueue(image_id)

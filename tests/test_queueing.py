from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.queueing import JobQueue
from app.types import JobRecord


@pytest.mark.unit
class TestJobQueue:
    @pytest.fixture
    def model_registry(self):
        registry = MagicMock()
        registry.is_model_known.return_value = True
        return registry

    @pytest.fixture
    def job_queue(self, model_registry):
        return JobQueue(model_registry=model_registry)

    @pytest.mark.asyncio
    async def test_enqueue_adds_job_to_queue(
        self, job_queue: JobQueue, sample_job_record: JobRecord
    ):
        await job_queue.enqueue(
            job_id=sample_job_record.job_id,
            model=sample_job_record.model,
            language=sample_job_record.language,
            callback_url=sample_job_record.callback_url,
            file_dir=sample_job_record.file_dir,
        )

        assert sample_job_record.job_id in job_queue._jobs
        assert job_queue._q.qsize() == 1

    @pytest.mark.asyncio
    async def test_start_workers_creates_correct_count(
        self, job_queue: JobQueue, mock_queueing_gpu
    ):
        await job_queue.start_workers()
        assert len(job_queue._workers) == 2
        await job_queue.stop_workers()

    @pytest.mark.asyncio
    async def test_distribution_across_gpus(
        self, job_queue: JobQueue, mock_queueing_gpu
    ):
        await job_queue.start_workers()
        await job_queue.enqueue(
            job_id="job1", model="v1_ctc", language="ru",
            callback_url="http://cb", file_dir="/tmp/j1"
        )
        await job_queue.enqueue(
            job_id="job2", model="v2_rnnt", language="ru",
            callback_url="http://cb", file_dir="/tmp/j2"
        )
        await job_queue.enqueue(
            job_id="job3", model="v3_conformer", language="ru",
            callback_url="http://cb", file_dir="/tmp/j3"
        )

        await asyncio.sleep(0.1)

        running = [s for s in job_queue._gpu_running.values()]
        assert len(running) <= 2

        await job_queue.stop_workers()

    @pytest.mark.asyncio
    async def test_worker_handles_transcribe_error(
        self, job_queue: JobQueue, mock_queueing_gpu
    ):
        job_queue._models.is_model_known.return_value = True

        await job_queue.start_workers()
        
        with patch("app.queueing.preprocess_to_wav", side_effect=RuntimeError("preprocess failed")):
            await job_queue.enqueue(
                job_id="job_error",
                model="v1_ctc",
                language="ru",
                callback_url="http://cb",
                file_dir="/tmp/job_error",
            )

            await asyncio.sleep(0.2)

        job = job_queue.get_job("job_error")
        assert job.status == "failed"
        assert job.error is not None
        
        await job_queue.stop_workers()

    @pytest.mark.asyncio
    async def test_callback_on_success(
        self, job_queue: JobQueue, sample_job_record: JobRecord, mock_httpx
    ):
        sample_job_record.status = "completed"
        sample_job_record.result = {"text": "привет", "segments": []}
        job_queue._jobs[sample_job_record.job_id] = sample_job_record

        await job_queue._deliver_callback_and_cleanup(sample_job_record)

        mock_httpx.return_value.__aenter__.return_value.post.assert_called_once()
        assert sample_job_record.callback_delivered_at_ms is not None

    @pytest.mark.asyncio
    async def test_callback_error_keeps_job_completed(
        self, job_queue: JobQueue, sample_job_record: JobRecord, mock_httpx
    ):
        sample_job_record.status = "completed"
        sample_job_record.result = {"text": "привет", "segments": []}
        job_queue._jobs[sample_job_record.job_id] = sample_job_record

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client.post.return_value = mock_response
        mock_httpx.return_value.__aenter__.return_value = mock_client

        await job_queue._deliver_callback_and_cleanup(sample_job_record)

        assert sample_job_record.status == "completed"
        assert sample_job_record.callback_error is not None

    @pytest.mark.asyncio
    async def test_cleanup_removes_job_directory(
        self, job_queue: JobQueue, sample_job_record: JobRecord, tmp_path: Path, mock_httpx
    ):
        sample_job_record.file_dir = str(tmp_path / "job_dir")
        job_dir = Path(sample_job_record.file_dir)
        job_dir.mkdir()
        (job_dir / "file.txt").write_text("test")

        sample_job_record.status = "completed"
        sample_job_record.result = {"text": "test", "segments": []}
        job_queue._jobs[sample_job_record.job_id] = sample_job_record

        with patch("shutil.rmtree") as mock_rmtree:
            await job_queue._deliver_callback_and_cleanup(sample_job_record)
            mock_rmtree.assert_called_once()

    @pytest.mark.asyncio
    async def test_serialize_job_computes_timings(
        self, job_queue: JobQueue, sample_job_record: JobRecord
    ):
        sample_job_record.started_at_ms = 1000000
        sample_job_record.finished_at_ms = 1005000
        job_queue._jobs[sample_job_record.job_id] = sample_job_record

        serialized = job_queue.serialize_job(sample_job_record)

        assert serialized["queue_time_s"] == 0.0
        assert serialized["processing_time_s"] == 5.0

    @pytest.mark.asyncio
    async def test_snapshot_ids_returns_correct_lists(
        self, job_queue: JobQueue, sample_job_record: JobRecord
    ):
        job_queue._jobs = {
            "job1": JobRecord(
                job_id="job1", status="queued", model="v1_ctc", language="ru",
                callback_url="cb", created_at_ms=1, started_at_ms=None,
                finished_at_ms=None, result=None, error=None,
                callback_delivered_at_ms=None, callback_error=None, file_dir="/tmp/j1"
            ),
            "job2": JobRecord(
                job_id="job2", status="running", model="v2_rnnt", language="ru",
                callback_url="cb", created_at_ms=1, started_at_ms=2,
                finished_at_ms=None, result=None, error=None,
                callback_delivered_at_ms=None, callback_error=None, file_dir="/tmp/j2"
            ),
            "job3": JobRecord(
                job_id="job3", status="completed", model="v3_conformer", language="ru",
                callback_url="cb", created_at_ms=1, started_at_ms=2,
                finished_at_ms=3, result={}, error=None,
                callback_delivered_at_ms=4, callback_error=None, file_dir="/tmp/j3"
            ),
        }

        queued, running = job_queue.snapshot_ids()
        assert queued == ["job1"]
        assert running == ["job2"]

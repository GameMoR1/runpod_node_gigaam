# Документация по тестам проекта GigaAM

## Обзор

В данном документе описывается структура и содержание тестов для проекта GigaAM. Тесты организованы в директории `tests/` и покрывают различные компоненты системы.

## Структура тестов

```
tests/
├── __init__.py
├── conftest.py              # Общие фикстуры для всех тестов
├── test_e2e.py              # End-to-end тесты API
├── test_ffmpeg_proc.py      # Тесты предобработки аудио
├── test_gigaam_runner.py    # Тесты запуска транскрипции на GPU
├── test_gpu.py              # Тесты утилит мониторинга GPU
├── test_model_registry.py   # Тесты реестра моделей
├── test_postprocess.py      # Тесты постобработки текста
├── test_queueing.py         # Тесты очереди задач
└── TEST_DOCUMENTATION.md    # Данный файл
```

## Подробное описание тестовых файлов

### conftest.py

Файл содержит общие pytest фикстуры, используемые в различных тестовых модулях:

- **event_loop** - Фикстура для создания и управления event loop в асинхронных тестах
- **mock_db_fetch** - Мок для имитации запросов к базе данных
- **mock_db_models** - Пример данных моделей из базы данных
- **mock_enabled_ids** - Пример включенных моделей
- **sample_model_states** - Пример состояний моделей для тестирования
- **sample_job_record** - Пример записи о задаче
- **mock_gigaam_assets** - Моки для активов GigaAM (is_gigaam_model, preload_tokenizer, gigaam_cache_dir)
- **mock_gigaam_load** - Мок для загрузки модели GigaAM
- **mock_ffmpeg** - Мок для имитации вызовов ffmpeg
- **temp_job_dir** - Временная директория для задачи с тестовым аудиофайлом
- **sample_wav_file** - Тестовый WAV-файл
- **mock_gpu_metrics** - Мок для метрик GPU
- **mock_gpu_count** - Мок для количества GPU
- **mock_queueing_gpu** - Моки для очереди и GPU
- **mock_torch** - Мок для PyTorch
- **mock_httpx** - Мок для HTTP клиента
- **mock_wave** - Мок для работы с WAV-файлами
- **mock_time** - Мок для функции времени
- **mock_fetch_hugging_face_token** - Мок для получения токена Hugging Face

### test_e2e.py

End-to-end тесты для проверки работы API endpoints:

- **TestE2E** - Основной класс тестов
  - **test_health_endpoint** - Проверка эндпоинта /health
  - **test_queue_endpoint_empty** - Проверка эндпоинта /queue при пустой очереди
  - **test_status_endpoint_job_not_found** - Проверка ответа 404 для несуществующей задачи
  - **test_transcribe_endpoint_requires_valid_model** - Проверка валидации модели при транскрипции
  - **test_transcribe_endpoint_requires_callback_url** - Проверка обязательного параметра callback_url
  - **test_transcribe_endpoint_requires_file** - Проверка обязательного параметра файла
  - **test_dashboard_state_endpoint** - Проверка эндпоинта /dashboard/state
  - **test_dashboard_html_returns_page** - Проверка главной страницы
  - **test_full_transcribe_flow_success** - Полный поток успешной транскрипции
  - **test_transcribe_with_invalid_model** - Транскрипция с невалидной моделью
  - **test_websocket_dashboard** - Проверка WebSocket соединения для дашборда
  - **test_error_handling_file_upload_failure** - Обработка ошибок загрузки файла
  - **test_error_handling_callback_unavailable** - Обработка недоступности callback URL
  - **test_parallel_jobs_on_different_gpus** - Проверка параллельной обработки на разных GPU
  - **test_service_restart_with_unloaded_models** - Проверка состояния сервиса при незагруженных моделях
  - **test_all_endpoints_validated** - Валидация всех основных эндпоинтов

### test_ffmpeg_proc.py

Тесты для модуля предобработки аудио (`app.ffmpeg_proc`):

- **TestFfmpegProc** - Класс тестов
  - **test_successful_preprocess_creates_wav_with_correct_params** - Проверка успешного создания WAV с правильными параметрами
  - **test_preprocess_fails_on_ffmpeg_error** - Проверка обработки ошибок ffmpeg
  - **test_preprocess_with_invalid_input_raises_error** - Проверка обработки невалидного входного файла

### test_gigaam_runner.py

Тесты для модуля запуска транскрипции на GPU (`app.gigaam_runner`):

- **TestGigaamRunner** - Класс тестов
  - **test_short_audio_calls_transcribe** - Проверка вызова transcribe для короткого аудио
  - **test_long_audio_calls_transcribe_longform** - Проверка вызова transcribe_longform для длинного аудио
  - **test_long_audio_fallback_to_chunking** - Проверка fallback на chunking при отсутствии transcribe_longform
  - **test_gpu_metrics_included** - Проверка включения метрик GPU в результат
  - **test_postprocess_applied** - Проверка применения постобработки
  - **test_postprocess_filters_triplet_repeats** - Проверка фильтрации триплетных повторов
  - **test_token_count_calculated** - Проверка подсчета количества токенов
  - **test_result_structure** - Проверка структуры результата транскрипции

### test_gpu.py

Тесты для утилит мониторинга GPU (`app.gpu`):

- **TestGpu** - Класс тестов
  - **test_gpu_count_with_nvml** - Проверка получения количества GPU через NVML
  - **test_gpu_count_fallback_to_torch** - Проверка fallback на PyTorch для подсчета GPU
  - **test_gpu_count_returns_zero_when_no_gpu** - Проверка возврата 0 при отсутствии GPU
  - **test_torch_cuda_available** - Проверка доступности CUDA через PyTorch
  - **test_torch_cuda_device_count** - Проверка получения количества устройств CUDA
  - **test_gpu_name_with_torch** - Проверка получения имени GPU через PyTorch
  - **test_gpu_name_fallback** - Проверка fallback имени GPU
  - **test_gpu_metrics_with_nvml** - Проверка получения метрик GPU через NVML
  - **test_gpu_metrics_fallback_to_torch** - Проверка fallback метрик на PyTorch
  - **test_gpu_metrics_returns_zeros_on_error** - Проверка возврата нулей при ошибке

### test_model_registry.py

Тесты для реестра моделей (`app.model_registry`):

- **TestModelRegistry** - Класс тестов реестра моделей
  - **test_load_from_db_and_prepare_fetches_gigaam_models** - Проверка загрузки GigaAM моделей из БД
  - **test_load_from_db_filters_only_enabled_models** - Проверка фильтрации только включенных моделей
  - **test_download_model_success_status_transitions** - Проверка успешного скачивания модели и переходов статусов
  - **test_download_model_failure_sets_error_status** - Проверка установки статуса ошибки при неудачном скачивании
  - **test_download_model_retries_on_failure** - Проверка повторных попыток при неудаче
  - **test_parallel_download_two_models** - Проверка параллельного скачивания двух моделей
  - **test_already_downloaded_model_skipped** - Проверка пропуска уже загруженной модели
  - **test_is_model_known_returns_true_for_downloaded** - Проверка распознавания известной модели
  - **test_unready_models_returns_non_downloaded** - Проверка получения списка незагруженных моделей

- **TestAppState** - Класс тестов состояния приложения
  - **test_fetch_hugging_face_token_sets_env_vars** - Проверка установки переменных окружения при получении токена Hugging Face

### test_postprocess.py

Тесты для модуля постобработки текста (`app.postprocess`):

- **TestPostprocess** - Класс тестов
  - **test_filters_short_lines** - Проверка фильтрации коротких строк (менее 2 символов)
  - **test_filters_non_cyrillic_lines** - Проверка фильтрации некириллических строк
  - **test_filters_triplet_repeats** - Проверка фильтрации триплетных повторов (aaa, ааа и т.д.)
  - **test_keeps_valid_lines** - Проверка сохранения валидных строк
  - **test_empty_input_returns_empty** - Проверка обработки пустого ввода
  - **test_triplet_repeat_detection** - Проверка функции обнаружения триплетных повторов
  - **test_handles_multiple_issues** - Проверка обработки текста с несколькими проблемами одновременно
  - **test_preserves_line_order** - Проверка сохранения порядка строк

### test_queueing.py

Тесты для модуля очереди задач (`app.queueing`):

- **TestJobQueue** - Класс тестов очереди задач
  - **test_enqueue_adds_job_to_queue** - Проверка добавления задачи в очередь
  - **test_start_workers_creates_correct_count** - Проверка создания правильного количества воркеров
  - **test_distribution_across_gpus** - Проверка распределения задач между GPU
  - **test_worker_handles_transcribe_error** - Проверка обработки ошибок транскрипции воркером
  - **test_callback_on_success** - Проверка успешной доставки callback
  - **test_callback_error_keeps_job_completed** - Проверка что задача остается completed даже при ошибке callback
  - **test_cleanup_removes_job_directory** - Проверка очистки директории задачи после завершения
  - **test_serialize_job_computes_timings** - Проверка вычисления временных характеристик задачи
  - **test_snapshot_ids_returns_correct_lists** - Проверка получения правильных списков queued и running задач

## Маркировки тестов

Тесты используют следующие маркировки pytest:

- `@pytest.mark.unit` - Юнит-тесты для отдельных компонентов
- `@pytest.mark.e2e` - End-to-end тесты для проверки интеграции компонентов
- `@pytest.mark.asyncio` - Отмечает асинхронные тесты

## Запуск тестов

Для запуска всех тестов:
```bash
pytest
```

Для запуска только юнит-тестов:
```bash
pytest -m unit
```

Для запуска только e2e-тестов:
```bash
pytest -m e2e
```

Для запуска тестов с подробным выводом:
```bash
pytest -v
```

## Требования для запуска тестов

Для корректного запуска тестов необходимо:
1. Установить зависимости из `requirements.txt`
2. Наличие тестовой базы данных или моков для доступа к данным
3. Для некоторых тестов может потребоваться доступ к GPU (хотя большинство использует моки)

## Заключение

Тестовое покрытие проекта включает:
- Тестирование отдельных компонентов (юнит-тесты)
- Тестирование интеграции компонентов (e2e-тесты)
- Проверку обработки ошибок и граничных случаев
- Тестирование асинхронной работы системы
- Проверку работы с внешними сервисами (ffmpeg, GPU, HTTP callbacks)

Это обеспечивает надежность и стабильность работы системы GigaAM при различных условиях эксплуатации.
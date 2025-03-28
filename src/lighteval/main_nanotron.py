# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# flake8: noqa: C901
import os

from typer import Option, Argument
from typing_extensions import Annotated, Optional


CACHE_DIR: str = os.getenv("HF_HOME", "/scratch")

HELP_PANEL_NAME_1 = "Common Parameters"
HELP_PANEL_NAME_2 = "Logging Parameters"
HELP_PANEL_NAME_3 = "Debug Parameters"
HELP_PANEL_NAME_4 = "Modeling Parameters"
HELP_PANEL_NAME_5 = "Judge Parameters"


SEED = 1234


def nanotron(
    # === general ===
    model_config_path: Annotated[
        str,
        Argument(
            help="Path to model config yaml file. (examples/model_configs/delta_model.yaml)"
        ),
    ],
    tasks: Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")],
    # === Common parameters ===
    use_chat_template: Annotated[
        bool, Option(help="Use chat template for evaluation.",
                     rich_help_panel=HELP_PANEL_NAME_4)
    ] = False,
    system_prompt: Annotated[
        Optional[str], Option(
            help="Use system prompt for evaluation.", rich_help_panel=HELP_PANEL_NAME_4)
    ] = None,
    dataset_loading_processes: Annotated[
        int, Option(help="Number of processes to use for dataset loading.",
                    rich_help_panel=HELP_PANEL_NAME_1)
    ] = 1,
    custom_tasks: Annotated[
        Optional[str], Option(
            help="Path to custom tasks directory.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    cache_dir: Annotated[
        str, Option(help="Cache directory for datasets and models.",
                    rich_help_panel=HELP_PANEL_NAME_1)
    ] = CACHE_DIR,
    num_fewshot_seeds: Annotated[
        int, Option(help="Number of seeds to use for few-shot evaluation.",
                    rich_help_panel=HELP_PANEL_NAME_1)
    ] = 1,
    load_responses_from_details_date_id: Annotated[
        Optional[str], Option(
            help="Load responses from details directory.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    # === parallelism parameters ===
    tp: Annotated[
        int, Option(help="Tensor parallelism size",
                    rich_help_panel=HELP_PANEL_NAME_4)
    ] = 1,
    pp: Annotated[
        int, Option(help="Pipeline parallelism size",
                    rich_help_panel=HELP_PANEL_NAME_4)
    ] = 1,
    dp: Annotated[
        int, Option(help="Data parallelism size",
                    rich_help_panel=HELP_PANEL_NAME_4)
    ] = 1,
    # === saving ===
    output_dir: Annotated[
        str, Option(help="Output directory for evaluation results.",
                    rich_help_panel=HELP_PANEL_NAME_2)
    ] = "results",
    push_to_hub: Annotated[
        bool, Option(help="Push results to the huggingface hub.",
                     rich_help_panel=HELP_PANEL_NAME_2)
    ] = False,
    push_to_tensorboard: Annotated[
        bool, Option(help="Push results to tensorboard.",
                     rich_help_panel=HELP_PANEL_NAME_2)
    ] = False,
    public_run: Annotated[
        bool, Option(help="Push results and details to a public repo.",
                     rich_help_panel=HELP_PANEL_NAME_2)
    ] = False,
    results_org: Annotated[
        Optional[str], Option(
            help="Organization to push results to.", rich_help_panel=HELP_PANEL_NAME_2)
    ] = None,
    save_details: Annotated[
        bool, Option(help="Save detailed, sample per sample, results.",
                     rich_help_panel=HELP_PANEL_NAME_2)
    ] = False,
    # === debug ===
    max_samples: Annotated[
        Optional[int], Option(
            help="Maximum number of samples to evaluate on.", rich_help_panel=HELP_PANEL_NAME_3)
    ] = None,
    override_batch_size: Annotated[
        int, Option(help="Override batch size for evaluation.",
                    rich_help_panel=HELP_PANEL_NAME_3)
    ] = -1,
    job_id: Annotated[
        int, Option(help="Optional job id for future reference.",
                    rich_help_panel=HELP_PANEL_NAME_3)
    ] = 0,
    nanotron_checkpoint_path: Annotated[
        Optional[str], Option(
            help="Path to the file containing the nanotron checkpoint to load.", rich_help_panel=HELP_PANEL_NAME_4)
    ] = None,
    # === judge parameters ===
    judge_api_key: Annotated[
        Optional[str], Option(
            help="API key for the LLM-as-a-judge model (e.g. Novita.ai or OpenAI)", rich_help_panel=HELP_PANEL_NAME_5)
    ] = None,
):
    """
    Evaluate models using Nanotron as backend.
    """
    from nanotron.config import Config, get_config_from_file

    from lighteval.config.lighteval_config import FullNanotronConfig, LightEvalConfig
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.logging.hierarchical_logger import htrack_block
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
    from lighteval.utils.imports import NO_NANOTRON_ERROR_MSG, is_nanotron_available
    from lighteval.utils.utils import EnvConfig

    env_config = EnvConfig(token=os.getenv("HF_TOKEN"),
                           cache_dir=cache_dir, judge_api_key=judge_api_key)

    if not is_nanotron_available():
        raise ImportError(NO_NANOTRON_ERROR_MSG)

    with htrack_block("Load nanotron config"):
        # Create nanotron config
        if not model_config_path.endswith(".yaml"):
            raise ValueError(
                "The model config path should point to a YAML file")

        model_config = get_config_from_file(
            model_config_path,
            config_class=Config,
            model_config_class=None,
            skip_unused_config_keys=True,
            skip_null_keys=True,
        )

        # We are getting an type error, because the get_config_from_file is not correctly typed,
        lighteval_config: LightEvalConfig = get_config_from_file(
            lighteval_config_path, config_class=LightEvalConfig)  # type: ignore
        nanotron_config = FullNanotronConfig(lighteval_config, model_config)

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
    )

    pipeline_parameters = PipelineParameters(
        launcher_type=ParallelismManager.NANOTRON,
        env_config=env_config,
        job_id=job_id,
        nanotron_checkpoint_path=nanotron_checkpoint_path,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        override_batch_size=override_batch_size,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_parameters,
        evaluation_tracker=evaluation_tracker,
        model_config=nanotron_config,
    )

    pipeline.evaluate()

    pipeline.show_results()

    pipeline.save_and_push_results()

"""Framework-free entry point for operational environment management."""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    # REMAINDER keeps command-local flags out of global prefix selection.
    prefix = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    prefix.add_argument("--work-dir", type=Path)
    # Consume the global value without configuring credentials for runtime env.
    prefix.add_argument("--hf-token-file", type=Path)
    prefix.add_argument("command", nargs=argparse.REMAINDER)
    args, extras = prefix.parse_known_args(sys.argv[1:])
    if not extras and args.command[:2] == ["runtime", "env"]:
        from klab.runtime_env_cli import main as environment_main
        from klab.workdir import resolve_work_dir

        if args.work_dir is not None:
            os.environ["KLAB_WORK_DIR"] = str(resolve_work_dir(args.work_dir))
        environment_main(args.command[2:])
    elif not extras and args.command[:2] == ["experiment", "recipe"]:
        from klab.experiments.recipe_cli import main as recipe_main

        # Inspection ignores execution-only global work/token selectors. In
        # particular it neither reads credentials nor configures a work root.
        # Preparation apply/resume requires its own explicit command-local root.
        recipe_main(args.command[2:])
    elif not extras and args.command[:2] == ["config", "validate"]:
        from klab.config.validation import main as validation_main

        validation_main(args.command[2:])
    elif not extras and args.command[:1] == ["lab-job"]:
        from klab.lab_job import main as job_main

        if args.work_dir is not None or args.hf_token_file is not None:
            prefix.error(
                "lab-job paths come only from its immutable manifest; global work/token options are unsupported"
            )
        job_main(args.command[1:])
    else:
        from klab.cli.main import main as application_main

        application_main()

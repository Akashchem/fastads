from pathlib import Path
from uuid import uuid4

import typer

from fastads.models import JobConfig
from fastads.pipeline import run_pipeline
from fastads.storage import create_job_dir, write_json

app = typer.Typer(help="FastAds command line interface.")


@app.callback()
def main() -> None:
    """FastAds CLI."""


@app.command()
def run(
    competitor: str = typer.Option(..., "--competitor", help="Competitor name."),
    market: str = typer.Option(..., "--market", help="Market code or region."),
    input: Path = typer.Option(..., "--input", exists=False, help="Input file path."),
) -> None:
    job_id = uuid4().hex[:12]
    job_dir = create_job_dir(job_id)

    job_config = JobConfig(
        job_id=job_id,
        competitor=competitor,
        market=market,
        input_path=str(input),
    )

    write_json(job_dir / "job_config.json", job_config.model_dump())

    typer.echo(f"competitor: {competitor}")
    typer.echo(f"market: {market}")
    typer.echo(f"input: {input}")
    typer.echo(f"job_id: {job_id}")

    run_pipeline(job_config)


if __name__ == "__main__":
    app()

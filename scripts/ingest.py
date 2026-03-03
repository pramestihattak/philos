#!/usr/bin/env python3
"""
Philos CLI — ingest documents into the local knowledge base.

Usage:
    python scripts/ingest.py data/documents/mybook.pdf
    python scripts/ingest.py data/documents/ --recursive
    python scripts/ingest.py --list
    python scripts/ingest.py --delete <doc_id>
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _init_services() -> None:
    from src.services.concurrency import init_semaphore
    from src.services.retriever import retriever
    init_semaphore()
    retriever.init()


def main(
    source: Path | None = typer.Argument(None, help="File or directory to ingest"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already stored"),
    list_docs: bool = typer.Option(False, "--list", "-l", help="List all ingested documents"),
    delete: str | None = typer.Option(None, "--delete", "-d", help="Delete document by doc_id"),
) -> None:
    """Philos document ingestion CLI."""
    _init_services()

    if list_docs:
        _do_list()
    elif delete:
        _do_delete(delete)
    elif source is not None:
        asyncio.run(_do_ingest(source, recursive=recursive, force=force))
    else:
        console.print("[yellow]No action specified. Use a path to ingest, --list, or --delete.[/yellow]")
        console.print("Run with --help for usage.")
        raise typer.Exit(1)


def _do_list() -> None:
    from src.services.retriever import retriever

    docs = retriever.list_documents()
    if not docs:
        console.print("[yellow]No documents in knowledge base.[/yellow]")
        return

    table = Table("Doc ID", "Filename", "Chunks")
    for d in docs:
        table.add_row(d["doc_id"], d["filename"], str(d["total_chunks"]))
    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {len(docs)} documents")


def _do_delete(doc_id: str) -> None:
    from src.services.retriever import retriever

    deleted = retriever.delete_document(doc_id)
    if deleted == 0:
        console.print(f"[red]No document found with id:[/red] {doc_id}")
        raise typer.Exit(1)
    console.print(f"[green]Deleted {deleted} chunks for doc_id={doc_id}[/green]")


async def _do_ingest(source: Path, recursive: bool, force: bool) -> None:
    from src.services.ingester import ingest_file, ingest_directory

    if source.is_dir():
        console.print(f"[bold]Ingesting directory:[/bold] {source} (recursive={recursive})")
        results = await ingest_directory(source, recursive=recursive)
    elif source.is_file():
        console.print(f"[bold]Ingesting file:[/bold] {source}")
        results = [await ingest_file(source, force=force)]
    else:
        console.print(f"[red]Path not found:[/red] {source}")
        raise typer.Exit(1)

    table = Table("Filename", "Status", "Chunks", "Doc ID", "Message")
    for r in results:
        color = {"ingested": "green", "skipped": "yellow", "error": "red"}.get(r.status, "white")
        table.add_row(
            r.filename,
            f"[{color}]{r.status}[/{color}]",
            str(r.chunks_created),
            r.doc_id[:12] + "..." if r.doc_id else "-",
            r.message or "",
        )
    console.print(table)

    ingested = sum(1 for r in results if r.status == "ingested")
    skipped = sum(1 for r in results if r.status == "skipped")
    errors = sum(1 for r in results if r.status == "error")
    console.print(f"\n[bold]Done:[/bold] {ingested} ingested, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    typer.run(main)

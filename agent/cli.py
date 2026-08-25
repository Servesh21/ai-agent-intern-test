"""Interactive command-line interface for the Aster & Row Support Agent."""

import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

from agent.agent_loop import SupportAgent
from agent.ingest import build_index


def main():
    parser = argparse.ArgumentParser(description="Aster & Row Support Agent CLI")
    parser.add_argument("--session-id", type=str, default=None, help="Custom session ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logs")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the ChromaDB vector index")
    args = parser.parse_args()

    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Aster & Row AI Support Agent[/bold cyan]\n"
            "[dim]Product and order support, grounded in our policies | Type 'exit' or 'quit' to quit, 'new' for new session[/dim]",
            border_style="cyan",
        )
    )

    if args.rebuild_index:
        console.print("[yellow]Building ChromaDB index...[/yellow]")
        build_index(force_rebuild=True)
        console.print("[green]Index build complete.[/green]\n")

    agent = SupportAgent()
    session_id = args.session_id

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting... Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "new":
            agent.session_manager.reset_session(session_id or "default")
            console.print("[yellow]Started new conversation session.[/yellow]")
            continue

        with console.status("[bold blue]Thinking...[/bold blue]"):
            response = agent.chat(user_input, session_id=session_id)

        # Print sources if present
        if response.sources:
            unique_sources = list(dict.fromkeys(response.sources))
            console.print(f"[dim cyan]Sources: {', '.join(unique_sources[:3])}[/dim cyan]")

        # Print tool calls if any
        if response.tool_calls:
            for tc in response.tool_calls:
                console.print(f"[dim magenta]Tool: {tc['name']}({tc.get('args', {})})[/dim magenta]")

        # Print agent response
        console.print(Markdown(response.text))

        # Highlight human handoff recommendation
        if response.handoff_recommended:
            console.print("[bold yellow]ℹ Human Support Recommended[/bold yellow]")


if __name__ == "__main__":
    main()

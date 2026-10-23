"""
Main Entrypoint for the Personal AI Research Agent.

Usage:
  # Interactive mode (demonstrating cross-session memory)
  python main.py

  # Direct research query
  python main.py --query "I'm researching Agentic RAG"

  # Memory management commands
  python main.py --list-memories
  python main.py --delete-memory <ID>
  python main.py --clear-memories
  python main.py --export-memories ./backup.json
"""

import sys
import argparse
from src.research_agent import PersonalResearchAgent
from src.models import ResearchBudget
from src.memory import MemoryStore


def print_banner():
    print("""
========================================================================
       PERSONAL AI RESEARCH AGENT with LONG-TERM MEMORY
========================================================================
Features:
  * Dynamic research planning & bounded autonomous web investigation
  * Strict budget enforcement (tool limits, tokens, execution timeouts)
  * Citation grounding audit & reflection loop (detects phantom sources)
  * Persistent SQLite + FTS5 full-text + vector semantic long-term memory
  * Zero-trust privacy scrubbing (redacts API keys and credentials)
========================================================================
""")


def interactive_session():
    print_banner()
    agent = PersonalResearchAgent(
        budget=ResearchBudget(
            max_tool_calls=6,
            max_execution_time_seconds=90.0,
            max_total_tokens=25000
        )
    )
    store = agent.memory
    session_count = 1

    print("Type your research questions below.")
    print("Example Flow:")
    print("  Session 1: 'I'm researching Agentic RAG.'")
    print("  Session 2: 'Continue my research and compare it with GraphRAG.'")
    print("Commands: ':memories' to view saved knowledge, ':clear' to wipe memories, ':exit' to quit.\n")

    while True:
        try:
            prompt = input(f"\n[Session {session_count}] Enter research question > ").strip()
            if not prompt:
                continue
            if prompt.lower() in (":exit", "exit", "quit", "q"):
                print("Exiting research agent. Goodbye!")
                break
            if prompt.lower() == ":memories":
                show_memories(store)
                continue
            if prompt.lower() == ":clear":
                cleared = store.clear_all()
                print(f"Memory store cleared ({cleared} records removed).")
                continue

            session_id = f"cli_session_{session_count}"
            state = agent.research(prompt, session_id=session_id)

            print("\n" + "=" * 70)
            print("RESEARCH DOSSIER COMPLETED")
            print("=" * 70)
            print(state.final_report)

            session_count += 1

        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated.")
            break


def show_memories(store: MemoryStore):
    memories = store.list_all()
    print("\n" + "=" * 60)
    print(f"STORED LONG-TERM RESEARCH MEMORIES ({len(memories)} records)")
    print("=" * 60)
    if not memories:
        print("No memories stored yet.")
        return
    for i, mem in enumerate(memories, 1):
        print(f"\n[{i}] ID: {mem.id} | Topic: {mem.topic}")
        print(f"    Confidence: {round(mem.confidence_score * 100, 1)}% | Access Count: {mem.access_count}")
        print(f"    Summary: {mem.summary[:140]}...")
        print(f"    Verified Claims: {len(mem.key_findings)} | Sources: {len(mem.sources)}")


def main():
    parser = argparse.ArgumentParser(description="Personal AI Research Agent CLI")
    parser.add_argument("--query", "-q", type=str, help="Run a single research mission directly")
    parser.add_argument("--list-memories", action="store_true", help="List all persistent research memories")
    parser.add_argument("--clear-memories", action="store_true", help="Clear all stored memories")
    parser.add_argument("--delete-memory", type=str, metavar="ID", help="Delete a specific memory by ID")
    parser.add_argument("--export-memories", type=str, metavar="FILE", help="Export memories to JSON")

    args = parser.parse_args()
    store = MemoryStore()

    if args.list_memories:
        show_memories(store)
        return

    if args.clear_memories:
        cleared = store.clear_all()
        print(f"Cleared {cleared} memories from persistent store.")
        return

    if args.delete_memory:
        deleted = store.delete(args.delete_memory)
        print(f"Memory '{args.delete_memory}' deleted: {deleted}")
        return

    if args.export_memories:
        count = store.export_to_json(args.export_memories)
        print(f"Exported {count} memories to {args.export_memories}")
        return

    if args.query:
        agent = PersonalResearchAgent()
        state = agent.research(args.query)
        print("\n" + "=" * 70)
        print("FINAL RESEARCH REPORT")
        print("=" * 70 + "\n")
        print(state.final_report)
        return

    # Default to interactive mode
    interactive_session()


if __name__ == "__main__":
    main()

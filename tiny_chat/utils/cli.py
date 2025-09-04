# tiny_chat/cli.py
import argparse
import asyncio
import json
from typing import Any

from tiny_chat.messages import ScriptBackground
from tiny_chat.providers.generate import (
    agenerate_action,
    agenerate_env_profile,
    agenerate_script,
)


def _print_result(obj: Any) -> None:
    """Pretty print for pydantic v2/v1 or plain dict/list."""
    try:
        model_dump = getattr(obj, "model_dump", None)
        if callable(model_dump):
            print(json.dumps(model_dump(), indent=2, ensure_ascii=False))
            return
    except Exception:
        pass
    try:
        to_dict = getattr(obj, "dict", None)
        if callable(to_dict):
            print(json.dumps(to_dict(), indent=2, ensure_ascii=False))
            return
    except Exception:
        pass
    if isinstance(obj, (dict | list)):
        print(json.dumps(obj, indent=2, ensure_ascii=False))
        return
    print(str(obj))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-chat",
        description="Tiny Chat - generate Sotopia-like artifacts via a simple CLI.",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )

    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument(
        "-m",
        "--model",
        default="gpt-4o-mini",
        help=argparse.SUPPRESS,
    )
    common_parent.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help=argparse.SUPPRESS,
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_env = sub.add_parser(
        "env-profile",
        parents=[common_parent],
        add_help=True,
        help="Generate environment profile (scenario + goals) from an inspiration prompt.",
        description="Generate environment profile (scenario + goals) from an inspiration prompt.",
    )
    p_env.add_argument(
        "inspiration",
        help="Inspiration prompt, e.g., 'asking my boyfriend to stop being friends with his ex'",
    )
    p_env.add_argument(
        "--examples",
        default="",
        help="Optional few-shot examples text.",
    )

    p_action = sub.add_parser(
        "action",
        parents=[common_parent],
        add_help=True,
        help="Generate one turn agent action.",
        description="Generate one turn agent action.",
    )
    p_action.add_argument(
        "--agent", required=True, help="Agent name (the speaking/acting agent)"
    )
    p_action.add_argument("--goal", required=True, help="Agent's hidden goal")
    p_action.add_argument(
        "--history", default="", help="Background + conversation history text"
    )
    p_action.add_argument(
        "--turn", type=int, default=1, help="Current turn number (default: 1)"
    )
    p_action.add_argument(
        "--action-type",
        dest="action_type",
        action="append",
        default=["speak"],
        help="Repeatable. Available action types (e.g., speak, leave). Default: speak",
    )
    p_action.add_argument(
        "--script-like",
        action="store_true",
        help="Model behaves like a playwright instead of the agent.",
    )

    p_script = sub.add_parser(
        "script",
        parents=[common_parent],
        add_help=True,
        help="Generate a script between two agents (<= 20 turns) or a single step.",
        description=(
            "Generate a script based on ScriptBackground. Use --background-file to pass a JSON with keys: "
            "scenario, p1_name, p2_name, p1_background, p2_background, p1_goal, p2_goal."
        ),
    )
    p_script.add_argument(
        "--background-file",
        required=True,
        help="Path to a JSON file containing ScriptBackground fields.",
    )
    p_script.add_argument(
        "--agent-name",
        dest="agent_name_list",
        action="append",
        default=[],
        help="Repeatable. Names of the agents appearing in the script (e.g., --agent-name A --agent-name B).",
    )
    p_script.add_argument(
        "--agent",
        default="",
        help="Focus agent name when using --single-step mode (optional).",
    )
    p_script.add_argument(
        "--history",
        default="",
        help="Optional prior conversation/history text to condition the generation.",
    )
    p_script.add_argument(
        "--single-step",
        action="store_true",
        help="Generate only one turn (single step) instead of a full script.",
    )

    return parser


async def _run_env_profile(args: argparse.Namespace) -> None:
    res = await agenerate_env_profile(
        model_name=args.model,
        inspiration_prompt=args.inspiration,
        examples=args.examples,
        temperature=args.temperature,
    )
    _print_result(res)


# ActionType in codebase is likely a Literal[...] type, so pass strings directly
async def _run_action(args: argparse.Namespace) -> None:
    # ActionType in codebase is likely a Literal[...] type, so pass strings directly
    res = await agenerate_action(
        model_name=args.model,
        history=args.history,
        turn_number=args.turn,
        action_types=args.action_type,
        agent=args.agent,
        goal=args.goal,
        temperature=args.temperature,
        script_like=args.script_like,
    )
    _print_result(res)


async def _run_script(args: argparse.Namespace) -> None:
    try:
        with open(args.background_file, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[error] Background file not found: {args.background_file}")
        return
    except json.JSONDecodeError as e:
        print(f"[error] Failed to parse JSON in {args.background_file}: {e}")
        return

    try:
        bg = ScriptBackground(**data)
    except Exception as e:
        print(
            "[error] Invalid background JSON. Expected keys: "
            "scenario, p1_name, p2_name, p1_background, p2_background, p1_goal, p2_goal"
        )
        print(f"Detail: {e}")
        return

    res, _raw = await agenerate_script(
        model_name=args.model,
        background=bg,
        temperature=args.temperature,
        agent_names=args.agent_name_list,
        agent_name=args.agent,
        history=args.history,
        single_step=args.single_step,
    )
    _print_result(res)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "env-profile":
        asyncio.run(_run_env_profile(args))
        return
    if args.cmd == "action":
        asyncio.run(_run_action(args))
        return
    if args.cmd == "script":
        asyncio.run(_run_script(args))
        return

    parser.error("No command specified.")


if __name__ == "__main__":
    main()

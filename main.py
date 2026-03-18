"""
Article Generator — entry point.

Usage:
    python main.py --input input.txt
    python main.py --input input.txt --words 1000 --model claude-opus-4-5
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from article_generator import run_pipeline
from input_parser import parse_input_file
from llm_client import DEFAULT_MODEL, LLMClient


def setup_logging(verbose: bool = False) -> None:
    """Configure logging to console and file."""
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated article generator using Anthropic LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input input.txt
  python main.py --input input.txt --words 1000
  python main.py --input input.txt --model claude-opus-4-5 --verbose
        """,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input text file (topic, bullets, language)",
    )
    parser.add_argument(
        "--words",
        type=int,
        default=800,
        help="Target word count for the article (default: 800)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("=== Article Generator started ===")
    logger.info(f"Input: {args.input} | Words: {args.words} | Model: {args.model}")

    try:
        # Step 1: Initialize LLM client
        client = LLMClient(model=args.model)

        # Step 2: Parse input (uses LLM to extract from free-form text)
        article_input = parse_input_file(args.input, client)

        # Step 3: Run pipeline
        draft_path, final_path = run_pipeline(
            article_input=article_input,
            client=client,
            word_count=args.words,
        )

        logger.info("=== Done ===")
        print(f"\n[OK] Article generated successfully!")
        print(f"   Draft : {draft_path}")
        print(f"   Final : {final_path}")

    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        print(f"\n[ERROR] Input error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(str(e))
        print(f"\n[ERROR] LLM error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

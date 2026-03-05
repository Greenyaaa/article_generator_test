# Article Generator

Automated article generator using the Anthropic Claude API.  
Takes a topic + bullet points as input, generates a draft, then runs an editor pass — saving both versions.

---

## Project Structure

```
article_generator/
├── main.py                  # Entry point, CLI
├── llm_client.py            # Anthropic API wrapper with retry logic
├── article_generator.py     # Generation pipeline (draft → edit → save)
├── input_parser.py          # Input file parser
├── prompts/
│   ├── draft_prompt.txt     # System prompt for draft generation
│   └── editor_prompt.txt    # System prompt for editor pass
├── tests/
│   ├── test_parser.py       # Tests for input parser
│   └── test_generator.py    # Tests for generation pipeline (mocked LLM)
├── output/                  # Generated articles saved here
│   ├── draft.md
│   └── final.md
├── input.txt                # Example input file
├── .env.example             # Environment variable template
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd article_generator
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
cp .env.example .env
# Edit .env and add your key:
# ANTHROPIC_API_KEY=your_api_key_here
```

Then export it in your shell:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

---

## Input File Format

Create a `.txt` file with the following structure:

```
Topic: Why small businesses should automate repetitive processes
Language: English
Bullets:
- manual work limits growth
- hiring increases costs
- automation improves scalability
```

- **Topic** — required. The article subject.
- **Language** — optional. Defaults to `English`.
- **Bullets** — required. At least one bullet point.

---

## Usage

```bash
# Basic usage
python main.py --input input.txt

# Custom word count
python main.py --input input.txt --words 1000

# Different model
python main.py --input input.txt --model claude-opus-4-5

# Verbose / debug logging
python main.py --input input.txt --verbose
```

### All CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | required | Path to input `.txt` file |
| `--words` | `800` | Target article word count |
| `--model` | `claude-sonnet-4-5` | Anthropic model to use |
| `--verbose` | off | Enable debug logging |

---

## Output

After a successful run:

```
output/
├── draft.md    ← First generation pass
└── final.md    ← Edited, improved version
```

Logs are written to `app.log` in the project root.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use a mocked LLM client — no API key required.

---

## How It Works

1. **Parse** — reads topic, language, and bullet points from the input file
2. **Draft** — sends a structured prompt to Claude, gets a ~800-word article body
3. **Edit** — sends the draft back to Claude with editor instructions (add intro/conclusion, fix structure, remove repetition)
4. **Save** — writes `draft.md` and `final.md` to the `output/` directory

The `LLMClient` retries up to 3 times on rate limits, connection errors, and 5xx server errors using exponential backoff.

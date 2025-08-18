# Summarization Length Targeting Experiments

This directory contains experiments to test different strategies for achieving target summary lengths.

## Overview

The experiment system tests various prompting strategies to see which ones most accurately hit target token counts when summarizing text. This helps identify the optimal approach for the RagZoom summarization system.

## Strategies Tested

1. **Absolute Token** - "Summarize in PRECISELY 200 tokens"
2. **Relative Token** - "Reduce by PRECISELY 250 tokens"  
3. **Absolute Character** - "Summarize in PRECISELY 1000 characters"
4. **Relative Character** - "Reduce by PRECISELY 1500 characters"
5. **Percentage** - "Compress to PRECISELY 50% of its original length"
6. **Word Count** - "Summarize in PRECISELY 150 words"

## Directory Structure

```
experiments/
├── strategies/           # Strategy implementations
│   ├── base_strategy.py # Abstract base class
│   └── *.py             # Individual strategies
├── results/             # Experiment outputs
│   ├── corpus.json      # Test corpus from the_hobbit.txt
│   ├── raw_results.json # Raw experiment data
│   └── *.png            # Visualization plots
├── generate_test_corpus.py  # Generate test chunks
├── run_experiments.py        # Main experiment runner
├── analyze_results.py        # Statistical analysis
└── test_single_strategy.py   # Quick test script
```

## Usage

### 1. Generate Test Corpus

```bash
python experiments/generate_test_corpus.py
```

This splits `test_data/the_hobbit.txt` into chunks of 200, 500, and 1000 tokens.

### 2. Run Experiments

```bash
# Run with all chunks (warning: many API calls!)
python experiments/run_experiments.py

# Run with a sample
python experiments/run_experiments.py --sample 50

# Specify output file
python experiments/run_experiments.py --output results/my_test.json
```

### 3. Analyze Results

```bash
python experiments/analyze_results.py

# Or specify input file
python experiments/analyze_results.py --input results/my_test.json
```

This generates:
- `analysis_report.md` - Statistical analysis
- Various `.png` files - Visualizations

## Quick Test

To test a single strategy:

```bash
python experiments/test_single_strategy.py
```

## Key Findings

After running experiments, the analysis will reveal:

1. **Most Accurate Strategy** - Which approach hits targets most precisely
2. **Systematic Biases** - Consistent over/under-shooting patterns
3. **Nice Fractions Hypothesis** - Whether round percentages (50%, 33%) work better than arbitrary ones (37%, 63%)
4. **Size Dependencies** - How performance varies with input text length
5. **Compensation Factors** - How to adjust for systematic biases

## Configuration

- Model: `gpt-5-nano` (hardcoded)
- Temperature: Minimal (via `reasoning_effort="minimal"`)
- Compression ratios tested: 10%, 20%, ..., 90% plus 37%, 42%, 58%, 63%, 73%

## Notes

- Experiments use simplified prompts focused only on length targeting
- No preceding context or other complexity
- Each experiment makes one API call
- Results include both successful and failed attempts
- Token counting uses `cl100k_base` encoding (GPT-4 tokenizer)
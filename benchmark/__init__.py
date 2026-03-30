"""
Clinical Trial Recruitment Benchmark

Evaluates LLMs on realistic patient recruitment tasks using gold-standard
patient-trial relevance judgments from TREC Clinical Trials.

Pipeline:
    1. python -m benchmark.generate_questions   → benchmark/data/questions.json
    2. python -m benchmark.run_benchmark --model <name> --provider <provider>
    3. python -m benchmark.evaluate             → comparison report
"""

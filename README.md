# Picto2Seq: Benchmarks for Picto to French Generation using LLMs 

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.40%2B-yellow)](https://huggingface.co/transformers/)
[![ARASAAC](https://img.shields.io/badge/Pictograms-ARASAAC-purple)](https://arasaac.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

PictoSeq is a simple, easily configurable experimental pipeline designed for surveying sequence-to-sequence LLMs for pictogram-to-text generation

## Research Objectives

Augmentative and Alternative Communication (AAC) systems rely on pictograms to help individuals with communication difficulties express themselves in natural language. This framework evaluates neural approaches for pictogram-to-text generationg:

1. **Architecture Comparison**: Evaluation of multiple state-of-the-art sequence-to-sequence models
2. **Training Strategy**: Comparison of direct, keyword-mediated, and hybrid training approaches  
3. **Evaluation**: Multi-metric assessment including BLEU, ROUGE-L, WER, and linguistic quality measures
4. **Reproducibility**: Complete experimental pipeline with detailed logging and result preservation

## Configurable Modules

### LLM Support

The model support can easily be extended in the ... Currently, we provide support for the following models: 

- **BARThez**: French BART model optimized for French text generation
- **French T5**: T5 model fine-tuned specifically for French summarization tasks
- **mT5-base**: Multilingual T5 model supporting cross-lingual capabilities

#### Adding New Models

Extend the framework by adding new model configurations:

```python
ModelConfig(
    name="new_model",
    model_id="huggingface/model-name",
    tokenizer_class="AutoTokenizer",
    description="Description of the model",
    is_multilingual=True
)
```

### Input Support

- **Direct**: Pictogram IDs → French sentences
- **Keywords**: ARASAAC keywords → French sentences  
- **Hybrid**: Pictogram IDs + keywords → French sentences
- **Multi-task**: Joint training on multiple representation types

### Decoding Strategies

- **Greedy Search**: Deterministic token selection
- **Beam Search**: Multiple hypothesis exploration with length penalty
- **Nucleus Sampling**: Stochastic generation with top-p filtering

###  Evaluation Metrcs

- **Automatic Metrics**: BLEU, ROUGE-L, Word Error Rate (WER), lexical diversity
- **Linguistic Analysis**: French-specific grammatical patterns and fluency assessment
- **Quality Metrics**: Generation success rate and adequacy scoring
- **Error Analysis**: Systematic categorization of generation failures

#### Custom Evaluation Metrics

Add domain-specific metrics by extending the `Evaluator` class:

```python
def _calculate_custom_metric(self, predictions, references):
    #  custom evaluation logic (pictoER..etc)
    return score
```


## Installation


### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/pictoseq.git
cd pictoseq

# Create and activate conda environment
conda create -n pictoseq python=3.10
conda activate pictoseq

# Install relevant dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets numpy matplotlib seaborn
pip install jiwer  # For WER calculation (optional)

pip install scikit-learn scipy tqdm requests
```

### Cluster Setup

For experiments with larger models, we recommend running the script on a distributed cluster, or a CUDA capable GPU. For ...

```bash
# Set up environment variables
export $path_to_data
export $path_to_outfolder

# Create directory structure
mkdir -p $path_to_data
mkdir -p $path_to_outfolder
```

## Dataset Prcoessing

### ProPicto Corpus Processing

The framework expects data in the processed ProPicto format. Place your data files in the following structure:

```
data/processed_propicto/
├── keywords_to_sentence/
│   ├── train/data.json
│   ├── valid/data.json
│   └── test/data.json
├── pictos_tokens_to_sentence/
│   ├── train/data.json
│   ├── valid/data.json
│   └── test/data.json
├── hybrid_to_sentence/
│   ├── train/data.json
│   ├── valid/data.json
│   └── test/data.json
└── direct_to_sentence/
    ├── train/data.json
    ├── valid/data.json
    └── test/data.json
```

Each `data.json` file should contain:

```json
[
  {
    "input_text": "mots: chat voir maison",
    "target_text": "je vois le chat à la maison",
    "pictogram_sequence": [37779, 11351, 17056]
  }
]
```

### ARASAAC Metadata

Download and prepare ARASAAC metadata for keyword extraction:

```bash
python scripts/data_processing/build_metadata.py \
    --data_file data/propicto_base.json \
    --cache_dir data/metadata \
    --build_images
```

## Usage

### Quick Start: Test Run

 small test run:

```bash
python main_runner.py --run-all --test-run
```

### Full Experimental Matrix

complete 12-experiment matrix (3 models × 4 data configurations):

```bash
python main_runner.py --run-all
```

### Single Experiment

specific model-data combination:

```bash
python main_runner.py --single-experiment \
    --model barthez \
    --data keywords_to_sentence
```

### SLURM support

For large-scale experiments using SLURM job scheduling

```bash
# Submit as SLURM job
sbatch --clusters=wice \
       --account=lp_your_project \
       --nodes=1 \
       --ntasks=18 \
       --partition=gpu_a100 \
       --gpus-per-node=1 \
       --time=24:00:00 \
       --wrap="python main_runner.py --run-all"
```

### Custom Configuration

```bash
python main_runner.py --run-all \
    --max-train 10000 \
    --max-val 1000 \
    --max-test 1000 \
    --epochs 3 \
    --batch-size 16 \
    --results-path $VSC_SCRATCH/custom_results
```

## Results Summary

The framework generates comprehensive results in organized directories:

```bash
results/
├── propicto_comprehensive_[timestamp]/
│   ├── experimental_matrix.json           # Overview of all experiments
│   ├── comprehensive_analysis.json        # Cross-experiment analysis
│   ├── visualizations/                    # Performance plots
│   │   ├── model_comparison.png
│   │   ├── data_config_comparison.png
│   │   └── decoding_strategy_comparison.png
│   └── [experiment_id]/                   # Individual experiment results
│       ├── experiment_config.json
│       ├── final_test_results.json
│       ├── final_test_metrics.json
│       ├── comprehensive_results.json
│       ├── strategy_comparison.png
│       └── final_model/                   # Trained model files
```


### Experimental Matrix

| Model Architecture | Training Approach | Decoding Strategies | Total Experiments |
|-------------------|------------------|-------------------|------------------|
| BARThez           | Direct, Keywords, Hybrid, Direct | Greedy, Beam, Nucleus | 12 |
| French T5         | Direct, Keywords, Hybrid, Direct | Greedy, Beam, Nucleus | 12 |
| mT5-base          | Direct, Keywords, Hybrid, Direct | Greedy, Beam, Nucleus | 12 |
| **Total**         | **4 approaches** | **3 strategies** | **36 configurations** |

### Evaluation Metrics

- **BLEU Score**: N-gram overlap with reference translations
- **ROUGE-L**: Longest common subsequence F1-score
- **Word Error Rate (WER)**: Edit distance-based error measurement
- **Generation Success Rate**: Percentage of valid outputs
- **Fluency Score**: Linguistic quality assessment
- **French Linguistic Metrics**: Article and pronoun usage patterns



### Folder tree

```
pictoseq/
├── main_runner.py                 # Main experimental pipeline
├── scripts/
│   ├── data_processing/          # Data preparation utilities
│   └── analysis/                 # Result analysis tools
├── models/                       # Trained model storage
├── data/                         # Dataset storage
└── results/                      # Experimental results
```

## References

1. Norré et al. (2022). Investigating the Medical Coverage of a Translation System into Pictographs for Patients with Intellectual Disability. [ACL](https://doi.org/10.18653/v1/2022.slpat-1.6)

2. Macaire et al. (2024). A Multimodal French Corpus of Aligned Speech, Text, and Pictogram Sequences for Speech-to-Pictogram Machine Translation. [ACL](https://aclanthology.org/2024.lrec-main.76/)

3. Mutal et al. (2022). A Neural Machine Translation Approach to Translate Text to Pictographs in a Medical Speech Translation System. [AMTA](https://aclanthology.org/2022.amta-research.19/)

4. ARASAAC. (2024). Aragonese Portal of Augmentative and Alternative Communication. [Website](https://arasaac.org/)

5. PropictoOrféo Corpus. (2022). A corpus of pictogram sequences aligned with French text. [Dataset](https://www.ortolang.fr/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


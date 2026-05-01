# Controlling Gender Bias

This repository contains the implementation for the paper:  
**"Controlling Gender Bias in Retrieval via a Backpack Architecture"**  
(arXiv: [2511.00875](https://arxiv.org/abs/2511.00875))

It provides a framework for debiasing text retrieval and ranking using **Backpack Language Models**, which represent words as weighted combinations of learned, non-contextual "senses" to enable fine-grained control over gender bias.

## 📄 Key Idea

Standard neural retrievers amplify societal biases present in training data. Backpack models allow us to adjust the contribution of specific word senses (e.g., those that encode gender stereotypes) during ranking. This framework:

- Identifies biased senses using small counterfactual probes
- Re-weights sense contributions at inference time to reduce bias
- Preserves retrieval quality (NDCG) while improving fairness metrics

## 📁 Repository Structure

- **`model/`** – Contains pre-trained or fine-tuned Backpack model components (sense vectors, configuration)
- **`Ranker_BP.py`** – Core ranking class; implements biased and debiased retrieval functions
- **`model_evaluation.py`** – Computes retrieval effectiveness (e.g., NDCG@10) and bias metrics (e.g., gender parity difference, bias amplification)
- **`sense_eval.py`** – Tools for attributing bias to specific sense vectors and for setting debiasing strength
- **`run_test.sh`** – Bash script that runs the full pipeline (index → retrieve → evaluate)


### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iman1234ahmadi/Backpack_Debias.git
   cd Backpack_Debias
   
   Create a virtual environment (optional):
   python -m venv venv
   source venv/bin/activate
   
   Install dependencies:
   
   pip install torch transformers numpy scikit-learn tqdm ir_measures
   
🧪 Running Experiments
Quick Test
Run the provided shell script to execute a full experiment (indexing, retrieval, evaluation):

bash run_test.sh

If you use this code in your research, please cite this paper:
```bibtex

@article{afzali2025controlling,
  title={Controlling Gender Bias in Retrieval via a Backpack Architecture},
  author={Afzali, Amirabbas and Velae, Amirreza and Ahmadi, Iman and Aliannejadi, Mohammad},
  journal={arXiv preprint arXiv:2511.00875},
  year={2025},
  url={https://arxiv.org/abs/2511.00875}
}

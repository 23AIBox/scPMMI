# scMamba

This is the official codebase for **scPMMI: A Pre-trained Multimodal Model for Integrative Analysis of Single-Cell Multi-Omics Beyond Highly Variable Feature Selection**.

scPMMI is a pre-trained multimodal model for integrative analysis of single-cell multi-omics data without relying on prior highly variable feature selection. The model introduces a cell tokenization strategy that aggregates features within the same genomic region while preserving genomic positional information, thereby enabling the efficient processing of tens of thousands of features per cell. Our results demonstrate that scPMMI learns biologically coherent cell representations by jointly modeling multi-omics data. Through further transfer learning, the model can be readily adapted to achieve superior performance across diverse downstream tasks, including multi-omics integration, cell type annotation, protein abundance prediction, transcriptomic enhancement, and trajectory reconstruction.


## Installation

- Python 3.10.13

  - `conda create -n your_env_name python=3.10`
- Install ``cuda vewrsion 12.8 `` and ``cudnn``
- Install ``pytorch 2.8.0``
- Requirements: requirements.txt

  - `pip install -r requirements.txt`
- Install ``causal_conv1d`` and ``mamba``

  - `pip install causal_conv1d==1.5.2`
  - `pip install mamba==2.2.5`

### Note

If ``cuda`` and ``cudnn`` aren't installed, ``causal_conv1d`` and ``mamba-ssm`` will be installed unsuccessfully.

- Install ``causal_conv1d``

```bash
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.5.2
pip install .

```

- Install ``mamba-ssm``

```bash
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v2.2.5
pip install -e .
```

## Tutorials
We offer several [notebooks](https://github.com/23AIBox/scPMMI/tree/main/tutorials) for for various downstream tasks as introductory tutorials. _Our latest studies demonstrate scPMMI is competitive on multi-omics integration, protein abundance prediction, cell type annotation, transcriptomic enhancement, and trajectory reconstruction tasks compared to other SOTA methods and pretrained models._


## Ckeckpoint

The pretraining checkpoints for the scPMMI model are available via [dropbox](https://www.dropbox.com/scl/fo/8lwe2kdvmgrc005m5209i/AK16rrYbW4Py3BxNMXd1icw?rlkey=tbkzbpoh9yw9mmmhxdjubrryb&st=slw892yz&dl=0). We might update our checkpoints from time to time.

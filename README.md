<div align="center">
  
# Candidate-Heuristic In-Context Learning: A New Framework for Enhancing MedVQA with Large Language Models

</div>

## 💡Overview
CH-ICL is a candidate-heuristic framework, which transforms images into text using just a few trainable parameters and leverages the contextual understanding capability of LLMs to enhance the performance of existing medical VQA models. The proposed method is plug-and-play, and its effectiveness has been demonstrated on three existing medical VQA datasets.

![overview](pic/overview.jpg)

## 📔Pathology Terminology Dictionary

The keywords of our dataset images are widely diverse, including various image types, systems and organs, diseases,
symptoms, staining techniques, etc.

![example](pic/example.jpg)

![distribution](pic/distribution.jpg)

## 🔨Setup

### Requirement
```
conda create -n chicl python=3.8
conda activate chicl
pip install -r requirements.txt
```

### 📑Data Preparation
Our data mainly comes from publicly available, free online Pathology Education Informational Resource ([PEIR](https://peir.path.uab.edu/library/index.php?/category/2)) Digital Library. 
We test our model on:
+ [VQA-RAD](https://osf.io/89kps/)
+ [SLAKE](https://www.med-vqa.com/slake/)
+ [PathVQA](https://github.com/UCSD-AI4H/PathVQA)

### 🔨Pre-trained weights
Coming Soon.

## 📝Acknowledgements
We also reference the excellent repos of BioMedCLIP(https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), PubMedCLIP(https://github.com/sarahESL/PubMedCLIP), in addition to other specific repos to the baselines we examined (see paper).

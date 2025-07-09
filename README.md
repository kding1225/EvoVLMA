# EvoVLMA: Evolutionary Vision-Language Model Adaptation, ACM Multimedia 2025

**Code will be released soon!**

## Introduction
Pre-trained Vision-Language Models (VLMs) have been exploited in various Computer Vision tasks (e.g., few-shot recognition) via model adaptation, such as prompt tuning and adapters. However, existing adaptation methods are designed by human experts, requiring significant time cost and experience. Inspired by recent advances in Large Language Models (LLMs) based code generation, we propose an Evolutionary Vision-Language Model Adaptation (EvoVLMA) method to automatically search training-free efficient adaptation algorithms for VLMs. We recognize feature selection and logits computation as the key functions in training-free VLM adaptation, and propose a two-stage LLM-assisted evolutionary algorithm for optimizing these parts in a sequential manner, effectively address the challenge posed by the expansive search space through a divide-and-conquer strategy. Besides, to enhance the stability and efficiency of searching process, we propose low-precision code conversion, web based code execution and process monitoring, leading to a highly effective automatic algorithm design system. Extensive experiments demonstrate that the algorithms found by EvoVLMA can obtain promising results compared to previous manually-designed ones. More specifically, in the 8-shot image classification setting, the classical APE algorithm can be improved by 1.91 points in recognition accuracy. This research opens new possibilities for automating the optimization of adaptation algorithms of pre-trained multimodal models.

## Citation

```bibtex
@article{evovlma,
    title={EvoVLMA: Evolutionary Vision-Language Model Adaptation},
    author={Kun Ding, Ying Wang, Shiming Xiang},
    journal = {ACM Multimedia},
    year={2025}
}
```

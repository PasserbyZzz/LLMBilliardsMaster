# LLMBilliardsMaster: A Hierarchical Framework for Billiards with Large Language Models

Codebase for paper: *LLMBilliardsMaster: A Hierarchical Framework for Billiards with Large Language Models*

[Kaiyang Xu](https://github.com/PasserbyZzz), [Han Wu](https://github.com/HanWu9918)

[Paper](https://github.com/PasserbyZzz/LLMBilliardsMaster/blob/main/report/LLMBilliardsMaster.pdf)

![Pipeline](https://github.com/PasserbyZzz/LLMBilliardsMaster/blob/main/images/llm_pipeline.png)

## Setup

We recommend Ubuntu 22.04 and Python 3.13.

### Setup conda env and package install

```
conda create -n poolenv python=3.13
conda activate poolenv
```

### Install pooltool

```
python -m pip install --extra-index-url https://archive.panda3d.org/simple --trusted-host archive.panda3d.org panda3d==1.10.15
python -m pip install --extra-index-url https://archive.panda3d.org/simple --trusted-host archive.panda3d.org pooltool-billiards==0.5.0
```

### Install other packages

```
pip install -r requirements.txt
```

### Acquire API Keys

This is required for prompting DeepSeek or Qwen LLMs. Put your key string in `aliyun_key.json`.

We recommand using API keys from [aliyun](https://bailian.console.aliyun.com). Otherwise, you can modify the following code to use OpenAI or other LLM APIs.

```python
client = openai.OpenAI(api_key=OPENAI_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
```

## Usage 

### Run an evaluation (LLMAgent vs BasicAgent)

```bash
python scripts/evaluate.py
```

### Ablation Study

We conducted 40 rounds across four models (DeepSeek-V3.2, Qwen-Plus, MoonShot-K2, and GLM-4.5-Air). The results demonstrate that our framework (Orange bars) consistently achieves a **>2x** improvement in Average Score compared to the raw LLM baseline (Gray bars).

![Ablation Study](https://github.com/PasserbyZzz/LLMBilliardsMaster/blob/main/images/llm_performance_comparison.png)

>*Note: The chart above is for reference only.*

## Future Work

We plan to fully implement the **Hierarchical Framework** as illustrated in the paper. 

![Hierarchical](https://github.com/PasserbyZzz/LLMBilliardsMaster/blob/main/images/hierarchical.png)

By tightly coupling the high-level LLM planner with the low-level Bayesian Optimization engine, we aim to combine strategic reasoning with pixel-level execution precision, ultimately pushing the agent towards human-level proficiency.


## Algorithmic Agents
If you are aiming for a higher and more stable winning rate, we also provide traditional algorithmic agents. 
We have implemented the following algorithmic agents:

- GeometricAgent
- Enhanced_Bayes_Agent(NewAgent)
- MCTSAgent
- EnsembleVotingAgent

You can find their implementation here: [Algorithmic Agents](/agent/AlgorithmicAgent/AlgorithmicAgents.py)

Their empirical winning rates are shown below:
![winning rate](/images/different_model_performance_comparison.png)

## Contact

Please direct to **`passerby_zzz@sjtu.edu.cn`** for any questions or suggestions. We welcome any **`Issues`** and **`Pull requests`**!

## References

- [pooltool: A sandbox billiards game that emphasizes realistic physics](https://github.com/ekiefl/pooltool)
- [RoCo: Dialectic Multi-Robot Collaboration with Large Language Models](https://github.com/MandiZhao/robot-collab)

## Cite

If you find this work helpful, please cite our project:

```
@article{xu2026llmbilliards,
  title={LLMBilliardsMaster: A Hierarchical Framework for Billiards with Large Language Models},
  author={Xu, Kaiyang and Wu, Han},
  journal={AI3603 Course Project, Shanghai Jiao Tong University},
  year={2026},
  publisher={GitHub},
  url={https://github.com/PasserbyZzz/LLMBilliardsMaster}
}
```

## **Wish for your Star⭐!**


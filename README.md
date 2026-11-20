# 中文 FAQ 检索：手写 BM25

输入一个问题，从 16 条虚构服务 FAQ 中返回相关问题和答案。NumPy 实现 BM25，与字符 TF-IDF 余弦相似度比较，适合展示信息检索的索引、排序与评估。

## 运行

在本项目目录使用 Python 3.11：

```bash
python -m pip install -r requirements.txt
python main.py
python main.py --query "快递到哪了怎么查询" --top-k 3
python main.py --query "量子纠缠实验"
```

`--output-dir` 修改输出位置。中文使用重叠双字片段，英文和数字使用词；无需分词词典或模型下载。

## 算法与数据

- `data/faqs.json`：16 条自编 FAQ；账号、课程和订单规则均为虚构示例。
- `data/queries.json`：24 条另写的查询及人工相关 FAQ ID，既不加入索引也不用于调参。
- BM25 包含 IDF、词频饱和与文档长度归一化，`k1=1.5, b=0.75`。查询里的重复词只计一次。字符 TF-IDF 的 2–3 gram 作为比较基线。
- 对全部 FAQ 排序，计算 MRR、Recall@3、Top-1 准确率。每个查询目前只有一个相关答案，因此 Recall@3 等于 HitRate@3。

零分文档不会返回；查询完全没有匹配片段时明确提示无匹配。非零分只说明词面重合，不能证明问题在知识库范围内，也不代表置信概率。此方法无法理解没有共同字词的同义表达。

## 输出

默认写入脚本旁的 `outputs/`：`metrics.json`（实际指标）、`evaluation.json`（每个查询的真实相关 ID 与排序）、`retrieval_metrics.png`（基线对比）、`demo.json`（用户查询的候选答案和分数）。

可直接增加 FAQ 和独立标注查询，不能用“文档自己查询自己”的结果代替泛化测试。这是自编小语料上的演示，没有使用公开检索基准，也未声称达成生产效果。

## 面试可以讲什么

BM25 为什么需要长度归一化、TF 与 IDF 各自解决什么问题、MRR 和 Recall@K 如何区别、为什么无词面交集的同义查询会失败。下一步可加入真实授权 FAQ、独立验证集、语义向量召回或重排序，并保持同一查询集合公平比较。

参考：[scikit-learn TF-IDF 文档](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)。代码与自编数据采用本目录 MIT 许可。

## 实际示例结果

见[本次实际运行结果](RESULTS.md)，包含生成的指标和结果图。重新运行 `python main.py` 可以复现流程。

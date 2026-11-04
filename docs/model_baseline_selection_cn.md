# 相关工作与主模型选择

范围：围绕蛋白工程/科学工具Agent选取六篇可核对原文的代表工作，包含2024—2026论文和预印本；这是有边界的定性对比，不是覆盖全领域的文献计量，也不按搜索结果数量声称某个精确型号全球最多。

| 原始工作 | 核对到的Agent模型 | 与本项目的对应及限制 |
|---|---|---|
| [ProtAgents，2024](https://arxiv.org/html/2402.04268v1) | GPT-4，经API | 多Agent调用蛋白建模/物理工具；不是GB1同预算闭环 |
| [ProteinEngine，2024](https://arxiv.org/html/2405.06658v1) | gpt-3.5-turbo，Implementations节 | 蛋白工具整合；不能误写成GPT-4o |
| [Virtual Lab，2025](https://www.nature.com/articles/s41586-025-09442-9)；[作者代码](https://github.com/zou-group/virtual_lab) | GPT-4o，论文使用gpt-4o-2024-08-06 | LLM组织研究、ESM等作为工具；两类模型角色分开 |
| [Aviary，2024预印本](https://arxiv.org/html/2412.21154v1) | Llama-3.1-8B-Instruct用于SeqQA/LitQA2训练；GPT-4o用于GSM8K/hotpotQA；Claude3.5用于强Agent比较 | 为可训练开源科学Agent提供依据；Llama实验是经BC/EI训练的Agent，不是本项目原模型，也不能声称论文用该Llama获得GB1进化成绩 |
| [VenusFactory2，2026预印本](https://arxiv.org/html/2603.27303v1) | 主干Claude3.7 Sonnet；匹配ProtAgent/SciToolAgent基线使用GPT-4o；另比较多个通用LLM | 说明固定工具和底层模型比较架构的必要性；不能将其所有比较模型都写成系统主干 |
| [BioDesignBench，2026预印本](https://pmc.ncbi.nlm.nih.gov/articles/PMC13174699/) | GPT-5、DeepSeek V3、Claude Sonnet4.5、Gemini2.5 Pro | 比较工具调用/评估深度，模型并非统一GPT-4o；成绩不是本任务fitness |

这六篇均出现GPT系列，但精确版本和承担角色不同；GPT-4/4o是常见历史参考，不能据此说GPT-4o是2026全领域唯一或绝对最多，也不能把Llama称为统计上最多。Llama有代表性开源Agent依据，当前本机已有可核验权重与真实运行记录。

结合用户已明确选择“优先Llama科学智能体对照”和本机16GB GPU、本地可执行要求，**主LLM固定Meta Llama 3.1 8B Instruct，实际实现是已核验的Q4_K_M GGUF**。报告同时注明量化、上下文、采样和推理后端；不把它称为原始FP16或复现Aviary的训练结果。GPT-4/4o保留文献参照，不列入没有调用过的实测成绩。ESM-2是可选蛋白表征扩展，P33权重仍缺失，不再阻塞Agent主线。

本次P34以完整当前数据集成为固定工具，增加原P2参照、随机、规则、普通Llama和知识Llama的同预算新比较。底层LLM不换型号、不重新下载已有正确权重，不根据新面板效果选模型。所有普通/知识LLM调用均真实本地推理，规则基线没有伪造的LLM调用。

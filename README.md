# FedMIT

In this paper, we explore a novel federated multimodal instruction tuning task(FedMIT), which is significant for collaboratively fine-tuning MLLMs on different types of multimodal instruction data on distributed devices. To solve the
new task, we propose a federated multimodal instruction tuning framework(Pilot). Our framework integrates two stages of
“adapter on adapter” into the connector of the vision encoder
and the LLM. In stage 1, we extract task-specific features and
client-specific features from visual information. In stage 2,
we build the cross-task Mixture-of-Adapters(CT-MoA) module to perform cross-task interaction. Each client can not only
capture personalized information of local data and learn taskrelated multimodal information, but also learn general knowledge from other tasks. In addition, we introduce an adaptive parameter aggregation strategy for text training parameters, which optimizes parameter aggregation by calculating
weights based on the euclidean distance between parameters,
so that parameter aggregation can benefit from positive effects to the greatest extent while effectively reducing negative effects. Our framework can collaboratively exploit distributed data from different local clients to learn cross-task
knowledge without being affected by the task heterogeneity
during instruction tuning. The effectiveness of our method is
verified in two different cross-task scenarios.

![image](https://github.com/user-attachments/assets/7c755511-edef-4d6b-a716-f7aac18c26cd)

The core code fine-tuning code has been released for everyone to refer to and learn from.

目标： 建立一个科学、健壮的模型训练与评估流程，通过引入验证集来有效监控模型训练、防止过拟合，并确保我们保存的是在未知数据上表现最优的模型。
第一步：数据层的改造 - 分割原始语料
核心任务： 修改 prepare_data.py，使其不再将所有数据都用于训练，而是按比例（如9:1）分割为训练集和验证集。
执行细节：
定位修改点： 在 PrepareData 类的 __init__ 方法中，找到 self.sentences = self.read_sentences() 这行代码。这是我们获取所有原始句子的位置。
实现分割逻辑：
在这行代码之后，引入 sklearn.model_selection.train_test_split 工具（如果没有，需添加到 requirements.txt）。这是一个非常标准且强大的数据分割工具。
使用 train_test_split 将 self.sentences 分割为 train_sentences 和 validation_sentences。可以设定 test_size=0.1（即10%作为验证集）和 random_state=42（确保每次分割结果一致，便于复现）。
调整数据处理流程：
将类中的 self.sentences 属性重命名为 self.train_sentences。
新增一个 self.validation_sentences 属性来存储验证集句子。
修改 build_basic_dataset 和 build_corpus_dataset 方法。让它们接收一个句子列表作为参数，而不是直接使用 self.sentences。
在 build_exec 方法中，分别调用这两个方法两次：一次传入训练集句子，生成 pku_training_words.txt 和 pku_training_labels.txt；另一次传入验证集句子，生成新的文件 pku_validation_words.txt 和 pku_validation_labels.txt。
更新文件初始化 (init.py)： init.py 的调用逻辑不需要改变，因为它只是执行 prepare_pku.build_exec()，而具体的分割逻辑已经被我们封装在 PrepareData 类内部了。
第二步：数据加载的适配 - 为模型准备验证数据
核心任务： 修改 transform_data_dnn.py，使其能够加载并处理我们新生成的验证集文件。
执行细节：
定位修改点： 在 TransformDataDNN 类的 __init__ 方法。
增加验证数据加载：
目前该类只加载 pku 语料。我们需要修改它的逻辑，使其可以同时处理训练数据和验证数据。
可以为 __init__ 方法增加一个 dataset_type 参数（值为 'train' 或 'validation'）。
当 dataset_type 是 'train' 时，加载 _training_ 相关文件。
当 dataset_type 是 'validation' 时，加载 _validation_ 相关文件。
实例化加载器：
在 train_models.py 中，我们将需要创建 TransformDataDNN 的两个实例：一个用于训练数据，一个用于验证数据。
第三步：模型训练流程的升级 - 引入验证评估
核心任务： 修改 seg_dnn.py 中的 train_optimized 方法，使其在每个训练轮次（epoch）后，使用验证集来评估模型并据此作出判断。
执行细节：
修改 train_optimized 签名：
为其增加两个新参数：x_val 和 y_val，用于接收验证集的特征和标签。
实现验证逻辑：
在每个 epoch 的主循环末尾（即打印 Epoch ... 总结 之后），增加一个“验证阶段”。
在这个阶段，调用 sess.run() 来计算模型在验证数据上的 total_loss 和 accuracy。
关键： 此时的 feed_dict 中，is_training 必须设置为 False，以关闭Dropout，确保评估的确定性。
更新Early Stopping机制：
创建一个新变量 best_val_loss 来记录最佳的验证集损失。
将原本用于判断是否保存模型的 if avg_loss < best_loss: 这行代码，修改为 if current_val_loss < best_val_loss:。
这意味着，我们从此只保存在验证集上表现最好的模型，这才是防止过拟合的正确做法。
日志与输出：
在每轮的总结输出中，同时打印 train_loss, train_acc, val_loss, val_acc。这能让我们清晰地监控模型是否出现过拟合（表现为 train_loss 持续下降，而 val_loss 开始上升）。
预期成果：
完成以上三步重构后，我们的项目将拥有一个工业级的训练流程。我们不仅能够更科学地训练和选择模型，还能通过观察训练和验证曲线，对模型的行为有更深刻的洞察，从而为后续更高阶的优化（如学习率调度、更复杂的模型结构等）打下坚实的基础。
> 主打一个快速开发.

# dataset builder

> 没错又是这个.

## Schema

包括3个agent自我交互: 拥有试题的teacher, 拥有向量库的good student和啥都没有的bad student.
teacher负责根据文档片段提出问题和对应的答案. students负责解答. 结果返回给teacher选出更好的答案.

## Todo

+ [x] bad student
+ [x] good student
+ [ ] teacher
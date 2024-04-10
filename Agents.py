__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import re
from langchain_core.runnables.base import Runnable, RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.config import RunnableConfig
from Messages import Question, Answer
from typing import Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from abc import abstractmethod
from functools import partial
from langchain.schema.document import Document




class Student(Runnable):

    def __init__(self, student_name:str) -> None:
        super().__init__()
        self.model_name = "/data/lixubin/models/Qwen/Qwen1.5-32B-Chat"
        self.base_url = "http://127.0.0.1:9779/v1"
        self.llm = ChatOpenAI(model=self.model_name ,base_url=self.base_url)
        self.format_input = RunnableLambda(self.format_input)
        self.format_output = RunnableLambda(self.format_output)
        self.chain = self.get_which_chain()
        self.student_name = student_name

    def format_output(self, input:AIMessage) -> Answer:
        return Answer(student_name=self.student_name, content=input.content, question=self.cur_question)

    @staticmethod
    def format_input(input:Question) -> str:
        return input.content

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    @abstractmethod
    def get_which_chain(self):
        raise NotImplementedError

    def invoke(self, 
        input: Question, 
        config: RunnableConfig | None = None
        ) -> Answer:

        self.cur_question = input


        ans:Answer = self.chain.invoke(input)
        ans.question = input
        return ans



class GoodStudent(Student):
    def get_which_chain(self):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = "/data/lixubin/models/m3e-base"
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': False}
        m3e = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048)
        loader = TextLoader("./index.md")
        docs = loader.load()
        chunked_docs = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            chunked_docs,
            m3e
        )

        retriever = vectorstore.as_retriever()
        prompt = ChatPromptTemplate.from_messages([
            ("user", """
                请一步一步思考, 利用下列检索到的上下文(Context)回答问题(Question)。
                如果上下文中没有有用的信息, 请自由发挥。
                Context:
                    {context}
                Question:
                    {question}
                """)
        ])
        chain = (
            {"context": RunnablePassthrough() | self.format_input | retriever | self.format_docs, "question": RunnablePassthrough() | self.format_input}
            | prompt
            | self.llm
            | self.format_output
        )

        return  chain



class NormalStudent(Student):
    
    def get_which_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("user", "请一步一步思考, 回答问题: {question}")
        ])
        chain = (
            {"question": RunnablePassthrough() | self.format_input}
            | prompt
            | self.llm
            | self.format_output
        )

        return chain



class Teacher(Runnable):
    """
    注意规则可控, 在prompt中写出来即可.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_name = "/data/lixubin/models/Qwen/Qwen1.5-32B-Chat"
        self.base_url = "http://127.0.0.1:9779/v1"
        self.llm = ChatOpenAI(model=self.model_name ,base_url=self.base_url)
        self.chain = self.get_chain()


    @staticmethod
    def find_student_name_fromlist(input:AIMessage) -> str:
        res = re.search(r"(?<=(最好的学生是)).*", input.content)
        assert not len(res.group()) == 0, ValueError(f"Can't match student name in content: {input.content}")
        return res.group()


    @staticmethod
    def format_student_answers(inputs:list[Answer]) -> str:
        answers = "\n\n".join([f"{ans.student_name}的答案:\n    {ans.content}" for ans in inputs])
        return answers


    def get_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("user", """你是一位阅卷教师, 请先根据问题(Question)正确答案(Correct_Answer)一步一步进行分析学生们的回答(Student_Answers)的好坏, 再根据打分标准(Regulars)从中挑选出最好的学生, 需要注意打分标准是按照重要性降序的.
                你的最后一句话应该是:"最好的学生是: (学生姓名)". 如果没有学生答对, 你的最后一句话应该是:"最好的学生是: none"
                Question:
                    {question}
                Correct_Answer:
                    {correct_answer}
                Student_Answers:
                    {student_answers}
                Regulars(从重要到次要):
                    1. 学生答案应该尽可能贴近标准答案,并且学生答案不应该过度分析;
                    1. 学生的答案应该尽可能简明扼要;
                    2. 答案应该有条理, 不应该逻辑混乱;
                    3. 学生应该充分考虑各种条件, 尽可能全面考虑.
                """)
        ])

        chain = (
            {
                "question": RunnablePassthrough() | RunnableLambda(lambda x: x[0].question.content),
                "correct_answer": RunnablePassthrough() | RunnableLambda(lambda x: x[0].question.correct_answer),
                "student_answers": RunnablePassthrough() | RunnableLambda(self.format_student_answers)
            }
            | prompt
            | self.llm
            | RunnableLambda(self.find_student_name_fromlist)
        )

        return chain
    

    def invoke(self, input: list[Answer], config: RunnableConfig | None = None) -> Answer:

        chain = self.get_chain()

        preference_studnet_name:str = chain.invoke(input)
        
        result = None
        for ans in input:
            if ans.student_name.lower() in preference_studnet_name.lower():
                result = ans
                break
        if "none" in preference_studnet_name.lower():
            result = Answer(student_name="correct_answer", content=input[0].question.correct_answer, question=input[0].question)

        assert not result is None

        return result



class Questioner(Runnable):

    def __init__(self) -> None:
        super().__init__()
        self.model_name = "/data/lixubin/models/Qwen/Qwen1.5-32B-Chat"
        self.base_url = "http://127.0.0.1:9779/v1"
        self.llm = ChatOpenAI(model=self.model_name ,base_url=self.base_url)
        self.chain = self.get_chain()

    @staticmethod
    def find_questions(input:AIMessage) -> list[Question]:
        # markdown_content = re.search(r"(?<=(```markdown))(.|[\r\n])*(?=(```))", input.content)
        markdown_content = re.search(r"(?<=(```markdown))(.|\n)*", input.content)
        assert not markdown_content is None, RuntimeError(f"Can't find markdown in: {input.content}")

        # questions = [i.group() for i in re.finditer(r"(?<=(题[:：])).*", markdown_content.group())]
        questions = [i.group() for i in re.finditer(r"(?<=(题目: ))[^\n]*", markdown_content.group())]
        answers = [j.group() for j in re.finditer(r"(?<=(答案: ))[^\n]*", markdown_content.group())]
        assert not (questions is None or answers is None), RuntimeError(f"Can't find questions or answers in: {input.content}")
        assert len(questions) == len(answers), RuntimeError(f"Numbers not match: {input.content}")

        return [Question(content=q.strip(), correct_answer=a.strip()) for q,a in zip(questions, answers)]

    @staticmethod
    def concat_docs(inputs:list[Document]) -> str:
        return "\n......\n".join([doc.page_content for doc in inputs])

    def get_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("user", """你是一个优秀教师, 你现在要根据课本知识出一些课堂测试题. 
                请针对含有知识点的课本片段内容(TextBook)制做3道高质量的困难的问题-答案对. 问题类型可以多样, 如阐述概念题,判断题,推理题等.
                你的回答应该使用markdown格式. 一定要参考接下来的示例输出, 在开头加上"```markdown", 在每个题目和问题前面都要加上"题目: ","答案: ".
                假设TextBook是关于阿尔卑斯山脉的信息, 对应的示例输出如下:
                示例输出:
                ```markdown
                + 第1题
                题目: 什么是阿尔卑斯山?
                答案: 阿尔卑斯山是欧洲最高及横跨范围最广的山脉，它覆盖了意大利北部边界、法国东南部、瑞士、列支敦士登、奥地利、德国南部及斯洛文尼亚。它可以被细分为三个部分：从地中海到勃朗峰的西阿尔卑斯山，从瓦莱达奥斯塔到布伦纳山口（奥地利和意大利交界处）的中阿尔卑斯山，从布伦纳山口到斯洛文尼亚的东阿尔卑斯山。欧洲许多大河都发源于此，水力资源丰富，为旅游、度假、疗养胜地。
                + 第2题
                题目: 为什么阿尔卑斯山南边和北边的气候差异大?
                答案: 阿尔卑斯山位在温带，但又有高海拔地形。世界上因为高海拔，以致气候类似极地的地区称为高山气候。由海平面往上升，气温会渐渐下降（见气温垂直递减率）。山上盛行风的影响，使得山下的暖空气流动到山上，其体积膨胀，因此会失去热量，因此水汽会凝结，产生降雨甚至降雪。阿尔卑斯山的高度阻挡了水汽，因此将阿尔卑斯山北边是水汽较多的气候，而南边则较为干燥。
                + 第3题
                题目: 阿尔卑斯山脉的地理意义是什么?
                答案: 阿尔卑斯山脉的气候成为中欧温带大陆性气候和南欧亚热带气候的分界线。山地气候冬凉夏暖。大致每升高200米，温度下降1℃，在海拔2000米处年平均气温为0℃。整个阿尔卑斯山湿度很大。年降水量一般为1200～2000毫米。海拔3000米左右为最大降水带。边缘地区年降水量和山脉内部年降水量差异很大。海拔3200米以上为终年积雪区。阿尔卑斯山区常有焚风出现，引起冰雪迅速融化或雪崩而造成灾害。阿尔卑斯山脉是欧洲许多河流的发源地和分水岭。多瑙河、莱茵河、波河、罗讷河都发源于此。山地河流上游，水流湍急，水力资源丰富，又有利于发电。此外,此地栖息着各种动植物,代表有阿尔卑斯大角山羊、山兔、雷鸟、小羚羊和土拨鼠等。
                ```
                (示例结束)

                TextBook:
                    {doc}
                """),
        ])
        
        # FIXME: docs can(should) be cross books
        chain = (
            {"doc": RunnablePassthrough() | RunnableLambda(self.concat_docs)}
            | prompt
            | self.llm
            | RunnableLambda(self.find_questions)
        )
        return chain

    def invoke(self, input:Document|list[Document], config: RunnableConfig | None = None) -> list[Question]:
        if not isinstance(input, list):
            input = [input]
        chain = self.get_chain()
        question = chain.invoke(input)
        return question




if __name__ == "__main__":
    import os
    os.environ["OPENAI_API_KEY"] = "sk-"

    # stu = GoodStudent("lithium")
    # question_1 = Question(content="春日影是什么?", correct_answer="是乐队CRYCHIC的代表作品.")
    # ans = stu.invoke(question_1)
    # print(ans)

    # question_1 = Question(content="春日影是什么?", correct_answer="是乐队CRYCHIC的代表作品.")
    # answer_1 = Answer(student_name="lithium",content="春日影是某个乐队的歌曲,具体来说是CRYCHIC",question=question_1)
    # answer_2 = Answer(student_name="andrewfleet",content="春日影的由来: 灯答应祥子组建乐队后，在祥子的引见下认识了吉他手若叶睦、贝斯手长崎爽世和鼓手椎名立希，五人共同组建了乐队CRYCHIC。由灯作词、祥子作曲，她们创作出了属于五人的第一首也是最后一首歌曲《春日影》。在正式练习的过程中，也许是由于灯向来孤僻内向的性格，担任主唱的她却唱不出声音某主唱：？。在其他人的全力帮助下，她们通过卡拉OK等方式让灯逐渐敢于开口歌唱，并最终录下了一曲完美的《春日影》。",
    # tea = Teacher()
    # prefernce = tea.invoke([answer_1, answer_2])
    # print(prefernce)

    maker = Questioner()
    doc = Document(page_content="""触发器（英语：Flip-flop, FF），中国大陆译作“触发器”、台湾及香港译作“正反器”，是一种具有两种稳态的用于储存的组件，可记录二进制数字信号“1”和“0”。触发器是一种双稳态多谐振荡器（bistable multivibrator）。该电路可以通过一个或多个施加在控制输入端的信号来改变自身的状态，并会有1个或2个输出。触发器是构成时序逻辑电路以及各种复杂数字系统的基本逻辑单元。触发器和锁存器是在计算机、通讯和许多其他类型的系统中使用的数字电子系统的基本组成部分。触发器的线路图由逻辑门组合而成，其结构均由SR锁存器派生而来（广义的触发器包括锁存器）。触发器可以处理输入、输出信号和时序脉波（CK）之间的相互影响。这里的触发器特指flip-flop，flip-flop一词主要是指具有两个状态相互翻转，例如编程语言中使用flip-flop buffer（翻译作双缓冲）""")
    print("[Text Chunck]\n"+doc.page_content)
    questions = maker.invoke(input=doc)
    print("\n[Make Questions]")
    print("\n".join([i.__repr__() for i in questions]))


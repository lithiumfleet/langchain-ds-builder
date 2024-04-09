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



class Student(Runnable):

    def __init__(self, student_name:str) -> None:
        super().__init__()
        self.model_name = "/data/lixubin/models/Qwen/Qwen1.5-32B-Chat"
        self.base_url = "http://127.0.0.1:9779/v1"
        self.llm = ChatOpenAI(model=self.model_name ,base_url=self.base_url)
        self.format_input = RunnableLambda(self.format_input)
        self.format_output = RunnableLambda(self.format_output)
        self.chain = None
        self.student_name = student_name

    def format_output(self, input:AIMessage) -> Answer:
        return Answer(student_name=self.student_name, content=input.content)

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

        chain = self.get_which_chain()

        ans:Answer = chain.invoke(input)
        ans.question = input
        return ans


class GoodStudent(Student):
    def get_which_chain(self):
        if not self.chain is None:
            return self.chain

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

        self.chain = chain
        
        return chain


class NormalStudent(Student):
    
    def get_which_chain(self):
        if not self.chain is None:
            return  self.chain
            
        prompt = ChatPromptTemplate.from_messages([
            ("user", "请一步一步思考, 回答问题: {question}")
        ])
        chain = (
            {"question": RunnablePassthrough() | self.format_input}
            | prompt
            | self.llm
            | self.format_output
        )

        self.chain = chain
        return 



class Teacher(Runnable):
    """
    注意规则可控, 在prompt中写出来即可.
    """

    def __init__(self) -> None:
        super().__init__()
        self.chain = None
        self.model_name = "/data/lixubin/models/Qwen/Qwen1.5-32B-Chat"
        self.base_url = "http://127.0.0.1:9779/v1"
        self.llm = ChatOpenAI(model=self.model_name ,base_url=self.base_url)


    @staticmethod
    def find_student_name_fromlist(input:AIMessage, namelist:list[str]):
        res = re.search(r"(?<=(最好的学生是[:：]))\s?[^\s]*(?=[\s。.])", input.content)
        assert not res is None, ValueError(f"Can't match student name in content: {input.content}")
        return res.group()


    @staticmethod
    def format_student_answers(inputs:list[Answer]) -> str:
        answers = "\n\n".join([f"{ans.student_name}的答案:\n    {ans.content}" for ans in inputs])
        return answers


    def get_chain(self):
        if not self.chain is None:
            return self.chain

        prompt = ChatPromptTemplate.from_messages([
            ("user", """你是一位资深心理学教授, 请先根据问题(Question)正确答案(Correct_Answer)一步一步进行分析学生们的回答(Student_Answers)的好坏, 再根据打分标准(Regulars)从中挑选出最好的学生, 需要注意打分标准是按照重要性降序的. 你的最后一句话应该是:"最好的学生是：(学生姓名)"
                Question:
                    {question}
                Correct_Answer:
                    {correct_answer}
                Student_Answers:
                    {student_answers}
                Regulars(从重要到次要):
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
            | self.find_student_name
        )
        
        self.chain = chain
        return chain
    

    def invoke(self, input: list[Answer], config: RunnableConfig | None = None) -> Answer:

        self.find_student_name = RunnableLambda(partial(self.find_student_name_fromlist,namelist=[i.student_name for i in input]))
        chain = self.get_chain()

        preference_studnet_name:str = chain.invoke(input)
        result = None
        for ans in input:
            if ans.student_name == preference_studnet_name:
                result = ans
                break
        assert not result is None

        return ans




if __name__ == "__main__":
    import os
    os.environ["OPENAI_API_KEY"] = "sk-"
    # stu = GoodStudent("lithium")
    question_1 = Question(content="春日影是什么?", correct_answer="是乐队CRYCHIC的代表作品.", doc=None)
    # ans = stu.invoke(question_1)
    # print(ans)
    answer_1 = Answer(student_name="lithium",content="春日影是某个乐队的歌曲,具体来说是CRYCHIC",question=question_1)
    answer_2 = Answer(student_name="andrewfleet",content="春日影的由来: 灯答应祥子组建乐队后，在祥子的引见下认识了吉他手若叶睦、贝斯手长崎爽世和鼓手椎名立希，五人共同组建了乐队CRYCHIC。由灯作词、祥子作曲，她们创作出了属于五人的第一首也是最后一首歌曲《春日影》。在正式练习的过程中，也许是由于灯向来孤僻内向的性格，担任主唱的她却唱不出声音某主唱：？。在其他人的全力帮助下，她们通过卡拉OK等方式让灯逐渐敢于开口歌唱，并最终录下了一曲完美的《春日影》。",
        question=question_1)
    tea = Teacher()
    prefernce = tea.invoke([answer_1, answer_2])
    print(prefernce)

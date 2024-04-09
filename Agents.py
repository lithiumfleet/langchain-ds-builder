__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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

    def __init__(self) -> None:
        super().__init__()
        self.chain = None
        self.model_name = "/data/lixubin/models/Qwen/Qwen1.5-32B-Chat"
        self.base_url = "http://127.0.0.1:9779/v1"
        self.llm = ChatOpenAI(model=self.model_name ,base_url=self.base_url)

    @staticmethod
    def find_student_name(inputs:AIMessage, namelist:list[str]):
        raise NotImplementedError

    @staticmethod
    def format_student_answers(inputs:list[Answer]) -> str:
        raise NotImplementedError


    def get_chain(self):
        if not self.chain is None:
            return self.chain

        prompt = ChatPromptTemplate.from_messages([
            ("user", """你是一位资深心理学教授, 请根据问题(Question)正确答案(Correct_Answer)从学生们的回答(Student_Answers)中挑选出最好的学生. 你需要先进行分析, 再说出"最好的学生是:(学生姓名)"
                Question:
                    {question}
                Correct_Answer:
                    {correct_answer}
                Student_Answers:
                    {student_answers}
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

        self.find_student_name = RunnableLambda(partial(self.find_student_name(namelist=[i.student_name for i in input])))
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
    # question_1 = Question(content="春日影是什么?", correct_answer="", doc=None)
    # ans = stu.invoke(question_1)
    # print(ans)
    tea = Teacher()
    filepath = "./index.md"

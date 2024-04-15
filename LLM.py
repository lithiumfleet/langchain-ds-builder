from langchain_openai import ChatOpenAI

def get_llm_client() -> ChatOpenAI:
    model_name = "/data/lixubin/models/Qwen/Qwen1.5-72B-Chat-GPTQ-Int4"
    base_url = "http://127.0.0.1:9780/v1"
    return ChatOpenAI(model=model_name ,base_url=base_url,api_key="sk-")

if __name__ == "__main__":
    llm = get_llm_client()
    print(llm.invoke("hello"))

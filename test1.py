from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Example
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    temperature = 0.1,
    return_full_text = False,
    max_new_tokens = 1024,
    task="text-generation"
)

system_prompt = "You are a helpful assistant answering general questions."
user_prompt = "{input}"

## llama 3
#token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

## phi 3
token_s, token_e = "<|system|>", "<|end|><|assistant|>"

prompt = ChatPromptTemplate.from_messages([
    ("system", token_s + system_prompt),
    ("user", user_prompt + token_e)
])

chain = prompt | llm

input = "Explain to me in up to 1 paragraph the concept of neural networks, clearly and objectively."

# res = chain.invoke({"input": input})
# print(res)
# print("-----")

# ### Example with Ollama (remove the triple quotes to run)
# from langchain_community.llms import HuggingFaceHub
# llm = HuggingFaceHub(repo_id="google/flan-t5-small")

llm = ChatOllama(
    model="tinyllama",
    num_gpu=0,  # Fuerza uso de CPU
    num_thread=2  # Limita hilos de procesamiento
)

chain3 = prompt | llm
res = chain3.invoke({"input": input})
print(res.content)
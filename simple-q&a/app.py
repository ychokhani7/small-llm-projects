import spaces
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # small, fast, no gated access needed

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to("cuda")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    return_full_text=False,
)

llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly, helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

def to_lc_history(history):
    msgs = []
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
    return msgs

@spaces.GPU(duration=120)  # GPU attached only while this runs
def respond(message, history):
    return chain.invoke({"history": to_lc_history(history), "question": message})

demo = gr.ChatInterface(fn=respond, title="My GPU Chatbot")

if __name__ == "__main__":
    demo.launch()
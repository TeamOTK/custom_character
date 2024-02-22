import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, TransformChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


def get_memory(): # 대화 기록을 저장하는 메모리
    memory = ConversationBufferMemory(memory_key="chat_history", ai_prefix="bot", human_prefix="you")
    return memory

def get_search_chain(name, set, line, situation): # 인격을 지정하기 위해 필요한 데이터를 가져오는 코드
    def get_data(input_variables):
        chat = input_variables["chat"]
        return {"name": name, "set": set, "line": line, "situation": situation}
    
    search_chain = TransformChain(input_variables=["chat"], output_variables=["name", "set", "line", "situation"], transform=get_data)
    return search_chain

def get_current_memory_chain(): # 현재 대화 기록을 가져오는 코드
    def transform_memory_func(input_variables):
        current_chat_history = input_variables["chat_history"].split("\n")[-10:]
        current_chat_history = "\n".join(current_chat_history)
        return{"current_chat_history": current_chat_history}
    
    current_memory_chain = TransformChain(input_variables=["chat_history"], output_variables=["current_chat_history"], transform=transform_memory_func)
    return current_memory_chain

def get_chatgpt_chain(): # GPT-4를 사용하여 대화를 생성하는 코드
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=os.environ["OPENAI_API_KEY"])
    
    template = """ 너는 'you'가 말을 했을 때 'bot'이 대답하는 것처럼 대화를 해 줘.
    
    'bot'의 이름은 {name}
    'bot'은 이런 설정을 갖고 있어. 각 설정은 키워드나 짧은 문장으로 되어 있고, /로 구분되어 있으니까 참고해서 설정에 오류가 없도록 대화를 해 줘.
    설정: {set}
    'bot'의 대사를 보여 줄 테니까, 이걸 보고 'bot'의 말과 습관을 잘 유추해서 말할 때 습관, 말투를 따라해. 각 문장은 /로 구분되어 있어.
    대사: {line}
    'bot'과 'you'는 이런 상황에서 대화를 하고 있어. 참고해서 상황에 맞춰서 대답해.
    상황: {situation}
    
    위에서 설명한 'bot'의 설정, 성격, 말투를 참고해서 'bot'의 말투와 성격을 잘 따라해 줘.
    다음 대화에서 'bot'이 할 것 같은 답변을 해 봐.
    1. 'bot'의 설정, 성격을 잘 반영해서 대답해야 돼. 없는 설정을 지어내면 안 돼.
    2. 자연스럽게 'bot'의 말투를 따라해서 주어진 상황에 맞춰서 대답해야 돼.
    3. 번역한 것 같은 말투, 기계적인 말투로 말하지 마.
    4. 'you'의 말을 이어서 만들지 말고, 'bot'의 말만 결과로 줘.
    5. 다섯 문장 이내로 짧게 대답해되, 'you'와 대화가 이어질 수 있도록 자연스럽게 말해 줘.
    6. 했던 말을 반복하거나, 앞뒤 문맥에 맞지 않는 말을 하지 마.
    7. 장애, 성차별, 폭력, 혐오, 성적인 내용을 포함한 대화는 하지 마. 윤리적이고 도덕적인 대화를 하도록 해.
    8. 대사에 써 있는 문장을 그대로 말하지 말고 한국어 문법과 문맥에 맞는 말을 해. 
    9. 대사에 써 있는 건 예시일 뿐이니까, 성격이나 설정에 혼동이 없도록 해. 그냥 말투만 참고하면 돼.
    
    이전 대화:
    {current_chat_history}
    you: {chat}
    bot: 
    """
    
    prompt_template = PromptTemplate(input_variables=["chat", "current_chat_history", "name", "set", "line", "situation"], template=template)
    chatgpt_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="received_chat")
    
    return chatgpt_chain

class Custom2:
    def __init__(self, name, set, line, situation) -> None:
        self.memory = get_memory()
        self.search_chain = get_search_chain(name, set, line, situation)
        self.current_memory_chain = get_current_memory_chain()
        self.chatgpt_chain = get_chatgpt_chain()
        
        self.overall_chain = SequentialChain(
            memory=self.memory,
            chains=[self.search_chain, self.current_memory_chain, self.chatgpt_chain],
            input_variables=["chat"],
            output_variables=["received_chat"],
            verbose=True
        )
    
    def receive_chat(self, chat):
        review = self.overall_chain.invoke({"chat": chat})
        return review['received_chat']
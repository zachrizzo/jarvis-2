import cv2
import time
import base64
import io
from PIL import Image
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.llms.ollama import Ollama
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Type, Literal
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain.memory import ChatMessageHistory
from langchain_core.messages import AnyMessage
import re
import dotenv
import os


dotenv.load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Define the chatbot's state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: dict
    ask_human: bool
    chat_history: list

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: dict

# Define a model for requesting human assistance
class RequestAssistance(BaseModel):
    """Tool to request human assistance when needed."""
    request: str = Field(description="The request for human assistance")

# Define the custom ImageReader tool
class ImageReaderInputs(BaseModel):
    """Inputs to the image reader tool."""
    input_text: str = Field(description="The user's prompt or input text for describing the image.")

class ImageReader(BaseTool):
    """Tool that captures an image from the camera and returns a description."""
    name: str = "image_reader"
    description: str = "Captures an image from the camera and returns a description."
    args_schema = ImageReaderInputs
    llm = Ollama(model='llava:13b', temperature=0)

    def _run(self, inputs: Dict[str, Any], run_manager=None) -> str:
        input_text = inputs.get("input_text", "")
        frame = self.capture_image()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_base64 = self.convert_to_base64(img)
        result = self.read_image(image_base64, input_text)
        if result is None:
            print("Failed to retrieve result.")
        print(f"Final result: {result}")
        return result

    def capture_image(self):
        camera = cv2.VideoCapture(0)
        time.sleep(2)
        ret, frame = camera.read()
        camera.release()
        if ret:
            print("Image captured.")
            return frame
        else:
            print("Failed to capture image.")
            return None

    def convert_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def read_image(self, image_base64, input_text=None):
        llm_with_image_context = self.llm.bind(images=[image_base64])
        response = llm_with_image_context.invoke(f"Describe the image in detail: here is the user prompt: {input_text}")
        print(f"LLM response: {response}")
        return response



assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. You can provide information, answer questions, and assist with tasks. your name is Jarvis."
            "Current user info:\n{user_info}\n"
            "Use this information to personalize your responses."
        ),
        ("placeholder", "{messages}"),
    ]
)

# Initialize the LLM and bind tools
llm = ChatOpenAI(model="gpt-4o", api_key=openai_key)
tool = TavilySearchResults(max_results=2)
image_tool = ImageReader()
tools = [tool, image_tool]
llm_with_tools = llm.bind_tools(tools)

def extract_info(text):
    # Dictionary of patterns to extract various types of information
    patterns = {
        'name': r'my name is (\w+)',
        'age': r'i am (\d+) years old',
        'location': r'i live in (\w+)',
        'occupation': r'i work as a(n?) (\w+)',
        'hobby': r'i enjoy (\w+ing)',
        'favorite_color': r'my favorite color is (\w+)',
        # Add more patterns here for other types of information
    }

    extracted_info = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text.lower())
        if match:
            extracted_info[key] = match.group(1)

    return extracted_info

def update_user_info(state: State) -> dict:
    user_info = state.get("user_info", {})
    last_user_message = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)

    if last_user_message:
        new_info = extract_info(last_user_message.content)
        user_info.update(new_info)

    return {"user_info": user_info}


def chatbot(state: State):
    chat_history = state.get("chat_history", [])
    user_info = state.get("user_info", {})

    # Add the user's message to chat history
    last_user_message = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if last_user_message:
        chat_history.append(last_user_message)

    # Create a formatted string of user info
    user_info_str = "\n".join([f"{key}: {value}" for key, value in user_info.items()])

    # Format the messages using the prompt
    formatted_messages = assistant_prompt.format_messages(
        user_info=user_info_str,
        messages=chat_history
    )

    # Invoke LLM with formatted messages
    response = llm_with_tools.invoke(formatted_messages)

    # Add AI's response to chat history
    chat_history.append(AIMessage(content=response.content))

    ask_human = False
    if response.tool_calls and response.tool_calls[0]["name"] == RequestAssistance.__name__:
        ask_human = True

    return {
        "messages": [response],
        "ask_human": ask_human,
        "chat_history": chat_history,
        "user_info": user_info
    }


# Define the human node function
def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )

def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        "messages": new_messages,
        "ask_human": False,
    }



# Define the conditional routing function
def select_next_node(state: State) -> Literal["human", "tools", "__end__"]:
    if state["ask_human"]:
        return "human"
    return tools_condition(state)


builder = StateGraph(State)

builder.add_node("update_user_info", update_user_info)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human", human_node)

builder.set_entry_point("update_user_info")
builder.add_edge("update_user_info", "chatbot")
builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {
        "human": "human",
        "tools": "tools",
        "__end__": "__end__",
    },
)
builder.add_edge("tools", "chatbot")
builder.add_edge("human", "chatbot")

memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["human"],
)

# Function to run the chatbot
def run_chatbot():
    config = {"configurable": {"thread_id": "1"}}
    chat_history = []
    user_info = {}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "chat_history": chat_history,
            "user_info": user_info,
            "ask_human": False
        }

        for event in graph.stream(initial_state, config, stream_mode="values"):
            print("Debug event:", event)
            for key, value in event.items():
                if key == "messages":
                    messages = value
                    if messages and isinstance(messages[-1], BaseMessage):
                        print("Assistant:", messages[-1].content)

            # Update chat_history and user_info after each event
            if "chat_history" in event:
                chat_history = event["chat_history"]
            if "user_info" in event:
                user_info = event["user_info"]

run_chatbot()



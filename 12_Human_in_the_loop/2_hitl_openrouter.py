import json
import os
from typing import Annotated, TypedDict
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError(
        "Set OPENROUTER_API_KEY in your environment or .env file before running this file."
    )


def ask_openrouter(messages: list[BaseMessage]) -> AIMessage:
    roles = {"human": "user", "ai": "assistant", "system": "system"}
    request_messages = []
    for message in messages:
        role = roles.get(message.type)
        if role is None:
            raise ValueError(f"Unsupported message type for OpenRouter: {message.type}")
        request_messages.append({"role": role, "content": message.content})

    request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(
            {
                "model": "openrouter/free",
                "messages": request_messages,
                "temperature": 0.7,
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter request failed with HTTP {error.code}: {error_body}"
        ) from error

    content = result["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("OpenRouter returned a non-text response.")
    return AIMessage(content=content)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> dict[str, list[BaseMessage]]:
    decision = interrupt(
        {
            "type": "approval",
            "reason": "The model is about to answer a user question.",
            "question": state["messages"][-1].content,
            "instruction": "Approve this question? yes/no",
        }
    )
    approval = decision.get("approved", "") if isinstance(decision, dict) else decision

    if isinstance(approval, str) and approval.strip().lower() in {"y", "yes"}:
        return {"messages": [ask_openrouter(state["messages"])]}

    return {
        "messages": [
            AIMessage(content="Question not approved. Please ask another question.")
        ]
    }


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
bot = graph.compile(checkpointer=MemorySaver())


def main() -> None:
    question = input("Ask a question: ").strip()
    if not question:
        raise ValueError("The question cannot be empty.")

    config = {"configurable": {"thread_id": str(uuid4())}}
    paused = bot.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    approval_request = paused["__interrupt__"][0].value
    print(f"\nApproval required: {approval_request['question']}")
    answer = input("Approve this question? (y/n): ").strip().lower()

    result = bot.invoke(
        Command(resume={"approved": answer}),
        config=config,
    )
    print(f"\n{result['messages'][-1].content}")


if __name__ == "__main__":
    main()

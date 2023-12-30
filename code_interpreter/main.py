from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_python_agent, create_csv_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.chat_models import ChatOpenAI

def main():
    print("Start...")
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # python_agent_executor.run(
    #     """generate and save in current working directory 15 QRcodes
    #                             that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    # )

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    csv_agent.run("how many columns are there in file episode_info.csv")
    csv_agent.run("print seasons ascending order of the number of episodes they have")


if __name__ == "__main__":
    main()

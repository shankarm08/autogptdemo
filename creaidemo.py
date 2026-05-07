from crewai import Agent, Task, Crew, LLM

# Use installed Ollama model
llm = LLM(
    model="ollama/llama3.2:1b",
    base_url="http://localhost:11434"
)

researcher = Agent(
    role="Researcher",
    goal="Find AI project ideas",
    backstory="AI expert",
    llm=llm
)

writer = Agent(
    role="Writer",
    goal="Write clean summaries",
    backstory="Technical content writer",
    llm=llm
)

coder=Agent(
     role="coder",
    goal="calculator app in python",
    backstory="Technical content writer",
    llm=llm

)



task1 = Task(
    description="Research 3 AI project ideas",
    expected_output="List of AI project ideas",
    agent=researcher
)

task2 = Task(
    description="Write short explanations for those ideas",
    expected_output="Short explanations",
    agent=writer
)
task3 = Task(
    description="calculator app in python",
    expected_output="Short explanations",
    agent=coder
)

crew = Crew(
    agents=[researcher, writer,coder],
    tasks=[task1, task2,task3]
)

result = crew.kickoff()

print(result)
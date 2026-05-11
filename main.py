from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_core.prompts import MessagesPlaceholder
import gradio as gr 
from data import employees, leave_balance

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

@tool
def get_employee_details(employee_id: str) -> dict:
    """Get employee details (name, department, role) using employee ID."""
    if employee_id in employees:
        return {
            "employee_id": employee_id,
            **employees[employee_id]
        }
    else:
        return {
            "error": f"Employee with ID {employee_id} not found"
        }
        
        
@tool       
def check_leave_balance(employee_id: str) -> dict:
    """Check the remaining leave balance for an employee using their ID."""
    if employee_id in leave_balance:
        return {
            "employee_id": employee_id,
            "remaining_leave_days": leave_balance[employee_id]
        }
    else:
        return {
            "error": f"Leave balance not found for employee ID {employee_id}"
        }

@tool
def generate_interview_questions(job_role: str) -> dict:
    """Generate interview questions for a given job role using AI."""

    prompt = f"""
    You are an HR expert.

    Generate 5 interview questions for the role: {job_role}

    Return only the questions as a numbered list.
    """

    response = llm.invoke(prompt)

    return {
        "job_role": job_role,
        "questions": response.content
    }
    
tools = [get_employee_details, check_leave_balance, generate_interview_questions]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful HR assistant. You can provide employee details, check leave balances, and generate interview questions based on job roles."), 
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def coding_assistant(message, history):
    response = agent_executor.invoke({"input" :message})
    return response['output']
if __name__ == "__main__":
    import gradio as gr
    iface = gr.ChatInterface(
        fn = coding_assistant,
        title="💬 Coding Assistant Chat",
        description="Chat with an AI that can execute Python code, debug, and search!",
    )
    iface.launch()

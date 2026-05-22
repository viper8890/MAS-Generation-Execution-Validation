import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.messages import TextMessage
from autogen_core.code_executor import CodeBlock
from autogen_core import CancellationToken


async def main():
    print("🚀 Initializing Modern AutoGen v0.4 Interactive CLI...\n")

    # 1. Define the LLM Client
    model_client = OpenAIChatCompletionClient(
        model="llama-3.3-70b-versatile", 
        api_key="xxxxxxxxxxxxxxxxxxxxxx",
        base_url="https://api.groq.com/openai/v1",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "llama3",
            "temperature": 0.0,
            },
        max_retries=5
    )

    
    print("="*60)
    print("🤖 Autonomous Coding Agent CLI - Type 'exit' to quit")
    print("="*60)
    
    task_counter = 1
    
    try:
        while True:
            user_prompt = input(f"\n[Task {task_counter}] Enter your coding prompt:\n> ")
            
            if user_prompt.strip().lower() in ['exit', 'quit', 'q']:
                print("\nShutting down the agent workspace. Goodbye!")
                break
                
            if not user_prompt.strip():
                continue
                
            print(f"\nAgents are working on Task {task_counter}...\n")
            print("📦 Booting ephemeral Docker container for this specific task...")
            
            # 2. Setup of Ephemeral Docker Code Execution (INSIDE the loop)
            # We use a unique workspace folder for each task to prevent file contamination
            task_workspace = f"workspace_task_{task_counter}"
            os.makedirs(task_workspace, exist_ok=True)
            
            code_executor = DockerCommandLineCodeExecutor(
                work_dir=task_workspace,
                image="python:3.11-slim",
            )
            await code_executor.start()
            await code_executor.execute_code_blocks(
                [CodeBlock(code="pip install requests", language="sh")],
                cancellation_token=CancellationToken()
            )
            
            try:
                # 3. Create the Agents
                coder = AssistantAgent(
                    name="Coder",
                    model_client=model_client,
                    system_message="""You are a senior Pyhton software engineer. 
                    Your objective is to write Python scripts to solve the user's request. 
                    You must output the complete, executable code within a single markdown code block. 
                    Do not output code diffs, partial updates, or pseudo-code. Anticipate missing dependencies and write defensive, error-handling logic."""
                )

                reviewer = AssistantAgent(
                    name="Reviewer",
                    model_client=model_client,
                    system_message="""You are a strict code reviewer. 
                    You will review the original request and the terminal output provided by the Executor. 
                    If the terminal output demonstrates that the code successfully accomplished the user's exact request, you must output the exact string 'TERMINATE' and nothing else. 
                    Do not offer polite remarks, congratulations, or suggestions for minor optimizations. 
                    If the code failed, provide a brief technical explanation of the traceback to the Coder, do not write the code yourself."""
                )

                # Executor is an AssistantAgent mapped to the code_executor tool
                executor = CodeExecutorAgent(
                    name="Executor",
                    code_executor=code_executor
                )

                # 4. Creating the Team (Group Chat)
                # The terminal condition stops the loop if someone says "TERMINATE" or after 10 messages
                text_termination = TextMentionTermination("TERMINATE")
                max_msgs = MaxMessageTermination(max_messages=10)
                termination = text_termination | max_msgs
    
                team = RoundRobinGroupChat(
                    participants=[coder, executor, reviewer],
                    termination_condition=termination
                )

                # 5. Running the Autonomous Loop
                async for message in team.run_stream(task=user_prompt):
                    if isinstance(message, TextMessage):
                        print(f"[{message.source}] =>\n{message.content}\n{'-'*40}")
                    
            finally:
                # 6. CRITICAL: Destroy the container when the task is done or fails
                print("🗑️ Destroying ephemeral container...")
                await code_executor.stop()
                
            task_counter += 1
            
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting...")
        

if __name__ == "__main__":
    # Modern AutoGen requires an asyncio event loop
    asyncio.run(main())
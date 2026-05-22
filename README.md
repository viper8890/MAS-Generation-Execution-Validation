# MAS-Generation-Execution-Validation
A Multi-Agent System for automatic code generation, execution and validation for generating automated code.
Autogen 0.4 is used for creating the AI agents and the groupchat. The coder and reviewer agents are AI agents powered by LLM (llama-3.3-70b-versatile) while the executor isn't an AI agent and operates the code execution in the Docker container, passing the stdout/stderr to the reviewer.

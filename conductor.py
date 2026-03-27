import json
import os
import subprocess
import sys

def load_plan(filename="MASTER_PLAN.json"):
    """Loads the JSON master plan."""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found in this directory.")
        sys.exit(1)
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    plan = load_plan()
    tasks = plan.get("tasks", [])
    
    if not tasks:
        print("No tasks found in MASTER_PLAN.json.")
        return

    for i, task in enumerate(tasks):
        task_id = task.get("id", f"Task_{i+1}")
        description = task.get("description", "")
        target_file = task.get("target_file", "")
        
        # Clever check: If the markdown file already exists, the task is done!
        if target_file and os.path.exists(target_file):
            print(f"✅ [{task_id}] Skipped: '{target_file}' already exists.")
            continue
            
        print(f"\n{'='*50}")
        print(f"🚀 STARTING TASK: {task_id}")
        print(f"{'='*50}\n")
        
        # Build the exact prompt for the AI
        prompt = (
            f"Please complete the following task:\n\n"
            f"{description}\n\n"
            f"When finished, you must write your summary and results to {target_file}."
        )
        
        # The command to trigger the Gemini CLI in non-interactive mode (-p)
        # We use shell=True because Windows requires it to find npm global packages
        command = f'gemini -p "{prompt}"'
        
        print(f"Handing instructions to Gemini CLI...\n")
        
        try:
            # This line wakes up the AI and gives it control of the terminal
            subprocess.run(command, shell=True, check=True)
            
            print(f"\n🛑 {task_id} completed by the agent.")
            print(f"Please review {target_file} and the code changes before running the conductor again.")
            break # We break here so it only does ONE task at a time, letting you verify.
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error: The agent crashed or was interrupted: {e}")
            break

if __name__ == "__main__":
    main()
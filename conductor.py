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

def run_agent(model_cli, prompt, image_path=""):
    """Executes the prompt using the specified CLI model."""
    
    # Check if an image path was provided AND if the file actually exists
    if image_path and os.path.exists(image_path):
        print(f"📎 Attaching image: {image_path}")
        command = f'{model_cli} -p "{prompt}" "{image_path}"'
    else:
        command = f'{model_cli} -p "{prompt}"'
        
    print(f"Handing instructions to {model_cli.upper()} CLI...\n")
    subprocess.run(command, shell=True, check=True)

def main():
    plan = load_plan()
    tasks = plan.get("tasks", [])
    
    if not tasks:
        print("No tasks found in MASTER_PLAN.json.")
        return

    for i, task in enumerate(tasks):
        # Extract variables mapping exactly to your new JSON keys
        task_id = task.get("id", f"Task_{i+1}")
        phase_goal = task.get("overarching_goal", "")
        description = task.get("task_description", "")
        target_file = task.get("target_file", "")
        starting_model = task.get("starting_ai_model", "gemini-3.1-pro")
        handoff_model = task.get("handoff_ai_model", "claude-opus-4.6")
        handoff_loops = task.get("#_agent_handoff_loops", 0)
        handoff_desc = task.get("handoff_description", "Please review the completed work for errors.")
        image_path = task.get("image_path", "")
        
        # Clever check: If the markdown file already exists, the task is done!
        if target_file and os.path.exists(target_file):
            print(f"✅ [{task_id}] Skipped: '{target_file}' already exists.")
            continue
            
        print(f"\n{'='*50}")
        print(f"🚀 STARTING TASK: {task_id}")
        print(f"{'='*50}\n")
        
        # ---------------------------------------------------------
        # PHASE 1: Execution by Starting Model
        # ---------------------------------------------------------
        print(f"--- Phase 1: Generation ({starting_model}) ---")
        prompt_1 = (
            f"Overarching Goal: {phase_goal}\n\n"
            f"Please complete the following specific task:\n\n"
            f"{description}\n\n"
            f"When finished, you must write your summary and results to {target_file}."
        )
        
        try:
            # Wake up the starting AI
            run_agent(starting_model, prompt_1, image_path)
            
            # ---------------------------------------------------------
            # PHASE 2: Verification by Handoff Model
            # ---------------------------------------------------------
            if handoff_loops > 0:
                print(f"\n--- Phase 2: Handoff & Verification ({handoff_model}) ---")
                
                for loop_num in range(handoff_loops):
                    print(f"Executing Handoff Loop {loop_num + 1}/{handoff_loops}...\n")
                    prompt_2 = (
                        f"You are reviewing a task just completed by another AI agent.\n\n"
                        f"Original Task Context:\n{description}\n\n"
                        f"Your Specific Review Instructions:\n{handoff_desc}\n\n"
                        f"Please read the results in {target_file} and the surrounding codebase. "
                        f"Make any necessary code corrections, and append your review notes to {target_file}."
                    )
                    
                    # We do not pass the image to the reviewer here, just the code/markdown check
                    run_agent(handoff_model, prompt_2)

            print(f"\n🛑 {task_id} fully completed.")
            
            # ---------------------------------------------------------
            # THE PAUSE: Wait for User Approval
            # ---------------------------------------------------------
            print(f"Please review {target_file} and your codebase.")
            print(f"💡 Tip: Verify the changes in your VS Code preview or Jupyter Notebook.")
            
            user_approval = input("\nDoes the code look good? Type 'y' to proceed to the next task, or any other key to stop: ")
            
            if user_approval.lower().strip() == 'y':
                print("\nProceeding to the next task in the plan...")
                continue # Moves to the next task loop in the JSON
            else:
                print("\nStopping conductor. You can run the script again later to resume.")
                break # Stops the script entirely
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error: The agent crashed or was interrupted: {e}")
            break

if __name__ == "__main__":
    main()
import subprocess

def run_script(script_path, conda_env):
    conda_path = r'C:\\Users\\lucia\\anaconda3\\Scripts\\conda.exe'  # Replace with the actual path
    cmd = [conda_path, 'run', '--no-capture-output', '-n', conda_env, 'python', script_path]
    subprocess.Popen(cmd, shell=True)


# Specify the path to the virtual environments for each script
behavioral_task_env = "psychopy"
external_device_env = "tmsi"

# Run scripts in their respective environments
run_script('scripts\\smts.py', behavioral_task_env)
run_script('scripts\\stream_lsl_eeg.py', external_device_env)

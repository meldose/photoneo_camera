import subprocess

def run_pointcloud_script():
    """
    Executes the main point cloud and shape detection script.
    """
    try:
        # Call the main script
        subprocess.run(["python", "infinite_with_shaped_objects_detection.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while executing the script: {e}")
    except FileNotFoundError:
        print("The main script file 'main_script.py' was not found. Please ensure it is in the same directory.")

if __name__ == "__main__":
    print("Starting the point cloud and shape detection process...")
    run_pointcloud_script()
    print("Process completed.")

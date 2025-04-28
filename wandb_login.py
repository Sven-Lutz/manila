import wandb
import os
from dotenv import load_dotenv

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

def login_wandb():
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        print("Login successful!")
    else:
        print("WANDB_API_KEY not found in .env file. Please add it to login to Weights and Biases.")

if __name__ == "__main__":
    login_wandb()

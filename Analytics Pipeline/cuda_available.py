import torch 

def check_cuda_availaibility():

    # Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID 
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Replace 1 with the GPU ID you want to use 


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Running on device: {device}") 

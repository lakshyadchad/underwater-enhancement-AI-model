import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time

# Import your custom modules from Step 1 and Step 2
from model_architecture import UNetGenerator
from custom_loss import UnderwaterLoss

# --- CONFIGURATION (The Knobs) ---
config = {
    "TRAIN_DIR_A": "dataset_final/trainA", # Murky Input
    "TRAIN_DIR_B": "dataset_final/trainB", # Clear Target
    "BATCH_SIZE": 16,          # Reduce to 4 or 2 if you run out of Memory (OOM)
    "LEARNING_RATE": 0.0002,  # Standard for GANs
    "EPOCHS": 100,            # How many times to show the whole dataset
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SAVE_DIR": "checkpoints/", # Where to save the trained brain
}

# Create checkpoint directory if it doesn't exist
os.makedirs(config["SAVE_DIR"], exist_ok=True)

class UnderwaterDataset(Dataset):
    def __init__(self, dir_A, dir_B):
        self.dir_A = dir_A
        self.dir_B = dir_B
        # Get list of filenames (assuming filenames are identical in both folders)
        self.image_names = sorted(os.listdir(dir_A))
        
        # Standard Transformations: Resize to 256x256 and Convert to Tensor
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Converts 0-255 to 0.0-1.0
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 1. Get the filename
        img_name = self.image_names[idx]
        
        # 2. Construct full paths
        path_A = os.path.join(self.dir_A, img_name)
        path_B = os.path.join(self.dir_B, img_name)
        
        # 3. Load Images
        img_A = Image.open(path_A).convert("RGB") # Murky
        img_B = Image.open(path_B).convert("RGB") # Clear
        
        # 4. Apply Transforms
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        
        return img_A, img_B

def main():
    print(f"--- Starting Training on {config['DEVICE']} ---")

    # 1. Prepare Data
    dataset = UnderwaterDataset(config["TRAIN_DIR_A"], config["TRAIN_DIR_B"])
    loader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=2)
    
    print(f"Images found: {len(dataset)}")

    # 2. Initialize Model
    model = UNetGenerator().to(config["DEVICE"])

    # 3. Initialize Loss Function
    criterion = UnderwaterLoss().to(config["DEVICE"])

    # 4. Initialize Optimizer
    # Betas (0.5, 0.999) are standard for GAN-like training
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"], betas=(0.5, 0.999))

    # --- START LOOP ---
    for epoch in range(config["EPOCHS"]):
        start_time = time.time()
        epoch_loss = 0
        
        model.train() # Set model to training mode
        
        for i, (murky, clear) in enumerate(loader):
            # Move data to GPU
            murky = murky.to(config["DEVICE"])
            clear = clear.to(config["DEVICE"])
            
            # A. Zero Gradients (Clear previous step's calculations)
            optimizer.zero_grad()
            
            # B. Forward Pass (The AI guesses)
            fake_clear = model(murky)
            
            # C. Calculate Loss (How bad was the guess?)
            loss = criterion(fake_clear, clear)
            
            # D. Backward Pass (Calculate corrections)
            loss.backward()
            
            # E. Optimization Step (Update weights)
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Print progress every 50 batches
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{config['EPOCHS']}] Batch {i}: Loss = {loss.item():.4f}")

        # --- END OF EPOCH ---
        avg_loss = epoch_loss / len(loader)
        time_taken = time.time() - start_time
        print(f"--- Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}. Time: {time_taken:.1f}s ---")
        
        # Save Model Checkpoint every 5 Epochs
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(config["SAVE_DIR"], f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")

    print("Training Complete.")

if __name__ == "__main__":
    main()





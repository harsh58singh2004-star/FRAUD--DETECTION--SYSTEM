from model import train_model
if __name__ == "__main__":
    result = train_model()
    print("Training complete")
    for key, value in result.items(): print(f"{key}: {value}")

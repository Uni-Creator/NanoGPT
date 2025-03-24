# NANOGPT
![GitHub Repo stars](https://img.shields.io/github/stars/Uni-Creator/NanoGPT?style=social)  ![GitHub forks](https://img.shields.io/github/forks/Uni-Creator/NanoGPT?style=social)

## 📌 Overview
The **NANOGPT** project is a lightweight implementation of GPT-style language models. It processes text data and can generate coherent text sequences based on a trained model. The project is designed to be simple and efficient while maintaining flexibility for experimentation.

## 🚀 Features
- **Lightweight GPT Model**: Efficient architecture for text generation.
- **Pretrained Model Support**: Load existing `model.pth` for inference.
- **Customizable Training**: Train on different text datasets.
- **Evaluation & Testing**: Evaluate performance using test scripts.
- **Minimal Dependencies**: Simple setup without heavy frameworks.

## 🏗️ Tech Stack
- **Python**
- **PyTorch** (for model training)
- **NumPy** (for data processing)
- **Torchvision** (for potential dataset handling)
- **Matplotlib** (for visualization)

## 📂 Project Structure
```
NANOGPT/
│── __pycache__/              # Cached Python files
│── main.py                   # Loads trained model and generates text
│── model.pth                 # Pretrained model checkpoint
│── Nature_of_Code.pdf        # Reference material for training data
│── shakes.txt                # Shakespeare dataset used for training
│── test.py                   # Testing script for evaluation
│── trainer.py                # Model training script
│── README.md                 # Project documentation
```

## 📦 Installation & Setup
1. **Clone the repository**
   ```sh
   git clone https://github.com/Uni-Creator/NanoGPT.git
   cd NANOGPT
   ```
2. **Install dependencies**
   ```sh
   pip install torch numpy matplotlib
   ```
3. **Train the model (if needed)**
   ```sh
   python trainer.py
   ```
4. **Run the model for text generation**
   ```sh
   python main.py
   ```

## 📊 How It Works
1. The model loads a pretrained `model.pth` or trains from scratch.
2. It processes an input text prompt.
3. The model generates a sequence of text based on learned patterns.
4. The output text is displayed and can be saved.

## 🛠️ Future Improvements
- Implement Transformer-based architecture for better efficiency.
- Expand dataset for broader language capabilities.
- Create an interactive web-based demo.

## 🤝 Contributing
Contributions are welcome! Feel free to open an **issue** or submit a **pull request**.

## 📄 License
This project is licensed under the **MIT License**.

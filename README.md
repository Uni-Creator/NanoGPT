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
   cd NanoGPT
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



# NanoGPT - Shakespeare Text Generation

![GitHub Repo stars](https://img.shields.io/github/stars/Uni-Creator/NanoGPT?style=social)  ![GitHub forks](https://img.shields.io/github/forks/Uni-Creator/NanoGPT?style=social)

## Overview
NanoGPT is a simple character-level transformer model built from scratch to generate Shakespearean-style text. It uses a custom tokenizer and trains on a dataset extracted from `shakes.txt`. The model is implemented in PyTorch and supports GPU acceleration.

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
### Prerequisites
Ensure you have Python 3.8+ and PyTorch installed. If not, install PyTorch using:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

1. **Clone the repository**
   ```sh
   git clone https://github.com/Uni-Creator/NanoGPT.git
   cd NanoGPT
   ```
2. **Install dependencies**
   ```sh
   pip install torch numpy matplotlib
   ```
3. **Train the model (if needed)**
   ```sh
   python trainer.py
   ```

## Usage
### Train the Model
To train the model from scratch, run:
```sh
python trainer.py
```
This will generate a `model.pth` file containing the trained weights.

### Generate Text
To generate text using the trained model, run:
```sh
python main.py
```
You will be prompted to enter a starting text, and the model will generate Shakespearean-style text based on your input.

## Example Output
```
Enter text: Enter BERTRAM, the COUNTESS of Rousillon, HELENA, and LAFEU, all in black.
Generated text:
Helena. And you, my lord, sir, captains again.
First Lord. None you shall healt make royal he did
Of daughter! Be thither was which
now wars; it in fither no fetters, or poor him appr.
```

## Customization
- Modify `trainer.py` to change model architecture, training hyperparameters, or dataset.
- Adjust `main.py` to refine text generation.

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



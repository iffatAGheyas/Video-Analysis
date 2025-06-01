# Video‐Based Customer Sentiment Classification

A proof‐of‐concept pipeline that classifies short video clips of customers as **happy** or **angry** using a lightweight CNN + LSTM model in PyTorch. All code is contained in `VideoAnalysis.ipynb`, and the trained model checkpoint is stored in the `Model` folder.

---

## 🚀 Overview

This repository demonstrates an end‐to‐end workflow for:

1. **Collecting** 42 short video clips (21 “happy,” 21 “angry”) from Storyblocks.  
2. **Preparing** a balanced dataset and performing an 80%/10%/10% stratified split into training, validation, and test sets.  
3. **Implementing** a custom PyTorch `VideoDataset` that samples 10 frames from the last 5 seconds of each clip, resizes them to 224×224, and normalises to ImageNet statistics.  
4. **Building** a model that leverages an ImageNet‐pretrained MobileNetV2 backbone (frozen) plus a lightweight LSTM for temporal aggregation.  
5. **Training** for a small number of epochs (5) and automatically saving the checkpoint with the best validation accuracy (`cnn_lstm_best_val.pth`).  
6. **Running inference** on a held‐out TestSet of completely new “happy”/”angry” clips, visualising sampled frames, ground‐truth vs. predicted labels, and saving results in a multi‐page PDF + individual PNGs.  

> **Note:** All video clips used for training and testing were collected from Storyblocks (royalty‐free).  

---

## 📂 Repository Structure

![image](https://github.com/user-attachments/assets/f3721a16-43ab-40f8-a7ea-50fecd714cd5)

![image](https://github.com/user-attachments/assets/c815f6c5-5428-4afb-b8ac-17eecd3e8793)
![image](https://github.com/user-attachments/assets/4a6b0df8-94d4-4fab-a5e6-5f02715ce386)




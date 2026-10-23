# DETECT-AI
DetectAI is a web-based system that detects whether an image or video is real or AI-manipulated (a deepfake). It combines a CNN (Xception/EfficientNetB0 with transfer learning) with a Random Forest classifier trained on forensic features. The two models’ scores are merged into one verdict with a confidence percentage. 

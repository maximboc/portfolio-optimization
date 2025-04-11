# Financial Portfolio Optimization Project

## Introduction

This project applies methods and tools from search problems, constraint programming (CSP), and advanced logical reasoning (SAT/SMT) to real-world financial portfolio optimization. We aim to develop a complete project from modeling to operational solution, focusing on selecting investments that maximize expected return under various constraints.

## Problem Statement

Select a set of investments (stocks, assets) that maximize expected return for a given risk level, adhering to constraints such as maximum budget and sector limits. This problem can be modeled using constraint programming (CP) or mixed-integer linear programming (MILP) to decide the fractions of capital allocated to each asset. Modern solvers can efficiently solve this type of problem, providing optimal solutions that respect risk management constraints.

**References:**
- Markowitz (1952), *Portfolio Selection* – mean/variance model.
- StackOverflow (2022) – formulation of a portfolio in CP-SAT (OR-Tools).
- Michalewicz & Fogel (2000), *How to Solve It: Modern Heuristics* – chapter on financial optimization.
- [Additional Resources](https://drive.google.com/file/d/1KPokq-5Z_aj_T5ysXyqnFebaoefpKU-6/view?usp=sharing)

## Our Project

We are students from EPITA's AI and Data major working on a portfolio optimization project for our 'Programmation par Contraintes' course.
Our team members are:
- Aurélien Daudin
- Maxim Bocquillion
- Khaled Mili
- Mateo Lelong
- Samy Yacef

### Implemented Models

- Markowitz Model - QP
- Markowitz Model - SLSQP
- Markowitz Model - GA
- CVaR with Constraints
- Markowitz Model with Additional Cardinality Constraints

### Additional Features

We have also added a chatbot called Aziz, The Financial ChatBro, to answer financial questions using mistralai/Mixtral-8x7B-Instruct-v0.1 from Hugging Face.

### Technologies Used

- **Frontend**: Streamlit - Python
- **Backend**: Python
- **Libraries**: yfinance, streamlit, langchain_huggingface, scipy, pandas, pygad, cvxpy, gekko

### Project Structure

- `src`: source code
- `src/models/`: Source code for the models
- `src/front/`: Streamlit function management
- `src/pages/`: Additional pages for Streamlit
- `src/utils/`: Utility functions and helpers
- `doc/report.pdf`: Report ofn the project with math fundations (https://www.overleaf.com/read/ppvbwgzdhtjq#411a3c)
- `doc/slides.pdf`: Slides presentation

### Usage Instructions

1. **Set Up the Chatbot**:
   - Create a Hugging Face token and add it to an `.env` file in the `llm/` directory:
     ```bash
     HF_TOKEN='{ADD YOUR TOKEN HERE}'
     ```

2. **Launch the Script**:
   - Run the following command to start the project:
     ```bash
     ./launch_project.sh
     ```

3. **Access the Application**:
   - Open [http://0.0.0.0:8501/](http://0.0.0.0:8501/) in your browser to view the application.

This is what you should get :

https://github.com/user-attachments/assets/c9f26eba-08a9-46ad-9b67-9b5a34380ed9

#!/bin/bash

docker build -t streamlit-app .

docker run -p 8501:8501 streamlit-app

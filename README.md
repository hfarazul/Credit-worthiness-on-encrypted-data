---
title: Encrypted Credit Card Approval Prediction Using Fully Homomorphic Encryption
emoji: 💳 ✅
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 3.40.0
app_file: app.py
pinned: true
tags:
  - FHE
  - PPML
  - privacy
  - privacy preserving machine learning
  - credit card approval
  - credit score
  - homomorphic encryption
  - security
python_version: 3.10.11
---

# Credit card approval using FHE

## Run the application locally

### Install the dependencies

First, create a virtual env and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then, install the required packages:

```bash
pip3 install pip --upgrade
pip3 install -U pip wheel setuptools --ignore-installed
pip3 install -r requirements.txt --ignore-installed
```

The above steps should only be done once.

## Run the app 

In a terminal, run:

```bash
source .venv/bin/activate
python3 app.py
```

## Interact with the application

Open the given URL link (search for a line like `Running on local URL:  http://127.0.0.1:8888/`).


## Development

If the model, data-set or pre/post-processing is modified locally, it is possible to initialize, 
fit and compile the model all at once using the following command. It will also save the deployment 
files at the right places in order to make sure the app works properly :

```bash
python3 development.py
```

# Pengun

The goal of this project is to develop a generative AI system capable of producing new guitar chords and original music compositions. Using Variational Autoencoders (VAEs), we aim to teach the model to recognize patterns in music, particularly guitar chord progressions and musical structures. The objective is to create a tool that allows users to generate fresh musical ideas, with the system functioning as a partner in the creative process.

## Team Members and Roles

* [Hamin Hong (Solo Project)](https://github.com/haminhong/CIS641-HW2-Hong)

## Table of Contents

- [Pengun](#pengun)
  - [Team Members and Roles](#team-members-and-roles)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Running the Project](#running-the-project)
    - [1. Data Preprocessing](#1-data-preprocessing)
    - [2. Training the VAE Model](#2-training-the-vae-model)
    - [3. Generating Music](#3-generating-music)
  - [Project Structure](#project-structure)
  - [Dependencies](#dependencies)
  - [Usage Examples](#usage-examples)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Prerequisites

Before you begin, ensure you have met the following requirements:

### Software Requirements

- **Operating System:** Windows, macOS, or Linux
- **Python:** Version 3.8 or higher

### Hardware Requirements

- **Processor:** Intel i5 or equivalent
- **Memory:** 8 GB RAM (16 GB recommended for training)
- **Storage:** At least 10 GB of free space
- **GPU:** NVIDIA GPU with CUDA support (optional but recommended for faster training)

### Libraries and Tools

- **Python Libraries:**
  - [NumPy](https://numpy.org/)
  - [Pandas](https://pandas.pydata.org/)
  - [PyTorch](https://pytorch.org/)
  - [PrettyMIDI](https://github.com/craffel/pretty-midi)
  - [Matplotlib](https://matplotlib.org/) (optional, for visualizations)
  - [Jupyter Notebook](https://jupyter.org/) (optional, for interactive development)

- **Other Tools:**
  - [Git](https://git-scm.com/) for version control

## Installation

Follow these steps to set up the Pengun project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/haminhong/Pengun.git
cd Pengun

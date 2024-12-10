# Pengun Team / AI Music Generation

The goal of this project is to develop a generative AI system capable of producing original music compositions for inspiration. Using Variational Autoencoders (VAEs), we're going to aim to teach the model to recognize patterns in music, particularly piano chord progressions. The objective is to create a tool that help users to generate fresh musical ideas, with the system functioning as a partner in the creative process.

## Team Members and Roles

* [Hamin Hong](https://github.com/haminhong/CIS641-HW2-Hong)

## Table of Contents

### Software Requirements

- **Operating System:** Windows, macOS, or Linux
- **Python:** Version 3.8 or higher


### Libraries and Tools

- **Python Libraries:**
  - [NumPy](https://numpy.org/)
  - [Pandas](https://pandas.pydata.org/)
  - [PyTorch](https://pytorch.org/)
  - [PrettyMIDI](https://github.com/craffel/pretty-midi)
  - [Matplotlib](https://matplotlib.org/) (optional, for visualizations)
  - [Jupyter Notebook](https://jupyter.org/) 

  use the Requirements.txt to install dependencies. 

- **Other Tools:**
  - [Git](https://git-scm.com/) for version control

## Installation


### 1. Clone the Repository

    git clone https://github.com/haminhong/GVSU-CIS641-Pengun.git

### 2. Get the dataset.  

    download the dataset from
    https://www.kaggle.com/datasets/imsparsh/lakh-midi-clean
    extract the files into clean_midi folder under src.

### 3. Follow the notebooks in order 

    After cloning the repository, follow the notebooks in order of 
        Load Instruments
        data preprocessor
        MusicVAE Model and training
        Reconstruction
    


# Attribution and References

- Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". [PhD Thesis, 2016](http://colinraffel.com/publications/thesis.pdf).

- Adam Roberts, Jesse Engel, Colin Raffel, Curtis Hawthorne, Douglas Eck, [A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music](https://arxiv.org/abs/1803.05428)

- [Librosa](https://doi.org/10.5281/zenodo.591533): Audio and music signal analysis in python

<h1 align="center">Generative Audio Ethics</h1>


This repo contains utilities for auditing audio & speech datasets used in generative AI models. 

0. Install MiniConda if necessary (this requires a shell restart). You can check whether you already have an installation with the shell command `conda`. If you get an error, run:
   ```
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

1. Create conda environment:
   ```
   conda create -y -n gen-audio-ethics python=3.9
   source activate gen-audio-ethics
   ```

   If you want to use Jupyter, run the following to add your conda environment as a kernel:
   ```
   conda install -y -c conda-forge jupyterlab
   conda install -y -c anaconda ipykernel
   python -m ipykernel install --user --name=gen-audio-ethics
   ```

2. Clone repository:
   ```
   git clone https://github.com/anniejchu/gen-audio-ethics.git
   ```

3. Install dependencies:
   ```
   cd gen-audio-ethics
   python -m pip install -r requirements.txt
   ```
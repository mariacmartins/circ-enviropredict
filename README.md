# circ-EnviroPredict

## A Machine Learning tool that allows the prediction of the possible involvement of circRNAs with abiotic stress. 🧬

Find out if a circRNA is potentially involved in some type of abiotic stress using its biological sequence.

**Web application developed with Streamlit.**

### Running circ-EnviroPredict locally (Setup):
```
Clone the repository, enter the created folder and run in Anaconda prompt:
$ conda env create --file environment.yml
$ conda activate circ_enviropredict
$ streamlit run app.py
```

To deactivate the virtual environment, run the command: `conda deactivate`

After the initial configuration, there is no need to create the virtual env again. To run the application locally later, simply enter in the project folder and run the commands:

```
$ conda activate circ_enviropredict
$ streamlit run app.py
```

### 📁 Project Structure:

- `app.py`: Main app file, responsible to run the web application.
- `environment.yml`: Environment configuration file.
- `requirements.txt`: Libs needed for the project.
- `models/`: Trained models (models are serialized using joblib).
- `notebooks/`: Jupyter notebooks containing exploratory analysis and evaluation of trained models.
- `python code/`: Python codes used to create models.
- `tests/`: Unit tests.
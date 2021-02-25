# Assignment 6


## Getting Started

You can start by using git pull on this repo

### Prerequisites

```
Python > 3.6
flask
pandas
altair
```

### Installation
Prerequisites can be installed using pip:
```
pip install flask
pip install pandas
pip install altair
```

Or using the conda environment
``` bash
conda env create -f environment.yml
conda activate icvassboassignment6
```

### Task 1 Time Plotter
To run the program, type
```
python web_visualization.py
```


Program have two methods:
1. plot_reported_cases() that takes three optional arguments:
                        - county name (if not chosen, it will plot all)
                        - Start date in form of string: day/month/year (21/02/2020 if not given)
                        - End date in form of string: day/month/year (24/11/2020 if not given)
This method with plot a bar plot

2. plot_comulative_cases() that takes three optional arguments:
                        - country name (else it will plot all 11)
                        - Start date in form: day/month/year (21/02/2020 if not given)
                        - End date in form: day/month/year (24/11/2020 if not given)
This method with plot a line plot

Counties user can choose from:
- Adger
- Innlandet
- Møre og Romsdal
- Nordland
- Oslo
- Rogland
- Troms og Finnmark
- Trøndelag
- Vestfold og Telemark
- Vestland
- Viken


### Task 2 
To run the program, type
```
python web_visualization.py
```
After that you will need to go to the local host: http://127.0.0.1:5000/task2 to
see results, or click on button right on http://127.0.0.1:5000/


### Task 3 
To run the program, type
```
python web_visualization.py
```
After that you will need to go to the local host: http://127.0.0.1:5000/dropDown to
see results, or click on button right on http://127.0.0.1:5000/


**Program Instructions**  

Install Dependencies
```Bash
pip install -r requirements.txt
```

**Dataset Download**
Go to https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011 and download the dataset.
Name the directory data_wikipedia and place it in the same directory as the program files.

For boolean flags, use --flagname to set to True and exclude flag altogether to set to False:
```Bash
python program.py --flagname    # "flagname" is true
python program.py               # "flagname" is false
```
**All Arguments are optional**  
**Question 1**  
Example Execution
```Bash
python wikipedia_processing.py --zipf --tokenize --stopwords --stemming --invertedindex
```

**Question 2**  
Example Execution
Sample Command Line Use
```Bash
python elias_coding.py [-h] --alg ALG (--encode | --decode) numbers
```

**When entering list of numbers there should be no spaces in the input**  
Example Execution
```Bash
python elias_coding.py --alg gamma --encode [7,25,69,1254,11000102]
python elias_coding.py --alg delta --decode [0100,01111,-2,01111,0100]
```

**Program Instructions**  
Sample Command Line Use (all fields are required)  
**No spaces in the nodes list**  
Sample Command Line Use
```Bash
python page_rank.py [-h] --maxiteration MAXITERATION --lambda LAMBDA_ --thr THR --nodes NODES
``` 
Example Execution
```Bash
python page_rank.py --maxiteration 20 --lambda 0.25 --thr 0.1 --nodes [5,100,50]
```

Sample Command Line Use
```Bash
python noisy_channel.py [-h] [--correct] words
python noisy_channel.py [-h] [--proba] words
```

**Exactly one Argument is required, either --correct or --proba**  
Example Execution
```Bash
python noisy_channel.py --correct advertice univercity university iimprove
python noisy_channel.py --proba advertise computer algorithm medicine874r
```


**Program Instructions**  
Sample Command Line Use
```Bash
python wikipedia_processing.py [-h] [--zipf] [--tokenize] [--stopwords] [--stemming] [--invertedindex]
```
For boolean flags, use --flagname to set to True and exclude flag altogether to set to False:
```Bash
python program.py --flagname    # "flagname" is true
python program.py               # "flagname" is false
**All Arguments are optional**  
Example Execution
```Bash
python wikipedia_processing.py --zipf --tokenize --stopwords --stemming --invertedindex
```
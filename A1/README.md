**Program Instructions**  

Install Dependencies
```Bash
pip install -r requirements.txt
```

Each question has its own README.md file with instructions on how to run the program in its directory.

**Put all URLs in quotes to avoid OS errors**

For boolean flags, use --flagname to set to True and exclude flag altogether to set to False:
```Bash
python program.py --flagname "https://www.example.com"  # "flagname" is true
python program.py "https://www.example.com"             # "flagname" is false
```

---

To run Question 1 
```Bash
cd "Assignment 1 Question 1/src"
python webcrawler1.py --maxdepth 3 --rewrite --verbose "https://stackoverflow.com/questions/58146520/crawling-and-scraping-random-websites"
```

To run Question 2
```Bash
cd "Assignment 1 Question 2/src"
python webcrawler2.py "https://scholar.google.ca/citations?user=86V0RSQAAAAJ&hl=en&oi=ao"
```

To run Question 3
```Bash
cd "Assignment 1 Question 3/src"
python webcrawler3.py "https://uwaterloo.ca/canadian-index-wellbeing/"
```
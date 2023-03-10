**Program Instructions**  

Sample Command Line Use
```Bash
python webcrawler1.py [-h] [--maxdepth MAXDEPTH] [--rewrite] [--verbose] initialURL
```
**Put all URLs in quotes to avoid OS errors**

For boolean flags, use --flagname to set to True and exclude flag altogether to set to False:
```Bash
 webcrawler1.py --maxdepth 3 --rewrite "https://example.com" # "rewrite" is true
 webcrawler1.py --maxdepth 3 "https://example.com"           # "rewrite" is false
```

Example Execution
```Bash
python webcrawler1.py --maxdepth 3 --rewrite --verbose "https://stackoverflow.com/questions/58146520/crawling-and-scraping-random-websites"
```
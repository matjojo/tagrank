# TagRank
is a hydrus API script that uses trueskill to figure out which tags you like best.
By making you choose which of two files you like more, over and over again, the trueskill system will figure out which tags you like most.
It does this by employing the same ranking algorithm that Microsoft uses for its Xbox online games.

TagRank will show you pairs of files, over and over again.
The more comparisons you make the more it learns your preferences.
You can stop at any time by pressing the `ESCape` key, your progress will be saved.
Press the `left arrow` or `A key` if you prefer the left image, the `right arrow` or `D key` for the right, and the `down arrow` or `S key` if there is no clear winner.
To go back one image pair, press `Backspace` or the `R key`. If you need to open the files externally to zoom in or pan you can press the `O key`. This will open the two files in the default program you have for that file.

TrueSkill uses these comparisons to create normal distributions for the "quality" of each tag, and a confidence score that says how sure it is of these results.
TagRank that uses these results to create a representation of this data.

When you are done rating games TagRank will show you the top 20 tags and their skill distributions.
The more to the right the distribution for a tag is the better, and the higher it is the more sure TagRank is of that ranking. 

If you want to do more with this data you can read it from the `ratings.json` file that TagRank creates.
This is a json list of `[tag_name, [mu, sigma]]` objects.
`mu` and `sigma` are the parameters for the normal distribution of that tags ranking.

TagRank stores a list of your previous comparisons in the `comparisons.json` file. It contains a list of lists with two file ids. First the winning id, then the losing id. It is possible that some pairs are in this list multiple times, and even in different orders. Since the list is in-order the last comparison between two file ids is the most recent. 

## Installation
- Clone the repository or download the repository in another way.
- make sure that you have python version 3.9 or higher installed.
- install the requirements in requirements.txt using pip.
- - For example, with `pip install requirements.txt`
- - Or with `pip install requests requests PySide6 matplotlib numpy scipy trueskill hydrus_api>=5.0.0`
- Now you can run main.py.


## Post-installation setup
- TagRank has a small number of configuration files that it will guide you in creating.
- Running main.py without setting up anything else will tell you what to do next.
- In short, you need the following files in the working directory of this program.
- - FILES_PATH: a file that contains the path to your client_files directory.
- - ACCESS_KEY: a file that contains the hydrus API access key.
- - URL: (optional) a file that contains the url that TagRank can access the hydrus API with.
- - SEARCH_QUERY: (optional) a file that contains, line delimited, the search query that TagRank will use to find files to compare. If not provided it will be created and a default query will be inserted for you.
- For all of the above it holds that just running main.py and letting it figure out what it needs from you is easier than trying to do it beforehand.
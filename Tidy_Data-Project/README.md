# Tidy Data Project

## Welcome to my *"Tidy Data Project,"* which aims to show the process and importance of organizing untidy data in a standardized structure to make collected values meaningful and digestible! 😄
## In this project, we will use and observe data from 1976 to 2017 concerning the federal spending on R&D for various government subsidiaries.
---

But before we step foot into the project... *What is tidy data and why is it important?* 😯

**Tidy data** is a when the values of a dataset are organized in a standardized manner. This creates a more decipherable dataset that can be interpreted universally and easily converted into other visualization tools, such as graphs and tables, for further and specified analyses. Hadley Wickham puts it best in her article [*Tidy Data*](https://vita.had.co.nz/papers/tidy-data.pdf):
> The tidy data standard has been designed to facilitate initial exploration and analysis of the data, and to simplify the development of data analysis tools that work well together... Tidy datasets and tidy tools work hand in hand to make data analysis easier, allowing you to focus on the interesting domain problem, not on the uninteresting logistics of data.

**Princples of Tidy Data** ⚡
+ Each variable forms a column.
+ Each observation forms a row.
+ Each type of observational unit forms a table.
---

🌟Now that we know the what and why behind tidy data, let's get straight into the the process! 🌟

🌱 **Prep List** 🍃
<br>Ensure you have the following prepared before you go:
+ an environment that can read .ipynb files
+ an environment or distribution that can `import pandas as pd`

### Downloading and moving the dataset 💫 

1. Click on the link to download the dataset locally onto your device:
[Federal R&D Spending as .csv file](https://canvas.nd.edu/courses/108340/files/4772293?wrap=1)
2. Download this project's script onto your device by clicking the download icon near the top right corner of the screen <image src="https://static.vecteezy.com/system/resources/previews/019/879/209/non_2x/download-button-on-transparent-background-free-png.png" alt="image" width="21"/>
3. Open File Explorer (Windows) <image src="https://static.wikia.nocookie.net/windows/images/0/04/File_Explorer_Icon.png/revision/latest?cb=20240208004644" alt="image" width="21"/> or Finder (Mac) <image src="https://upload.wikimedia.org/wikipedia/commons/c/c9/Finder_Icon_macOS_Big_Sur.png" alt="image" width="21"/>
4. Locate both files (the dataset and the script) and place them into a new folder named appropriately together (ex. TidyDataProject)
   <br>- This will place the data file into the same directory as the script and allow the script to access the data

### Running the notebook 🌀

1. Open an environment that can be used to open and read a .ipynb file such as:
-  Jupyter Notebook <image src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/1200px-Jupyter_logo.svg.png" alt="image" width="21"/>
- Google Colab <image src="https://img.icons8.com/?size=512&id=lOqoeP2Zy02f&format=png" alt="image" width="21"/>
- Visual Studio Code <image src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Visual_Studio_Code_1.35_icon.svg/2048px-Visual_Studio_Code_1.35_icon.svg.png" alt="image" width="21"/>
- Spyder IDE <image src="https://img.icons8.com/?size=512&id=0S1Hoidfnk7H&format=png" alt="image" width="21"/>
2. Open the folder that contains both the data file and the script
3. Open the notebook file within the environment and:
  - install dependecy through code `import pandas as pd`
  - call dataset into script via `df_fedrd = pd.read_csv('fed_rd_year&gdp.csv')`
    <br>(the two steps above should be included in the second code box within the notebook)
4. Check that the dataset has been properly accessed by running the third code box containing `print(df_fedrd.head())`
- if done properly, you should be able to see the first five rows of the dataset
- if a  `FileNotFound` error appears, make sure that you downloaded the proper file and that the file is located in the same folder as the script

**Now you should be ready to run the following cells as directed by the markdowns and explore data tidying!** 🥳🎊

---
Here is a reference sheet for the department abbreviations:
| Abbreviation | Department Name |
| --- | --- |
| DOD | Department of Defense |
| NASA | National Aeronautics and Space Administration |
| DOE | Department of Energy |
| HHS | Department of Health and Human Services |
| NIH | National Institute of Health |
| NSF | National Science Foundation |
| USDA | US Department of Agriculture |
| Interior | Department of Interior |
| DOT | Deparment of Transportation |
| EPA | Environmental Protection Agency |
| DOC | Department of Corrections |
| DHS | Department of Homeland Security |
| VA | Department of Veterands Affairs |
| other | other research and development spending |
---
### References & Resources 📖
**Dataset** 🐾<br> 
The dataset was adapted from the [*rfordatascience* GitHub account](https://github.com/rfordatascience/tidytuesday/tree/main/data/2019/2019-02-12) 
<br>Here are some links provided by the respository for more data:
- https://www.aaas.org/programs/r-d-budget-and-policy/historical-trends-federal-rd
- https://www.aaas.org/programs/r-d-budget-and-policy/historical-rd-data

**Tidy Data**🌐
<br>Here are the referneces used to learn about and carry out data tidying through words and code lines. Both links are highly recommended for a clearer look into the art of structured organization in data!

* Information on the role and importance of tidy data was found in Hadley Wickham's article [*Tidy Data*](https://vita.had.co.nz/papers/tidy-data.pdf) published as part of the journal *Journal of Statistical Software*
* Here is an amazing guide to data wrangling using pandas: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
---
### Hope you enjoyed learning and experiencing the data tidying process just as much as I did! 💞


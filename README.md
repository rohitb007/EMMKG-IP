# EMMKG-IP

**About** <br>
A github repository that aims to develop an Event MultiModal Knowledge Graph (EMMKG) based on events in the chosen dataset.

**Dependencies**: <br>
Please run **_pip install <module_name>_** to install the following modules:<br>
rdflib, pandas, requests, imagehash, pydotplus, networkx, wikipedia, nltk, sklearn, beautifulsoup, youtubetranscriptapi

**How to run?** <br>
Run **_python emmkg.py_** on terminal.

**Usage:**<br>
To download the images for each event, bing image downloader library is used. To extract images for each event from the web, we have used the Bing Image Downloader library. Bing Image Downloader is a Python library that can be used to download images from the Bing search engine. Firstly, we installed the library using pip. We then imported the library and created an instance of the downloader. Next, we used the downloader.download() method to extract images. We specified the search term and the number of images to be downloaded.

![i1](https://github.com/rohitb007/EMMKG-IP/assets/55681115/bb445109-75d0-4003-9181-b64fb57b9507)

Here, query is the particular event, count is the number of images to be retrieved. The count can be changed according the user's convenience.

**Output:**

1. For each event, a seperate directory will be created in the current working directory under the name _event_ (for eg. _Fifa World Cup_), containing the images for each event. <br>
2. _ekg.png_ will be generated containing the desired Event Multimodal Knowledge Graph (EMMKG).<br>
3. _output.xml_ will be generated containing all the attributes, classes, properties and multimodalities of the EMMKG.<br>

**Zoomed in snippet of the sample output**
![i4](https://github.com/rohitb007/EMMKG-IP/assets/55681115/2968842c-9ca0-42e6-8a4a-08f64ed1a3c8)

# EMMKG-IP

A github repository that aims to develop an event multimodal knowledge graph (EMMKG) based on events in the chosen dataset.

Usage: 
To download the images for each event, bing image downloader library is used. To extract images for each event from the web, we have used the Bing Image Downloader library. Bing Image Downloader is a Python library that can be used to download images from the Bing search engine. Firstly, we installed the library using pip. We then imported the library and created an instance of the downloader. Next, we used the downloader.download() method to extract images. We specified the search term and the number of images to be downloaded.

![i1](https://github.com/rohitb007/EMMKG-IP/assets/55681115/bb445109-75d0-4003-9181-b64fb57b9507)

Here, query is the particular event, count is the number of images to be retrieved. The count can be changed according the user's convenience. 

Run emmkg.py.

Output:
1. For each event, a seperate folder will be created in the current working directory under the folder name ‘event’ (for eg. Fifa World Cup). containing the images for each event.
2. ekg.png will be generated containing the desired Event Multimodal Knowledge Graph (EMMKG). 
3. output.xml will be generated containing all the attributes, classes, properties and multimodalities of the EMMKG.

Zoomed in snippet of the sample output
![i4](https://github.com/rohitb007/EMMKG-IP/assets/55681115/2968842c-9ca0-42e6-8a4a-08f64ed1a3c8)








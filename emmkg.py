from bs4 import BeautifulSoup
import pandas as pd
import re
import requests
from PIL import Image
import imagehash
from rdflib import Graph, RDF, URIRef
from rdflib.namespace import RDF, RDFS
import io
import os
from PIL import Image
import rdflib
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import networkx as nx
import matplotlib.pyplot as plt
import downloader
import urllib.request
import re
import itertools
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import wikipedia
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def csv_to_dict(file):
    '''
    function to convert 2 columns of a csv into a dictionary.
    Paramters: 
        file: csv file
        
    Returns:
        dic: a dictionary
    ''' 
    col1=file.iloc[:,2].values
    col2=file.iloc[:,6].values
    dic={}
    for (a,b) in zip(col1,col2):
        dic[a]=b
    return dic


def similarity(event):
    '''
    function to check similarity between all images in a folder using image hashing.
    Paramters:
        event (str): a string
    Returns: 
        returns void
    ''' 
    # Define the threshold for image similarity
    threshold = 10

    # Load the images
    img_folder = "./"+event
    images = []
    for file in os.listdir(img_folder):
        if file.endswith(".jpg"):
            img_path = os.path.join(img_folder, file)
            img = Image.open(img_path)
            images.append((img, img_path))

    # Compute the hash value for each image
    hashes = []
    for img, img_path in images:
        hash_val = imagehash.average_hash(img)
        hashes.append((hash_val, img_path))

    # Compare the hash values of each image
    similar_images = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hashes[i][0] - hashes[j][0] < threshold:
                if hashes[j][1] not in similar_images:
                    similar_images.append(hashes[j][1])

    # Remove the similar images
    for img_path in similar_images:
        # os.remove(img_path)
        print(img_path)
    


def create_dataframe(matrix, tokens):
    '''
    function to convert a matrix into a dataframe with columns as tokens

    Parameters:
        matrix (list): a list of values
        tokens (list): a list of values
    Returns:
        df (dataframe) : a 2*2 dataframe with columns and tokens
    '''
    doc_names = [f'text{i+1}' for i, _ in enumerate(matrix)]
    df = pd.DataFrame(data=matrix, index=doc_names, columns=tokens)
    return(df)


def text_similarity(text1, text2):
    '''
    function to check similarity between two strings

    Paramters: 
        text1 (str) :a string
        text2 (str) :a string
    Returns:
        val (float): cosine similarity between 2 strings
    '''
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    lemmatizer = WordNetLemmatizer()
    tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
    tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    # Remove stopwords
    stop_words = stopwords.words('english')
    tokens1 = [token for token in tokens1 if token not in stop_words]
    
    
    tokens2 = [token for token in tokens2 if token not in stop_words]
    list1=' '.join(tokens1)
    list2=' '.join(tokens2)

    data=[list1, list2]

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vector_matrix = vectorizer.fit_transform(data)
    similarity = cosine_similarity(vector_matrix)
    df=create_dataframe(similarity,['text1','text2'])
    val = df['text2'].values[0]

    return val

def function_article(search_keyword, content):
    '''
    function to return top 10 videos for a article with on the basis of maximum similarity with the related content of the article

    Parameters:
        search_keyword (str): string with the title of the article
        content (str): a string 
    Returns:
        links (list) : list of top 10 youtube links
    ''' 
    # search_keyword="fifa+world+cup"
    html = urllib.request.urlopen("https://www.youtube.com/results?search_query={}".format(search_keyword))
    video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
    print(video_ids)
    # print(len(video_ids))

    videos={}
    for video_id in video_ids: 
        try:     
            list_=YouTubeTranscriptApi.get_transcript(video_id)
            text=""
            for dict in list_:
                text+=dict['text']+" "
            videos[video_id]={'text':text, 'similarity':text_similarity(text, content)}

        except Exception as e: 
            
            print("-------------------------")
            pass
            


    
    print(len(videos))
    temp = {k: v for k, v in sorted(videos.items(), key=lambda x:x[1]['similarity'], reverse=True)}
    sorted_videos= {k:v for k, v in itertools.islice(temp.items(), 10)}
    print(sorted_videos)
    print(len(sorted_videos))
    links= ['https://www.youtube.com/watch?v=' + item for item in list(sorted_videos.keys())]
    return links

def function(search_keyword, wiki_search):
    '''
    function to return top 10 videos for a event on the basis of maximum similarity with the wikipedia summary of the event

    Parameters:
        search_keyword (str): string with the title of the article
        wiki_search (str): a string
    Returns:
        links (list) : list of top 10 youtube links
    ''' 
    
    html = urllib.request.urlopen("https://www.youtube.com/results?search_query={}".format(search_keyword))
    video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
    print(video_ids)
    

    wiki = wikipedia.page(wiki_search)
    wiki_content = wiki.summary
    print(wiki_content)
    videos={}
    for video_id in video_ids: 
        try:     
            list_=YouTubeTranscriptApi.get_transcript(video_id)
            text=""
            for dict in list_:
                text+=dict['text']+" "
            videos[video_id]={'text':text, 'similarity':text_similarity(text, wiki_content)}

        except Exception as e: 
            
            print("-------------------------")
            pass



    print(len(videos))
    temp = {k: v for k, v in sorted(videos.items(), key=lambda x:x[1]['similarity'], reverse=True)}
    sorted_videos= {k:v for k, v in itertools.islice(temp.items(), 10)}
    print(sorted_videos)
    print(len(sorted_videos))
    links= ['https://www.youtube.com/watch?v=' + item for item in list(sorted_videos.keys())]
    return links



def visualize(g):
    '''
    converts a rdflib graph to an png image
    Parameters:
        g (graph): an rdflib graph
    Returns:
        void
    '''
    stream = io.StringIO()
    rdf2dot(g, stream, opts = {display})
    dg = pydotplus.graph_from_dot_data(stream.getvalue())
    png = dg.create_png()
    display(Image(png))

def fetch_wikidata_api(query):
    '''
    function to return iri of the query from wikidata
    Parameters:
        query(str): a string
    Returns: 
        s (str) :  a string with iri of the query
    '''
    API_ENDPOINT = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'search': query
    }
    r = requests.get(API_ENDPOINT, params = params)
    s='http:'+ r.json()['search'][0]['url']
    return s

EKG = Graph()
df = pd.read_csv('dataset.csv')
df=df.iloc[:500]

Language_dict={}
Event_dict={}
Article_dict={}

Language = URIRef(fetch_wikidata_api('Language'))
Event = URIRef(fetch_wikidata_api('Event'))
Article = URIRef(fetch_wikidata_api('Article'))
Website = URIRef(fetch_wikidata_api('Website'))
based_on=URIRef(fetch_wikidata_api('based on'))
published_in=URIRef(fetch_wikidata_api('published in'))
RP = URIRef(fetch_wikidata_api('image'))
size=URIRef(fetch_wikidata_api('size'))

#EKG
EKG.add((Language, RDF.type, RDFS.Class))
EKG.add((Event, RDF.type, RDFS.Class))
EKG.add((Article, RDF.type, RDFS.Class))
EKG.add((Website, RDF.type, RDFS.Class))

for language in df['Language'].unique():
    Language_dict[language]=fetch_wikidata_api(language)
    EKG.add((URIRef(Language_dict[language]), RDF.type, Language))

for event in df['Event'].unique():
    Event_dict[event]=fetch_wikidata_api(event)
    EKG.add((URIRef(Event_dict[event]), RDF.type, Event))
    
    
    #####images part 
    #downloading images for each event
    count=5
    query=event
    images=downloader.download(query, limit=count, adult_filter_off=True, force_replace=False, timeout=60)
    #dictionary for image and its link
    print(images) 
    
    img_folder = "./"+event
    # similarity(event)
    count=0
    for file in os.listdir(img_folder):
        if count>10:
            break
        img_path = os.path.join(img_folder, file)
        img = Image.open(img_path)
        width, height = img.size
        img_size=f'{width} x {height}'
        filename = os.path.splitext(os.path.basename(file))[0]
        # print(filename)
        
        uri_image=URIRef(images[filename])
        print(filename +" " + uri_image)
        #adding size and relation to the image
        EKG.add((uri_image, RP,URIRef(Event_dict[event] )))
        EKG.add((uri_image, size, rdflib.Literal(img_size)))
        count+=1
    
    ####videos part
    video=URIRef(fetch_wikidata_api('video'))
    keyword=event.replace(' ','+')
    #list for top 10 links related to the event
    links=function(keyword, event)
    print(links)
    for i in links:
        uri_link=URIRef(i)
        EKG.add((uri_link, video, URIRef(Event_dict[event])))

Article_dict=csv_to_dict(df)
j=1
for article in df['Article Title'].unique():
    # running for two articles
    if j<=2:
        url=Article_dict[article]
        page=requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')

        # finding img src links
        img_tags = soup.find_all('img')
        para_tags = soup.find_all('p')
        img_urls=[]
        para_texts=[]
        i=1
        image_dict={}
        for img in img_tags:
            img_urls.append(img['src'])
            url=img['src']
            if url[0]=='h':
                print(url)
                response = requests.get(url)
                folder_path = "./"+'harvard'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                name='image'+str(i)+'.jpg'
                file_path = os.path.join(folder_path, name)
                image_dict[name]=url
                with open(file_path, "wb") as f:
                    f.write(response.content)
                i+=1

            uri_image=URIRef(image_dict[name])
            #adding size and relation to the image
            EKG.add((uri_image, RP,URIRef(Article_dict[article] )))
            EKG.add((uri_image, size, rdflib.Literal(img_size)))
            i+=1
        

        for para in para_tags:
            para_texts.append(para.get_text())
        # video=URIRef(fetch_wikidata_api('video'))
        keyword=article.replace(' ','+')
        #list for top 10 links related to the event
        links=function_article(keyword, para_texts)
        print(links)
        for i in links:
            uri_link=URIRef(i)
            EKG.add((uri_link, video, URIRef(Article_dict[article])))
    
    j+=1


    
i=0
for index, row in df.iterrows():
    if i>20:
        break
    if type(row["Article url"]) is not str or type(row["Website"]) is not str or type(row["Event"]) is not str or type(row["Language"]) is not str:
        continue
    language_URI = URIRef(Language_dict[row['Language']])
    event_URI = URIRef(Event_dict[row['Event']])
    article_URI = URIRef(row["Article url"])
    publisher_URI = URIRef(row["Website"])
    i+=1

    EKG.add((article_URI, RDF.type, Article))
    EKG.add((publisher_URI, RDF.type, Website))
    EKG.add((article_URI, RDF.langString, language_URI))
    EKG.add((article_URI, based_on, event_URI))
    EKG.add((article_URI, published_in, publisher_URI))
    # EKG.add((article_URI,RDF.subject, event_URI))



EKG.serialize(destination='output.xml',format='xml')


G = rdflib_to_networkx_multidigraph(EKG)

# Plot Networkx instance of RDF Graph
pos = nx.spring_layout(G, scale=2)
fig = plt.figure(figsize=(50, 50))
edge_labels = nx.get_edge_attributes(G, 'r')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
nx.draw(G, with_labels=True)
plt.axis('off')
plt.savefig('output.pdf')
#if not in interactive mode for 
plt.show()
import io
import pydotplus
from IPython.display import display, Image
from rdflib.tools.rdf2dot import rdf2dot



visualize(EKG)

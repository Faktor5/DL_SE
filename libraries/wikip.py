#!/usr/bin/env python

"""
Read articles from Wikipedia
vers 0.5

source: https://www.geeksforgeeks.org/how-to-extract-wikipedia-data-in-python/
"""

from bs4 import *
import requests
import re
 
def get_wikipedia(url="https://de.wikipedia.org/wiki/Data_Science"):
    """
    Read articles from Wikipedia
    in: address
    out: paragraphs (filtered), heading, link, images
    """

    # Fetch URL Content
    r = requests.get(url)
     
    # Get body content
    soup = BeautifulSoup(r.text,'html.parser').select('body')[0]
     
    # Initialize variable
    paragraphs = []
    images = []
    link = []
    heading = []
    remaining_content = []
     
    # Iterate through all tags
    for tag in soup.find_all():
         
        # Check each tag name
        # For Paragraph use p tag
        if tag.name=="p":
           
            # use text for fetch the content inside p tag
            paragraphs.append(tag.text)
             
        # For Image use img tag
        elif tag.name=="img":
           
            # Add url and Image source URL
            images.append(url+tag['src'])
             
        # For Anchor use a tag
        elif tag.name=="a":
           
            # convert into string and then check href
            # available in tag or not
            if "href" in str(tag):
               
              # In href, there might be possible url is not there
              # if url is not there
                if "https://en.wikipedia.org/w/" not in str(tag['href']):
                    link.append(url+tag['href'])
                else:
                    link.append(tag['href'])
                     
        # Similarly check for heading
        # Six types of heading are there (H1, H2, H3, H4, H5, H6)
        # check each tag and fetch text
        elif "h" in tag.name:
            if "h1"==tag.name:
                heading.append(tag.text)
            elif "h2"==tag.name:
                heading.append(tag.text)
            elif "h3"==tag.name:
                heading.append(tag.text)
            elif "h4"==tag.name:
                heading.append(tag.text)
            elif "h5"==tag.name:
                heading.append(tag.text)
            else:
                heading.append(tag.text)
                 
        # Remain content will store here
        else:
            remaining_content.append(tag.text)
            
    #filter references like [7]
    p_filter = [ re.sub('\[.*?\]', '', p) for p in paragraphs if len(p)>1]
             
    #print(paragraphs, images, link, heading, remaining_content)
    return p_filter, heading, link, images
    
def get_wikipedia_text(url): 
    """
    get all paragraphs from page as single string. Paragraphs are separated by two  \n
    if link is different from wikipedia text may contain unwanted text.
    in: url
    out: full text
    """
    return  '\n'.join(get_wikipedia(url)[0])

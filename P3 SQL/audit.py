# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:04:32 2017

@author: mnshr
"""

"""
Your task in this exercise has two steps:

- audit the OSMFILE and change the variable 'mapping' to reflect the changes needed to fix 
    the unexpected street types to the appropriate ones in the expected list.
    You have to add mappings only for the actual problems you find in this OSMFILE,
    not a generalized solution, since that may and will depend on the particular area you are auditing.
- write the update_name function, to actually fix the street name.
    The function takes a string with street name as an argument and should return the fixed name
    We have provided a simple test so that you see what exactly is expected
"""
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE = "raleigh-sample.osm"
#OSMFILE = "raleigh_North-Carolina.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
post_re=re.compile(r'^\D*(\d{5}).*', re.IGNORECASE) #^\D*(\d{5}).*

expected = ["Street", "Avenue", "Boulevard", 
            "Drive", "Court", "Place", 
            "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons"]

# UPDATE THIS VARIABLE
mapping = {"St": "Street", "St.": "Street", "street": "Street", 
            "Ave": "Avenue", "Ave.": "Avenue", "Avene": "Avenue", "Avene.": "Avenue",
            "Rd": "Road", "Rd.": "Road", "Pkwy": "Parkway", 'Pkwy.': "Parkway",
            "Ln": "Lane", "Ln.": "Lane", "lane": "Lane",
            "Hwy": "Highway", "Hwy.": "Highway", "HWY": "Highway",
            "Expwy": "Expressway", "Expwy.": "Expressway",
            "Dr": "Drive", "Dr.": "Drive", "Blvd": "Boulevard", "Blvd.": "Boulevard", "N.": "North",
            "Cir": "Circle"}


def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)

def audit_postcode(postcodes, code):
    #if not re.match(r'^\D*(\d{5}).*', code):
    print code
    if not re.match(r'^\d{5}$', code):
        postcodes[code] += 1
        
def is_street_name(elem):
    #print elem.attrib['k']
    return (elem.attrib['k'] == "addr:street")

def is_postcode(elem):
    #print elem.attrib['k']
    return (elem.attrib['k'] == "addr:postcode")

def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    postcodes = defaultdict(int)
    
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
                if is_postcode(tag):
                    #print "Check postcode: ", tag.attrib['v']
                    audit_postcode(postcodes, tag.attrib['v'])
    osm_file.close()
    print '----------------'
    pprint.pprint(postcodes)
    print '----------------'
    return street_types, postcodes


def update_name(name, mapping):
    for word in name.split(" "):
        if word in mapping.keys():
            #print "calling replace for word: ", word, " mapping to: ", mapping[word]
            name = name.replace(word, mapping[word])
    return name

def update_postcode(postcode):
    search = re.match(r'^\D*(\d{5}).*', postcode)
    valid_postcode=search.group(1)
    return valid_postcode

def test_audit(filename):
    st_types, post_types = audit(filename)
    #assert len(st_types) == 3
    #pprint.pprint(dict(st_types))
    print '---printing postcode dict for cleaning---'
    pprint.pprint(dict(post_types))

    for st_type, ways in st_types.iteritems():
        for name in ways:
            better_name = update_name(name, mapping)
            #print name, "=>", better_name
            if name == "West Lexington St.":
                assert better_name == "West Lexington Street"
            if name == "Baldwin Rd.":
                assert better_name == "Baldwin Road"

    for item in post_types:
        cleaned = update_postcode(item)
        print cleaned
        
if __name__ == '__main__':
    test_audit(OSMFILE)
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

#OSMFILE = "raleigh-sample.osm"
OSMFILE = "raleigh_north-carolina.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
post_re=re.compile(r'^\D*(\d{5}).*', re.IGNORECASE) #^\D*(\d{5}).*
phone_re=re.compile(r'\d{3}\s\d{3}\s\d{4}', re.IGNORECASE)

expected = ["Street", "Avenue", "Boulevard", "Extension",
            "Drive", "Court", "Place", "Bypass", "Crossing",
            "Square", "Lane", "Road", "Alley", "West",
            "Trail", "Parkway", "Commons", "Circle", 
            "Run", "Ridge", "Plaza", "Loop", "Crescent"]

# UPDATE THIS VARIABLE
mapping = {"St": "Street", "St.": "Street", "street": "Street", "ST": "Street", "St,": "Street",
            "Ave": "Avenue", "Ave.": "Avenue", "Avene": "Avenue", "Avene.": "Avenue",
            "Rd": "Road", "Rd.": "Road", "Pkwy": "Parkway", 'Pkwy.': "Parkway", "Pky": "Pkwy",
            "Ln": "Lane", "Ln.": "Lane", "lane": "Lane",
            "Hwy": "Highway", "Hwy.": "Highway", "HWY": "Highway",
            "Expwy": "Expressway", "Expwy.": "Expressway",
            "Dr": "Drive", "Dr.": "Drive", "Driver": "Drive", 
            "Blvd": "Boulevard", "Blvd.": "Boulevard", "N.": "North",
            "Cir": "Circle", "CIrcle": "Circle", "Ct": "Court",
            "Ext": "Extension", "LaurelcherryStreet": "Laurelcherry Street",
            "Pl": "Place", "PI": "Place"}

def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

def is_postcode(elem):
    return (elem.attrib['k'] == "addr:postcode")

def is_phone(elem):
    return ((elem.attrib['k']=="phone") or (elem.attrib['k']=="contact:phone"))

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)

def audit_postcode(postcodes, code):
    #if not re.match(r'^\D*(\d{5}).*', code):
    if not re.match(r'^\d{5}$', code):
        postcodes[code] += 1
        
def audit_phone(phone, code):
    if not phone_re.match(code):
        phone[code] += 1
    
def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    postcodes = defaultdict(int)
    phones = defaultdict(int)
    
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
                if is_postcode(tag):
                    #print "Check postcode: ", tag.attrib['v']
                    audit_postcode(postcodes, tag.attrib['v'])
                if is_phone(tag):
                    audit_phone(phones, tag.attrib['v'])
    osm_file.close()
    pprint.pprint(phones)
    return street_types, postcodes, phones


def update_name(name, mapping):
    for word in name.split(" "):
        if word in mapping.keys():
            #print "calling replace for word: ", word, " mapping to: ", mapping[word]
            name = name.replace(word, mapping[word])
    return name

def update_postcode(postcode):
    if post_re.search(postcode):
        found = post_re.match(postcode)
        valid_postcode=found.group(1)
        return valid_postcode
    else:
        # Return a default postcode if no 5 digit number found 
        return "27601"
        
def update_phone(phone):
    if phone:
        # Look for the phone number matching the regular expression
        phone_m=phone_re.match(phone)

        if phone_m is None:
            # substitute hyphens and remove brackets
            if "-" in phone:
                phone = re.sub("-", " ", phone)
            elif "(" in phone or ")" in phone:
                phone = re.sub("[()]", "", phone)
            
            # Search for 10/11 digits phone numbers and add spaces in between area codes
            if re.search(r'\d{10}', phone):
                phone = phone[:3] + " " + phone[3:6] + " " + phone[6:]
            elif re.search(r'\d{11}', phone):
                phone = phone[:1] + " " + phone[1:4] + " " + phone[4:7] + " " + phone[7:]
            # Handle cases where there are last 7 digits grouped 
            elif re.search(r'\s?(\d{1}\s?\d{3}\s?\d{7})', phone):
                phone = phone[:10] + " " + phone[10:] 
                
            # USA code with a + mark
            if re.match(r'\d{3}\s\d{3}\s\d{4}', phone) is not None:
                phone = "+1 " + phone
            elif re.match(r'1\s\d{3}\s\d{3}\s\d{4}', phone) is not None:
                phone = "+" + phone
            
        return phone

def test_audit(filename):
    st_types, post_types, phone_types = audit(filename)
    #assert len(st_types) == 3
    pprint.pprint(dict(st_types))
    #print '---printing postcode & phones dicts for cleaning---'
    pprint.pprint(dict(post_types))
    pprint.pprint(dict(phone_types))
    for item in phone_types:
        cleaned_phone = update_phone(item)
        print cleaned_phone

    for st_type, ways in st_types.iteritems():
        for name in ways:
            better_name = update_name(name, mapping)
            print name, "=>", better_name
            if name == "West Lexington St.":
                assert better_name == "West Lexington Street"
            if name == "Baldwin Rd.":
                assert better_name == "Baldwin Road"

    for item in post_types:
        cleaned = update_postcode(item)
        print cleaned

if __name__ == '__main__':
    test_audit(OSMFILE)
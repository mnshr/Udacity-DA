# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 06:32:39 2017

@author: Manish
"""

"""
Question 1
Given two strings s and t, determine whether some anagram of t is a substring 
of s. For example: if s = "udacity" and t = "ad", then the function returns 
True. Your function definition should look like: question1(s, t) and return a 
boolean True or False.
"""

def char_cnt(string):
    ret = {}
    for char in string:
        ret[char] = ret.get(char, 0) + 1
    return ret

def question1(s, t):
#    print ("-------------Start---------------")
    # Handle corner cases
    if s is None:
        return False

    # Handle invalid input
    if (len(t) > len(s)):
        return False

    # Handle empty string
    if t == '':
        return True
    
    hd = 0
    tl = hd + len(t) - 1
    char_cnt_t = char_cnt(t)
    #print char_cnt_t
    
    #Count the char in array from head to tail
    char_cnt_s = char_cnt(s[hd:tl + 1])
    #print char_cnt_s
    
    while tl < len(s):
        #print 'tail: ', tl, ', head: ', hd, ' len(s): ', len(s)
        #print 'char_cnt_s: ', char_cnt_s, ' || char_cnt_t: ', char_cnt_t
        if char_cnt_t == char_cnt_s:
            return True
        #print 'char_cnt_s[s[hd]]: ', char_cnt_s[s[hd]]
        #Remove the leading char
        if char_cnt_s[s[hd]] == 1:
            del char_cnt_s[s[hd]]
        else:
            char_cnt_s[s[hd]] -= 1
        #print 'char_cnt_s: ', char_cnt_s, ' || char_cnt_t: ', char_cnt_t
        #Move head and tail forward in the string
        hd += 1
        tl += 1
        if tl < len(s):
            #Get the next character
            char_cnt_s[s[tl]] = char_cnt_s.get(s[tl], 0) + 1
        else:
            return False
    return False

def test_q1():
    assert question1('udacity', 'ad') == True
    assert question1('udacity', 'adc') == True
    assert question1('coursera', 'or') == False
    #Corner cases
    assert question1('Coursera', '') == True
    assert question1(None, 'cit') == False
    assert question1(None, None) == False
    assert question1('', '') == True
    print('---Q1---')

"""
Question 2
Given a string a, find the longest palindromic substring contained in a. 
Your function definition should look like question2(a), and return a string.
"""
#Function to check if the input string is Palindrome
def chk_pldrm(s):
    if s is None:
        return False
    #reverse compare
    return s == s[::-1]

def question2(a):
    # No palindromes in None
    if a is None:
        return None
    # Longest palindrome in an empty string is an empty string
    if a == '':
        return ''    
    #print a
    
    l = len(a)
    lp_len, l_idx, r_idx = 0, 0, 0
    for i in xrange(0, l):
        for j in xrange(i + 1, l + 1):
            #Create a substring for longest palindrome check
            lp_str = a[i:j]
            #print substr
            #Check if the current substring is palindrome and the logest so far
            if chk_pldrm(lp_str) and len(lp_str) > lp_len:
                lp_len = len(lp_str)
                l_idx = i
                r_idx = j
                #print 'longest ---> ', longest
                
    #Use the indexs to create the longest substring
    lp = a[l_idx:r_idx]
    return lp

def test_q2(): 
    assert question2('zusnantpr') == 'nan'
    assert question2('ayananaya') == 'ayananaya'
    assert question2("abcCIVVICdef") == 'CIVVIC'
    assert question2('dessertsstressed') == 'dessertsstressed'
    #Corner cases
    assert question2('') == ''
    assert question2(None) is None
    print('---Q2---')

"""
Question 3
Given an undirected graph G, find the minimum spanning tree within G. 
A minimum spanning tree connects all vertices in a graph with the smallest 
possible total weight of edges. Your function should take in and return an 
adjacency list structured like this:

{'A': [('B', 2)],
 'B': [('A', 2), ('C', 5)], 
 'C': [('B', 5)]}
Vertices are represented as unique strings. The function definition should be 
question3(G)
"""
class Graph(object):
    def __init__(self, nodes=None, edges=None):
        if nodes:
            self.nodes=set(nodes)
        else:
            self.nodes = set([])
        self.edges = edges or {}
        

    def insert_edge(self, node_from, node_to, weight):
        self.nodes.add(node_from)
        self.nodes.add(node_to)
        if node_from not in self.edges:
            self.edges[node_from] = set([])
        self.edges[node_from].add((node_to, weight))

    def ret_list(self):
        ret_dict = {}
        for n, es in self.edges.items():
            #print n, es
            ret_dict[n]=list(es)
        return ret_dict

def question3(G):
    if not G:
        return None

    #Dictionary from Graph
    trees = {node: node for node in G}
    #print trees

    #Extract edges and sort by weight of each edge
    edges = sorted([(wt, node1, node2) for node1, node_arr in G.iteritems() for node2, wt in node_arr])

    min_sp_tree = Graph()
    #print 'min_sp_tree: ', min_sp_tree
    #print 'edges: ', edges

    for wt, node1, node2 in edges:
        #print wt, node1, node2
        #print trees[node1], trees[node2]
        #Add nodes to Min Spanning Tree if Node1 and Node2 belong to different trees
        if trees[node1] != trees[node2]:
            trees[node2] = trees[node1]
            min_sp_tree.insert_edge(node1, node2, wt)
    return min_sp_tree.ret_list()

def test_q3():
    a1 = {
        'A': [('B', 2)],
        'B': [('A', 2), ('C', 5)],
        'C': [('B', 5)],
        'D': [('A', 1), ('C', 4)]
    }
    g1 = {
        'A': [('B', 2)],
        'D': [('A', 1), ('C', 4)]
    }
    a2 = {'A': [('B', 7), ('D', 5)],
         'B': [('A', 7), ('C', 8), ('D', 9), ('E', 7)],
         'C': [('B', 8), ('E', 5)],
         'D': [('A', 5), ('B', 9), ('E', 15), ('F', 6)],
         'E': [('B', 7), ('C', 5), ('D', 15), ('F', 8), ('G', 9)],
         'F': [('D', 6), ('E', 8), ('G', 11)],
         'G': [('E', 9), ('F', 11)]}
    
    g2 = {'A': [('D', 5), ('B', 7)],
         'B': [('C', 8), ('E', 7)],
         'C': [('E', 5)],
         'D': [('F', 6)],
         'E': [('G', 9)]
    }
    a3 = {}
    
    
    assert question3(a1) == g1
    assert question3(a2) == g2
    #Corner cases
    assert question3(a3) == None
    assert question3(None) is None

    print('---Q3---')


def main():
   # test_q1()
   # test_q2()
    test_q3()
   # test_q4()
   # test_q5()


if __name__ == '__main__':
    main()
    
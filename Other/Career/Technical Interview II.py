# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 06:32:39 2017

@author: mnshr
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

def q1_test():
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

def q2_test(): 
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

def question3(g):
    #Handle null case
    if not g:
        return None

    #Dict to store the adjacency list
    adj_list = {}  
    #Store here the nodes visited
    visited_nodes = []
    tmp = []
    for v in g.keys():
        #Initializing the adjaceny list with all keys of the graph as None lists
        adj_list[v] = []
        tmp = v

    # Visit the last node from the loop above
    visited_nodes.append(tmp)

    tmp = None
    #Visit as long as all the nodes in the graph aren't checked
    while len(visited_nodes) < len(g):
        #print 'Visited Node Length: ', len(visited_nodes), ' Graph Len: ', len(g)
        for vst in visited_nodes:
            #print 'Visiting the node: ', vst
            for conn_node in g[vst]:
                #print 'Connected node is: ', conn_node, ' | Checked node is: ', tmp
                #Check the weights, if lesser weight and node is not visited
                #prepare to add them to adjacency list
                if not tmp or conn_node[1] < tmp[1]:
                    if conn_node[0] not in visited_nodes:
                        tmp = conn_node
                        to_node = conn_node[0]
                        from_node = vst
        if not tmp:
            break
        #Add to adjacency list
        adj_list[from_node].append(tmp)
        adj_list[to_node].append((from_node, tmp[1]))
        visited_nodes.append(to_node)
        #Reset the pointers
        tmp = None
        to_node = None
        from_node = None
    return adj_list

def q3_test():
    a = {
        'A': [('B', 2)],
        'B': [('A', 2), ('C', 5)],
        'C': [('B', 5)],
        'D': [('A', 1), ('C', 4)]
    }
    
    mst = {'A': [('D', 1), ('B', 2)], 
    'C': [('D', 4)], 
    'B': [('A', 2)], 
    'D': [('A', 1), ('C', 4)]}
    
    a1 = {'A': [('B', 7), ('D', 5)],
         'B': [('A', 7), ('C', 8), ('D', 9)],
         'C': [('B', 8), ('E', 5)],
         'D': [('A', 5), ('B', 9), ('E', 15), ('F', 6)],
         'E': [('C', 5), ('D', 15), ('F', 8), ('G', 9)],
         'F': [('D', 6), ('E', 8), ('G', 11)],
         'G': [('E', 9), ('F', 11)]}
    
    mst1 = {'A': [('D', 5), ('B', 7)], 
    'C': [('E', 5)], 
    'B': [('A', 7)], 
    'E': [('F', 8), ('C', 5), ('G', 9)], 
    'D': [('F', 6), ('A', 5)], 
    'G': [('E', 9)], 
    'F': [('D', 6), ('E', 8)]}
    
    a2 = {}
    
    a3 = {'A': [('B', 2)],
                 'B': [('A', 2), ('C', 5), ('D', 3)],
                 'C': [('B', 5), ('D', 4)],
                 'D': [('C', 4), ('B', 3)]}
    mst3 = {'A': [('B', 2)], 
    'C': [('D', 4)], 
    'B': [('D', 3), ('A', 2)], 
    'D': [('B', 3), ('C', 4)]}
    
    assert question3(a) == mst
    assert question3(a1)==mst1
    assert question3(a3) == mst3
    #Corner cases
    assert question3(a2) == None
    assert question3(None) is None

    print('---Q3---')

"""
Question 4
Find the least common ancestor between two nodes on a binary search tree. 
The least common ancestor is the farthest node from the root that is an 
ancestor of both nodes. For example, the root is a common ancestor of all 
nodes on the tree, but if both nodes are descendents of the root's left child,
then that left child might be the lowest common ancestor. You can assume that
both nodes are in the tree, and the tree itself adheres to all BST properties.
The function definition should look like question4(T, r, n1, n2), where T is 
the tree represented as a matrix, where the index of the list is equal to the 
integer stored in that node and a 1 represents a child node, r is a 
non-negative integer representing the root, and n1 and n2 are non-negative
integers representing the two nodes in no particular order. For example, one 
test case might be

question4([[0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0]],
          3,
          1,
          4)
and the answer would be 3.
"""        
class Node_tree(object):
    def __init__(self, value):
        self.left = None
        self.value = value
        self.right = None

def create_tree(T, node):
    for i in range(len(T)):
        # If value is 1, add a child
        if T[node.value][i] == 1:
            # Add child to the right if its value is greater
            if i > node.value:
                #print 'Adding ', i, ' to right of ', node.value
                node.right = i
                #Recurse to add nodes below, if they exist
                create_tree(T, Node_tree(i))
        
            # Add child to the left if its value is smaller
            if i < node.value:
                #print 'Adding ', i, ' to left of ', node.value
                node.left = i
                #Recurse to add nodes below, if they exist
                create_tree(T, Node_tree(i))

def find_lca(root, n1, n2):
    if root == None:
        return None
    #print root.value, n1, n2
    # Search recursively for least common ancestor in the left
    if(root.value > n1 and root.value > n2):
        return find_lca(root.left, n1, n2)
 
    # Search recursively for lca in the right
    if(root.value < n1 and root.value < n2):
        return find_lca(root.right, n1, n2)
 
    return root.value 

def question4(T, r, n1, n2):
    # Sanity checks
    if T == None:
        return None
    if r < 0 or n1 < 0 or n2 < 0:
        return None
    
    # Create a root node for the tree
    root = Node_tree(r)
    
    # Create binary search tree from given 2D array
    create_tree(T, root)
    
    # Return the least common ancestor
    return find_lca(root, n1, n2)

def q4_test():                    
    T1 =[[0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]] 
    assert question4(T1, 3, 1, 4) == 3
    T2 = [[0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]]
    assert question4(T2, 5, 4, 6)==5
    T3 = [[0, 0, 0],
          [1, 0, 1],
          [0, 0, 0]]
    assert question4(T3, 1, 0, 2) == 1

    #Corner cases
    assert question4(None, None, None, None) is None
    assert question4(T1, -1, 2, 3) == None
    assert question4([], None, None, None) is None
    
    print('---Q4---')

"""
Question 5
Find the element in a singly linked list that's m elements from the end. 
For example, if a linked list has 5 elements, the 3rd element from the end is 
the 3rd element. The function definition should look like question5(ll, m), 
where ll is the first node of a linked list and m is the "mth number from the 
end". You should copy/paste the Node class below to use as a representation of
 a node in the linked list. Return the value of the node at that position.
"""
class Node_ll(object):
  def __init__(self, data):
    self.data = data
    self.next = None


def LinkedList(arr):
    head = None
    if arr:
        # Create the head node
        head = Node_ll(arr[0])
        current = head
        # Add more nodes if given 
        if len(arr) > 1:
            for a in arr[1:]:
                # Add the node as next element
                current.next = Node_ll(a)
                current = current.next
    return head
    
def question5(ll, m):
    if not ll:
        return None
    #Using leading and lagging pointers in the linked list
    lead = ll
    lagg = ll

    # Point the leading pointer m-1 nodes forward
    for i in range(m - 1):
        lead = lead.next
        if lead is None:
            # List not long enough for m
            return None

    # Move leading and lagging pointers till list ends
    # lagging pointer should be at the required node
    while True:
        if lead.next is None:
            return lagg.data
        if lagg.next is None:
            return None
        lead = lead.next
        lagg = lagg.next


def q5_test():
    # 23 -> 49 -> 93 -> 30 -> 11 -> 67
    l1 = LinkedList([23, 49, 93, 30, 11, 67])
    assert question5(l1, 3) == 30
    assert question5(l1, 4) == 93
    assert question5(l1, 5) == 49
    assert question5(l1, 6) == 23
    
    #Corner Cases
    assert question5(None, 4) is None
    l2 = LinkedList([])
    assert question5(l2, 3) is None
    #print l1
    print('---Q5---')

def main():
   q1_test()
   q2_test()
   q3_test()
   q4_test()
   q5_test()


if __name__ == '__main__':
    main()
    
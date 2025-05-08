import threading
import time
import random


class TrieNode:
    def __init__(self):
        self.is_end = False
        self.children = {}


class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def add(self, path):
        cur = self.root
        for p in path.split("/"):
            if p not in cur.children:
                cur.children[p] = TrieNode()
            cur = cur.children[p]
            
        cur.is_end = True

    def search(self, path):
        cur = self.root
        for p in path.split("/"):
            if p not in cur.children:
                return False
            cur = cur.children[p]

        return cur.is_end
    
    def startswith(self, pre):
        cur = self.root
        for p in pre.split("/"):
            if p not in cur.children:
                return False
            cur = cur.children[p]
        return True
    
def p1(arr, low, high):
    x = arr[low]
    i = high
    for j in range(low, high, -1):
        if arr[j] > x:
            arr[i], arr[j] = arr[j], arr[i]
            i -= 1
    arr[i], arr[low] = arr[low], arr[i]
    return i

def p(arr, low, high):
    x = arr[high]
    i = low
    for j in range(low, high):
        if arr[j] < x:
            arr[j], arr[i] = arr[i], arr[j]
            i += 1
    arr[i], arr[high] = arr[high], arr[i]
    return i
    

def qs(arr, low, high):
    if low < high:
        m = p(arr, low, high)
        qs(arr, low, m-1)
        qs(arr, m+1, high)



def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        merge_sort(left)
        merge_sort(right)

        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1


def bubble_sort(arr):
    n = len(arr)
    for i in range(0, n-1):
        for j in range(n-1-i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def insert_sort(arr):
    n = len(arr)
    for i in range(1, n):
        x = arr[i]
        j = i-1
        while j >= 0 and arr[j] < x:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = x


def get_depth(root):
    if not root:
        return 0
    ld = get_depth(root.left)
    rd = get_depth(root.right)
    return max(ld, rd) + 1

def get_ans(root, p, q):
    if not root or root == p or root == q:
        return root
    l = get_ans(root.left, p, q)
    r = get_ans(root.right, p, q)
    if l and r:
        return root
    if l:
        return l
    if r:
        return r
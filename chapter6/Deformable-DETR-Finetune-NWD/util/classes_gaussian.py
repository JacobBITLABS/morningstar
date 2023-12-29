"""
Module Responsible for a running GMM of class predictions 
"""
from collections import deque

def queue():
    print()
    q = deque(maxlen=3) 

    # Adding of element to queue
    q.put('a')
    q.put('b')
    q.put('c')
    q.put('d')

    # Return Boolean for Full 
    # Queue 
    print("\nFull: ", q.full()) 
    
    # Removing element from queue
    print("\nElements dequeued from the queue")
    print(q.get())
    print(q.get())
    print(q.get())


class PredictionQueue:
    """
    Prediction Queue
    """

    def __init__(self, class_id, maxsize=100):
        self.queue = deque(maxlen=maxsize) 
        self.class_id = class_id
        self.current_size = 0
        self.max_size = maxsize

    def insert(self, val):
        """
        Insert value into queue. If the queue is of maxsize size, then the First last element in, is popped before inserting
        """
        if self.current_size >= self.max_size:
            self.queue.popleft() # pop FIFO element
            self.queue.append(val) # insert new val
        else:
            self.queue.append(val)
            self.current_size += 1

    def print_queue(self):
        for e in self.queue:
            print(e)

    

if __name__ == "__main__":
    print("Classes Gaussian")
    q = PredictionQueue(1, 3)
    q.insert(1)
    q.insert(2)
    q.insert(3)
    q.print_queue()
    print("****")
    q.insert(4)
    q.print_queue()
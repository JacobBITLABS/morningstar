

class SAMQueue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, data): # data = sam
        # new_item = QueueItem(data)
        self.items.append(data)
    
    def dequeue(self):
        if self.items:
            return self.items.pop(0)
        else:
            return None
    
    def get_all_items(self):
        return [item.data for item in self.items]
    
    def get_first_claimed_item_data(self):
        while True:
            print("Waiting for SAM item")
            for item in self.items:
                print("Testing SAM id: ", str(item.id))
                is_clamimed = item.claimed
                print("IS_CLAIMED: ", is_clamimed)
                if is_clamimed == False:
                    return item # return sam object
        #return None  # Return None if no claimed items are found
    
    def get_claimed_items(self):
        return [item.data for item in self.items if item.claimed]
    
    def get_unclaimed_items(self):
        return [item.data for item in self.items if not item.claimed]
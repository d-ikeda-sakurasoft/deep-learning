class Man:
    def __init__(self, name):
        self.name = name
    
    def hello(self):
        print(self.name)

m = Man("Ken")
m.hello()

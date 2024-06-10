from ftools import ftimer

    
def add(x, y):
    with ftimer.Ftimer(name="Test Timer", logging=True) as t:
        x = 1
        y = 2
        z = x + y
    
def main():
    add(1, 2)
    
if __name__ == "__main__":
    main()
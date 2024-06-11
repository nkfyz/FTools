from ftools import Ftimer

def add(x, y):
    with Ftimer(name="Test Timer", type="TIMER", logging=True) as t:
        x = 1
        y = 2
        z = x + y
    
def main():
    add(1, 2)
    
if __name__ == "__main__":
    main()
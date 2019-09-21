import tkinter
import pygeoip
import pylab as pl


def test():
    x=[1,3,4,6,7]
    y=[2,4,5,6,2]

    pl.plot(x,y)
    for i in range(x.__len__()):
        if i%2==0:
            pl.plot(x[i],y[i],"or")
        else:
            pl.plot(x[i],y[i],"ob")
    pl.show()


if __name__ == "__main__":
    test()
    # root = tkinter.Tk()
    # root.title("test")
    # tkinter.mainloop()
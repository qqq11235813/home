import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import Tkinter
import time

def  init(data):
    data.timerDelay=10
    data.x=np.arange(0.0,3.0,0.01)
    data.t=0
    data.omigaP=0
    data.omigaG=0

def sliderPressed(slider,data,flag):
    scale=slider.get()
    #scale=10
    if(flag=='GroupV'):
        data.omigaG=float(scale)/100
    elif(flag=='PhaseV'):
        data.omigaP=float(scale)/100
    else:print 'not a suitable variable'

def timerFired(data):
    data.t+=1

def redrawAll(ax,data):
    x0=data.x
    omigaP0=data.omigaP
    omigaG0=data.omigaG
    t0=data.t
    y=np.cos(np.pi*(3*x0-(omigaP0+omigaG0)*t0))+np.cos(np.pi*(2*x0-(omigaP0-omigaG0)*t0))
    z=2*np.cos(np.pi*(0.5*x0-omigaG0*t0))

    ax.plot(x0,y)
    ax.plot(x0,z)
    
def run():
    def redrawAllWrapper(canvas,ax,data):
        ax.clear()
        redrawAll(ax,data)

        canvas.show()

    def sliderPressedWrapper(event,canvas,ax,slider,data,flag):
        sliderPressed(slider,data,flag)
        redrawAllWrapper(canvas,ax,data)

    def timerFiredWrapper(canvasWhole,canvas,ax,data):
        timerFired(data)
        redrawAllWrapper(canvas,ax,data)
        canvasWhole.after(data.timerDelay,timerFiredWrapper,canvasWhole,canvas,ax,data)
    
    ########################################################
                  #init
    ##################################################
    class Struct(object):pass
    data=Struct()
    init(data)
    root=Tkinter.Tk()
    root.wm_title("Phase Velocity and Group Velocity")

    fig=Figure(figsize=(10,4),dpi=100)
    ax=fig.add_subplot(111)
    canvas=FigureCanvasTkAgg(fig,master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)

    canvasWhole=Tkinter.Canvas(root,width=1000,height=50)
    canvasWhole.pack()
    #############################################################
    
    ###########################################################
    #animation
    ############################################################
    timerFiredWrapper(canvasWhole,canvas,ax,data)
    ################################################################
    
    #######################################################
    #slider
    #########################################################
    slider_GroupV=Tkinter.Scale(master=root,from_=-10.0,to=10.0,resolution=0.1,
                     label="group velocity",
                     orient=Tkinter.HORIZONTAL,
                     length=500,
                     )
    slider_PhaseV=Tkinter.Scale(master=root,from_=-10.0,to=10.0,resolution=0.1,
                     label="phase velocity",
                     orient=Tkinter.HORIZONTAL,
                     length=500,
                     )
    slider_GroupV.pack()
    slider_PhaseV.pack()
    slider_GroupV.bind("<ButtonRelease-1>",lambda event:sliderPressedWrapper(event,canvas,ax,slider_GroupV,data,'GroupV'))
    slider_PhaseV.bind("<ButtonRelease-1>",lambda event:sliderPressedWrapper(event,canvas,ax,slider_PhaseV,data,'PhaseV'))
    ##########################################################
    
    root.mainloop()

run()

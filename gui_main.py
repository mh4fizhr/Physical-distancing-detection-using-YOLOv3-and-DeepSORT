import cv2
from tkinter import *
from tkinter import filedialog
import os
import tkinter as tk
from PIL import Image, ImageTk

def showimage():
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title='Select Image File', filetypes=(('JPG file','*.jpg'), ('PNG file','*.png'), ('All Files', '*.*')))
    img = Image.open(fln)
    img.thumbnail((350,350))
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image = img
    
def showframe():
    #Graphics window
    # mainWindow = tk.Tk()
    # mainWindow.configure(bg=lightBlue2)
    # mainWindow.geometry('%dx%d+%d+%d' % (maxWidth,maxHeight,0,0))
    # mainWindow.resizable(0,0)
    # mainWindow.overrideredirect(1)

    #Capture video frames
    # lmain = tk.Label(mainFrame)
    # lmain.grid(row=0, column=0)

    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title='Select video File', filetypes=(('MP4 file','*.mp4'), ('AVI file','*.avi'), ('All Files', '*.*')))
    cap = cv2.VideoCapture(fln)
    ret, frame = cap.read()
    cv2image   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img   = Image.fromarray(cv2image).resize((760, 400))
    imgtk = ImageTk.PhotoImage(image = img)
    lbl.imgtk = imgtk
    lbl.configure(image=imgtk)
    lbl.after(10, showframe)


root = Tk()

frm = Frame(root)
frm.pack(side=BOTTOM, padx=15, pady=15)

lbl = Label(root)
lbl.pack()

btn = Button(frm, text="Browse Image", command=showimage)
btn.pack(side=tk.LEFT)

btn3 = Button(frm, text="Browse Video", command=showframe)
btn3.pack(side=tk.LEFT)

btn2 = Button(frm, text="Exit", command=lambda: exit())
btn2.pack(side=tk.LEFT, padx=10)

root.title("image browser")
root.geometry("300x350")
root.mainloop()
import tkinter
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
import os
from function_yolo_deepsort_distance import yolo_deepsort_distance
import time

class App:
    video = 'null'

    def __init__(self, window, window_title, image_path="instastory.jpg"):
        self.window = window
        self.window.title('Physical distancing detection ver 1.0')

        self.canvas = tk.Canvas(self.window, width=600, height=350)
        self.canvas.pack()

        # Label Judul
        self.label_title = tkinter.Label(window, 
                                        text="Physical distancing detector", 
                                        width=100,height=4,
                                        fg="red")
        self.label_title.config(font=('helvetica', 18))
        self.canvas.create_window(300, 50, window=self.label_title)

        # Label status
        self.label_file_explorer = tkinter.Label(window,
                                    text = "Status",
                                    width = 100, height = 4,
                                    fg = "blue")
        self.canvas.create_window(300, 120, window=self.label_file_explorer)

        # button browse file
        self.btn = tkinter.Button(window, text="Browse video",width=20,height=2, command=self.browse_file)
        self.canvas.create_window(300, 170, window=self.btn)

        ############ 
        # Distance #
        ############
        self.label_distance = tkinter.Label(window,
                                    text = "distance (cm)   :",
                                    width = 100, height = 4)
        self.canvas.create_window(150, 250, window=self.label_distance)

        self.input_distance = tkinter.Entry (window,width=15,state=tkinter.DISABLED) 
        self.canvas.create_window(250, 250, window=self.input_distance)

        ############ 
        #  output  #
        ############
        self.label_output = tkinter.Label(window,
                                    text = "output (.mp4)   :",
                                    width = 100, height = 4)
        self.canvas.create_window(150, 290, window=self.label_output)

        self.input_output = tkinter.Entry (window,width=15,state=tkinter.DISABLED) 
        self.canvas.create_window(250, 290, window=self.input_output)
        
        # button detection
        self.btn2 = tkinter.Button(window, text="Detection",width=20,height=4, command=self.detection,state=tkinter.DISABLED)
        self.canvas.create_window(400, 270, window=self.btn2)
        

        self.interval = 5 # Interval in ms to get the latest frame

        # self.update_clock()
        
        self.window.mainloop()

    def browse_file(self):
        self.fln = filedialog.askopenfilename(initialdir=os.getcwd(), title='Select video File', filetypes=(('MP4 file','*.mp4'), ('AVI file','*.avi'), ('All Files', '*.*')))
        self.cap = cv2.VideoCapture(self.fln)

        # Change label contents
        self.label_file_explorer.configure(text="File Opened : "+self.fln)

        # change button detection
        self.btn2.config(state=tkinter.NORMAL)
        self.input_distance.config(state=tkinter.NORMAL)
        self.input_output.config(state=tkinter.NORMAL)

        # self.update_image(self.cap)
        App.video = self.fln

        

    def update_image(self,cap):
        # self.cap = cv2.VideoCapture('example2.mp4')

        self.window = self.window
        # self.cap = cap
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Create canvas for image
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.pack()

        self.video_run()

    def video_run(self):
        # Get the latest frame and convert image format
        self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB) # to RGB
        self.image = Image.fromarray(self.image) # to PIL format
        self.image = ImageTk.PhotoImage(self.image) # to ImageTk format
        
        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        
        # Repeat every 'interval' ms
        self.window.after(self.interval, self.video_run)

    def detection(self):
        # Change label contents
        self.label_file_explorer.configure(text="Processing...")
        
        cap = App.video
        distance = self.input_distance.get()
        output = self.input_output.get()
        yolo_deepsort_distance(cap,distance,output)
        
        self.label_file_explorer.configure(text="Finish : "+output+".avi")

    # def update_clock(self):
    #     now = time.strftime("%H:%M:%S")
    #     self.label_file_explorer.configure(text=now)
    #     self.window.after(1000, self.update_clock)


# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
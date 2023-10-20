import os
import cv2
from PIL import Image, ImageTk
import tkinter.filedialog
from tkVideoPlayer import TkinterVideo
import tkinter as tk
from tkinter import ttk

class App():
    def __init__(self):
        self.root = tk.Tk() # create main window of the app
        self.root.geometry('1000x600') # width x height
        self.root.title("Muaythai comparison") # title of the window
   
        self.layout()
        self.widget()

        self.root.mainloop() # show the window on screen

    def layout(self):
        self.mainframe = tk.Frame(self.root) # create frame and told its parent (root)
        self.mainframe.pack(fill='both', expand=True) # fill window with frame

        self.expert_area = tk.LabelFrame(self.mainframe, text="Expert")
        self.expert_area.grid(row=0,column=0, sticky='W',padx=5, pady=5)

        self.expert_area_vid = tk.Label(self.mainframe)
        self.expert_area_vid.grid(row=1,column=0,padx=5, pady=5)

        self.student_area = tk.LabelFrame(self.mainframe, text="Student")
        self.student_area.grid(row=0,column=1, sticky='E', padx=5, pady=5)

        self.student_area_vid = tk.Label(self.mainframe)
        self.student_area_vid.grid(row=1,column=1, padx=5, pady=5)

        self.result_area = tk.LabelFrame(self.mainframe, text="Result")
        self.result_area.grid(row=2, column=0, rowspan=2, columnspan=3 , sticky='S',padx=5, pady=5)
        

    def widget(self):
        self.text_expert = ttk.Label(self.expert_area, text='วิดิโอครูฝึก', font=("Brass Mono", 30)) # put text on mainframe
        self.text_expert.grid(row = 0, column = 0) # specify where text is in mainframe

        upload_button_expert = ttk.Button(self.expert_area, text = 'Upload video', command=self.upload_vid_expert) # command = link to onclick function
        upload_button_expert.grid(row=1, column= 0, pady=10)
        play_button_expert = ttk.Button(self.expert_area, text = 'Play', command=self.play_video_expert) # command = link to onclick function
        play_button_expert.grid(row=2, column= 0, pady=10)

        self.text_student = ttk.Label(self.student_area, text='วิดิโอนักเรียน', font=("Brass Mono", 30)) # put text on mainframe
        self.text_student.grid(row = 0, column = 0) # specify where text is in mainframe

        upload_button_student = ttk.Button(self.student_area, text = 'Upload video', command=self.upload_vid_student) # command = link to onclick function
        upload_button_student.grid(row=1, column= 0, pady=10)
        play_button_student = ttk.Button(self.student_area, text = 'Play', command=self.play_video_student) # command = link to onclick function
        play_button_student.grid(row=2, column= 0, pady=10)
    

    def upload_vid_expert(self):
        
        self.expert_area.filename = tk.filedialog.askopenfilename(initialdir=f"C:/Users/{os.getlogin()}/Desktop", title="Please Select a Video", filetypes=(("mp4 file","*.mp4"),("all files", "*.*")))
        self.video_dir = ttk.Label(self.expert_area, text=self.expert_area.filename)
        self.video_dir.grid(row = 3, column=0)

        self.player_expert = TkinterVideo(self.expert_area_vid, keep_aspect=True)
        self.player_expert.load(self.expert_area.filename)
        
    def upload_vid_student(self):
        
        self.student_area.filename = tk.filedialog.askopenfilename(initialdir=f"C:/Users/{os.getlogin()}/Desktop", title="Please Select a Video", filetypes=(("mp4 file","*.mp4"),("all files", "*.*")))
        self.video_dir = ttk.Label(self.student_area, text=self.student_area.filename)
        self.video_dir.grid(row = 3, column=0)

        self.player_student = TkinterVideo(self.student_area_vid, keep_aspect=True)
        self.player_student.load(self.student_area.filename)

    
    def play_video_expert(self):
        
        self.player_expert.pack(expand=True, fill="both")
        self.player_expert.play()

    def play_video_student(self):
    
        self.player_student.pack(expand=True, fill="both")
        self.player_student.play()



if __name__ == "__main__":
    App()
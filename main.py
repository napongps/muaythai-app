import os
import cv2
import tkinter.filedialog
from tkVideoPlayer import TkinterVideo
import tkinter as tk
# from tkinter import ttk
import ttkbootstrap as ttk
from detect_landmark import landmark_detection
from angle import *
from cosine import *
from DTW import dtw

class App():
    def __init__(self):
        self.root = ttk.Window(themename = 'darkly') # create main window of the app
        self.root.geometry('1000x600') # width x height
        self.root.title("Muaythai comparison") # title of the window

        self.weight_angle= [1]*12
        self.weight_cosine = [1]*16
        self.window = 0

        self.layout()
        self.widget()

        self.root.mainloop() # show the window on screen

    def layout(self):
        self.mainframe = ttk.Frame(self.root) # create frame and told its parent (root)

        # expert frame
        self.expert_area = ttk.LabelFrame(self.mainframe, text="Expert")
        self.expert_area.grid(row=0,column=0, sticky='W',padx=5, pady=5)

        self.expert_area_vid = ttk.Label(self.mainframe)
        self.expert_area_vid.grid(row=1,column=0,padx=5, pady=5)

        # student frame
        self.student_area = ttk.LabelFrame(self.mainframe, text="Student")
        self.student_area.grid(row=0,column=1, sticky='E', padx=5, pady=5)

        self.student_area_vid = ttk.Label(self.mainframe)
        self.student_area_vid.grid(row=1,column=1, padx=5, pady=5)

        # result frame
        self.result_area = ttk.LabelFrame(self.mainframe, text="Result")
        self.result_area.grid(row=3, column=0, rowspan=2, columnspan=2 , sticky='S',padx=5, pady=5)

        # option frame
        self.option_area = ttk.Labelframe(self.mainframe, text="Options")
        self.option_area.grid(row=0,column=2, sticky='E', padx=5, pady=5, ipadx= 5, ipady=5)
        
        self.mainframe.pack(fill='both', expand=True) # fill window with frame

    def widget(self):
        
        #Heading
        text_expert = ttk.Label(self.expert_area, text='วิดิโอครูฝึก', font=("Brass Mono", 30)) # put text on mainframe
        text_expert.grid(row = 0, column = 0) # specify where text is in mainframe

        text_student = ttk.Label(self.student_area, text='วิดิโอนักเรียน', font=("Brass Mono", 30)) # put text on mainframe
        text_student.grid(row = 0, column = 0) # specify where text is in mainframe

        # upload button
        upload_button_expert = ttk.Button(self.expert_area, text = 'Upload video', command=self.upload_vid_expert) # command = link to onclick function
        upload_button_expert.grid(row=1, column= 0, pady=10)

        upload_button_student = ttk.Button(self.student_area, text = 'Upload video', command=self.upload_vid_student) # command = link to onclick function
        upload_button_student.grid(row=1, column= 0, pady=10)

        # play button
        play_button_expert = ttk.Button(self.expert_area, text = 'Play', command=self.play_video_expert) # command = link to onclick function
        play_button_expert.grid(row=2, column= 0, pady=10)
        
        play_button_student = ttk.Button(self.student_area, text = 'Play', command=self.play_video_student) # command = link to onclick function
        play_button_student.grid(row=2, column= 0, pady=10)

        # calculate button
        cal_button = ttk.Button(self.mainframe, text = 'Go', command=self.calculate_video) # command = link to onclick function
        cal_button.grid(row=2, column= 0, columnspan=3, pady=10)

        # method dropdown
        method_list=["Cosine_similarity", "Angle_Similarity"]
        self.method = tk.StringVar()

        method_dropdown = ttk.OptionMenu(self.option_area, self.method, method_list[0], *method_list)
        method_dropdown.grid(row=0,column=0, columnspan=2, padx=5, pady=5)

        # norm value dropdown
        text_norm = ttk.Label(self.option_area, text='Highest Difference') # put text on mainframe
        text_norm.grid(row = 1, column = 0, sticky='W', padx=5, pady=5) # specify where text is in mainframe

        norm_list = ["45", "90", "180"]
        self.norm = tk.StringVar()

        norm_value =  ttk.OptionMenu(self.option_area, self.norm, norm_list[2], *norm_list)
        norm_value.grid(row=1, column=1, padx=5, pady=5)

        # option checkbox
        self.weight_var = tk.IntVar()
        self.weight_var.set(0)

        weight_check = ttk.Checkbutton(self.option_area, text='Weight', variable=self.weight_var, command=self.input_weight)
        weight_check.grid(row=2, sticky='W', padx=5)        

        self.MAW_var = tk.IntVar()
        self.MAW_var.set(0)

        MAW_check = ttk.Checkbutton(self.option_area, text='Moving Weight', variable=self.MAW_var, command=self.MAW_window)
        MAW_check.grid(row=3, sticky='W', padx=5)

        self.thresh_var = tk.IntVar()
        self.thresh_var.set(0)

        thresh_check = ttk.Checkbutton(self.option_area, text='Threshold', variable=self.thresh_var)
        thresh_check.grid(row=4, sticky='W', padx=5)

        self.expo_var = tk.IntVar()
        self.expo_var.set(0)

        expo_check = ttk.Checkbutton(self.option_area, text='Exponential', variable=self.expo_var)
        expo_check.grid(row=5, sticky='W', padx=5)

        self.kf_var = tk.IntVar()
        self.kf_var.set(0)

        kf_check = ttk.Checkbutton(self.option_area, text='Keyframes', variable=self.kf_var)
        kf_check.grid(row=6, sticky='W', padx=5)

    def input_weight(self):
        
        if 'cosine' in self.method.get().lower():
            weight_label_LR=['เท้า','หน้าแข้ง','ต้นขา','สีข้าง','มือ','ปลายแขน','ต้นแขน']
            cosine_flag = True
            weight_amount = 7
            self.weight_select = self.weight_cosine

        else:
            weight_label_LR=['ข้อมือ','ข้อศอก','หัวไหล่','สะโพก','เข่า','ข้อเท้า']
            cosine_flag = False
            weight_amount = 6
            self.weight_select = self.weight_angle

        if (self.weight_var.get()):
            input_weight_window = ttk.Toplevel()
            input_weight_window.title('Weight')

            input_weight_area = ttk.LabelFrame(input_weight_window, text="Weight")
            input_weight_area.pack(fill='both', expand=True, padx=5, pady=5)

            weight_temp = []
            
            #right
            right = ttk.LabelFrame(input_weight_area, text='Right')
            right.grid(row=0,column=1,padx=5,pady=5,ipadx=5,ipady=5)

            for i in range(weight_amount):
                weight_label = ttk.Label(right, text=weight_label_LR[i],font=16)
                weight_label.grid(row=i, column=0, padx=5, pady=2)
                weight_input = ttk.Entry(right)
                weight_input.insert(0, self.weight_select[i])
                weight_input.grid(row=i,column=1, padx=5,pady=2)
                weight_temp.append(weight_input)

            #left
            left = ttk.LabelFrame(input_weight_area, text='Left')
            left.grid(row=0,column=0,padx=5,pady=5,ipadx=5,ipady=5)

            for i in range(weight_amount):
                weight_label = ttk.Label(left, text=weight_label_LR[i],font=16)
                weight_label.grid(row=i, column=0, padx=5, pady=2)
                weight_input = ttk.Entry(left)
                weight_input.insert(0, self.weight_select[i+weight_amount])
                weight_input.grid(row=i,column=1, padx=5,pady=2)
                weight_temp.append(weight_input)
            
            if cosine_flag:
                shoulder_label = ttk.Label(input_weight_area,text='ไหปลาร้า',font=16)
                shoulder_label.grid(row=1,column=0, padx=5,pady=2)
                shoulder_input = ttk.Entry(input_weight_area)
                shoulder_input.insert(0, self.weight_select[14])
                shoulder_input.grid(row=1, column=1, padx=5,pady=2)
                weight_temp.append(shoulder_input)

                hip_label = ttk.Label(input_weight_area,text='สะโพก',font=16)
                hip_label.grid(row=2,column=0, padx=5,pady=2)
                hip_input = ttk.Entry(input_weight_area)
                hip_input.insert(0, self.weight_select[15])
                hip_input.grid(row=2, column=1, padx=5,pady=2)
                weight_temp.append(hip_input)

            def weight_get():
                if '' in list(map(lambda x: x.get(), weight_temp)):
                    tk.messagebox.showerror("Error", "Please fill in all part.")
                else:
                    try:
                        self.weight_select = list(map(lambda x: float(x.get()), weight_temp))
                        if 'cosine' in self.method.get().lower():
                            self.weight_cosine = self.weight_select
                        else:
                            self.weight_angle = self.weight_select
                        
                    except:
                        tk.messagebox.showerror("Error", "Weight must be a number")
                    input_weight_window.destroy()

            submit_button = ttk.Button(input_weight_area, text='Done', command=weight_get)
            submit_button.grid(row=3,pady=10, columnspan=3)



    def MAW_window(self):

        MAW_window = ttk.Toplevel()
        MAW_window.title('Moving Weight')

        MAW_frame = ttk.LabelFrame(MAW_window, text='Moving Weight')
        MAW_frame.grid(row=0,column=1,padx=5,pady=5,ipadx=5,ipady=5)

        Window_label = ttk.Label(MAW_frame, text='Window',font=16)
        Window_label.grid(row=0,column=0, padx=5,pady=2)

        Window_input = ttk.Entry(MAW_frame)
        Window_input.insert(0, self.window)
        Window_input.grid(row=0,column=1, padx=5,pady=2)

        def window_get():
            try:
                self.window = int(Window_input.get())
            except:
                tk.messagebox.showerror("Error","Window must be a number")
            MAW_window.destroy()

        submit_button = ttk.Button(MAW_frame, text='Done', command=window_get)
        submit_button.grid(row=2,pady=5, columnspan=2)




    def upload_vid_expert(self):
        
        self.expert_area.filename = tk.filedialog.askopenfilename(title="Please Select a Video", filetypes=(("mp4 file","*.mp4"),("all files", "*.*")))
        self.video_dir = ttk.Label(self.expert_area, text=self.expert_area.filename)
        self.video_dir.grid(row = 3, column=0)

        self.player_expert = TkinterVideo(self.expert_area_vid, keep_aspect=True)
        self.player_expert.load(self.expert_area.filename)
        
    def upload_vid_student(self):
        
        self.student_area.filename = tk.filedialog.askopenfilename(title="Please Select a Video", filetypes=(("mp4 file","*.mp4"),("all files", "*.*")))
        self.video_dir = ttk.Label(self.student_area, text=self.student_area.filename)
        self.video_dir.grid(row = 3, column=0)

        self.player_student = TkinterVideo(self.student_area_vid, keep_aspect=True)
        self.player_student.load(self.student_area.filename)

    
    def play_video_expert(self):
        
        self.player_expert.grid()
        self.player_expert.play()

    def play_video_student(self):
    
        self.player_student.grid()
        self.player_student.play()

    def calculate_video(self):

        try:
            cam_ladk_expert, world_ladk_expert, all_frame_expert = landmark_detection(self.expert_area.filename)
        except:
            tk.messagebox.showerror("Cannot calculate","Please upload expert's video.")
        try:
            cam_ladk_student, world_ladk_student, all_frame_student = landmark_detection(self.student_area.filename)
        except:
            tk.messagebox.showerror("Cannot calculate","Please upload student's video.")

        if self.weight_var.get():
            if 'cosine' in self.method.get().lower():
                weight_arr = np.array(self.weight_cosine)
            else:
                weight_arr = np.array(self.weight_angle)
        else:
            weight_arr = np.ones(16)
        
        print(weight_arr)
        print(self.MAW_var.get())
        print(self.norm.get())
        print(self.thresh_var.get())
        print(self.expo_var.get())
        if 'cosine' in self.method.get().lower():
            limb_expert = find_limb(world_ladk_expert)
            limb_student = find_limb(world_ladk_student)
            path, dist_mat, dist_lndmk_mat, cost_mat, cost = dtw(limb_expert,limb_student,cosine_similarity,
                                                                weight=weight_arr, MAW=self.MAW_var.get(), 
                                                                norm_value=int(self.norm.get()), 
                                                                windows=50, thresh=self.thresh_var.get(),
                                                                expo=self.expo_var.get())
            print(cost)

        else:
            angle_expert = find_angle(world_ladk_expert)
            angle_student = find_angle(world_ladk_student)
            path, dist_mat, dist_lndmk_mat, cost_mat, cost = dtw(angle_expert,angle_student,angle_similarity,np.ones(12),45)
            print(cost)


        

if __name__ == "__main__":
    App()
import threading
import os
from concurrent.futures import ProcessPoolExecutor
import time
import cv2


from PIL import ImageTk, Image
import tkinter.filedialog
from tkVideoPlayer import TkinterVideo
import tkinter as tk
# from tkinter import ttk
import ttkbootstrap as ttk

from detect_landmark import landmark_detection
from angle import *
from cosine import *
from DTW import dtw
from result import *

class App():
    def __init__(self):

        self.root = ttk.Window(themename = 'darkly') # create main window of the app
        self.root.geometry('1200x800') # width x height
        self.root.resizable(False,False)
        self.root.title("Muaythai comparison") # title of the window
        

        self.weight_angle= [1]*12
        self.weight_cosine = [1]*16
        self.window_output = 0
        self.thread_list=[]
        print(len(self.thread_list))
        self.layout()
        self.widget()

        self.root.mainloop() # show the window on screen

    def start_thread(self):
        self.thread_list.append(threading.Thread(target=self.calculate_video))
        self.thread_list[-1].start()
    
    def end_thread(self):
        print(len(self.thread_list))
        if len(self.thread_list) > 0:
            for i,th in enumerate(self.thread_list):
                if th.is_alive():
                    th.join()
                    del self.thread_list[i]

    def layout(self):
        s = ttk.Style()
        s.configure('Frame1.TFrame', background='#FF4C33')
        self.top = ttk.Frame(self.root,style='Frame1.TFrame') # create frame and told its parent (root)
        s2 = ttk.Style()
        s2.configure('Frame2.TFrame', background='#33FFF3')
        self.bottom = ttk.Frame(self.root, style='Frame2.TFrame')
        s3 = ttk.Style()
        s3.configure('Frame3.TFrame', background='#E9FF33')
        self.right = ttk.Frame(self.root, style='Frame3.TFrame')

        # expert frame
        self.expert_area = ttk.LabelFrame(self.top, text="Expert")
        self.expert_area.grid(row=0,column=0,padx=30, pady=5)
        # self.expert_area.grid_propagate(0)

        self.expert_area_vid = ttk.Frame(self.top)
        self.expert_area_vid.grid(row=1,column=0,padx=30, pady=5)

        # student frame
        self.student_area = ttk.LabelFrame(self.top, text="Student")
        self.student_area.grid(row=0,column=1, padx=30, pady=5)
        # self.student_area.grid_propagate(0)

        self.student_area_vid = ttk.Frame(self.top)
        self.student_area_vid.grid(row=1,column=1, padx=100, pady=5)

        # result frame
        self.result_area = ttk.LabelFrame(self.bottom, text="Result")
        self.result_area.grid(row=3, column=0, rowspan=2, columnspan=2 , sticky='S',padx=5, pady=5)
        

        # option frame
        self.option_area = ttk.Labelframe(self.right, text="Options")
        self.option_area.grid(row=0,column=2, padx=5, pady=5, ipadx= 5, ipady=5)

        self.top.grid(row=0,column=0)
        # self.top.grid_propagate(0)
        self.bottom.grid(row=1,column=0)
        # self.bottom.grid_propagate(0)
        self.right.grid(row=0, column=2, rowspan=2)
        # self.right.grid_propagate(0)

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

        # path label
        self.video_dir_expert = ttk.Label(self.expert_area, text='')
        self.video_dir_expert.grid(row = 3, column=0)
        self.video_dir_student = ttk.Label(self.student_area, text='')
        self.video_dir_student.grid(row = 3, column=0)

        # calculate button
        cal_button = ttk.Button(self.top, text = 'Go', command=self.start_thread) # command = link to onclick function
        cal_button.grid(row=2, column= 0, columnspan=3, pady=10)

        # Similarity score label
        self.score_text = ttk.Label(self.result_area, text="Score here!",font=16)
        self.score_text.grid(row=0,column=0, columnspan=6, padx=10,pady=10)

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

        self.weight_check = ttk.Checkbutton(self.option_area, text='Weight', variable=self.weight_var, command=self.input_weight)
        self.weight_check.grid(row=2, sticky='W', padx=5)        

        self.MAW_var = tk.IntVar()
        self.MAW_var.set(0)

        self.MAW_check = ttk.Checkbutton(self.option_area, text='Moving Weight', variable=self.MAW_var, command=self.MAW_window)
        self.MAW_check.grid(row=3, sticky='W', padx=5)

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
    
        if not self.weight_var.get():
            self.MAW_check['state']=tk.NORMAL
        else:
            self.MAW_check['state']=tk.DISABLED

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

        if not self.MAW_var.get():
            self.weight_check['state']=tk.NORMAL
        else:
            self.weight_check['state']=tk.DISABLED
        
        if (self.MAW_var.get()):
            MAW_window = ttk.Toplevel()
            MAW_window.title('Moving Weight')

            MAW_frame = ttk.LabelFrame(MAW_window, text='Moving Weight')
            MAW_frame.grid(row=0,column=1,padx=5,pady=5,ipadx=5,ipady=5)

            Window_label = ttk.Label(MAW_frame, text='Window',font=16)
            Window_label.grid(row=0,column=0, padx=5,pady=2)

            window = tk.StringVar(value=60)
            Window_input = ttk.Entry(MAW_frame, textvariable=window)
            # Window_input.insert(0, self.window_output)
            Window_input.grid(row=0,column=1, padx=5,pady=2)

            def window_get():
                
                try:
                    self.window_output = int(window.get())
                except:
                    tk.messagebox.showerror("Error","Window must be a number")
                MAW_window.destroy()

            submit_button = ttk.Button(MAW_frame, text='Done', command=window_get)
            submit_button.grid(row=2,pady=5, columnspan=2)


    def upload_vid_expert(self):
        self.end_thread()
        self.expert_area.filename = tk.filedialog.askopenfilename(title="Please Select a Video", filetypes=(("mp4 file","*.mp4"),("all files", "*.*")))
        self.video_dir_expert.config(text=self.expert_area.filename)

        self.player_expert = TkinterVideo(self.expert_area_vid, keep_aspect=True)
        self.player_expert.load(self.expert_area.filename)
        self.player_expert.bind("<<Loaded>>", lambda e: e.widget.config(width=400, height=250))
        
    def upload_vid_student(self):
        self.end_thread()
        self.student_area.filename = tk.filedialog.askopenfilename(title="Please Select a Video", filetypes=(("mp4 file","*.mp4"),("all files", "*.*")))
        self.video_dir_student.config(text=self.student_area.filename)

        self.player_student = TkinterVideo(self.student_area_vid, keep_aspect=True)
        self.player_student.load(self.student_area.filename)
        self.player_student.bind("<<Loaded>>", lambda e: e.widget.config(width=400, height=250))

    
    def play_video_expert(self):
        
        self.player_expert.grid(row=0)
        self.player_expert.play()

    def play_video_student(self):
    
        self.player_student.grid(row=0)
        self.player_student.play()   

    def calculate_video(self):

        self.score_text.config(text='Calculating...')

        t0 = time.time()

        exe = ProcessPoolExecutor(os.cpu_count())
        
        try:
            ladk_expert = exe.submit(landmark_detection, self.expert_area.filename)
        except:
            tk.messagebox.showerror("Cannot calculate","Please upload expert's video.")
        try:
            ladk_student = exe.submit(landmark_detection, self.student_area.filename)
        except:
            tk.messagebox.showerror("Cannot calculate","Please upload student's video.")

        cam_ladk_expert, world_ladk_expert, all_frame_expert = ladk_expert.result()
        cam_ladk_student, world_ladk_student, all_frame_student = ladk_student.result()
        

        if self.weight_var.get():
            if 'cosine' in self.method.get().lower():
                weight_arr = np.array(self.weight_cosine)
            else:
                weight_arr = np.array(self.weight_angle)
        else:
            if 'cosine' in self.method.get().lower():
                weight_arr = np.ones(16)
            else:
                weight_arr = np.ones(12)
            

        if 'cosine' in self.method.get().lower():
            limb_expert_exe = exe.submit(find_limb, world_ladk_expert)
            limb_student_exe = exe.submit(find_limb, world_ladk_student)
            limb_expert = limb_expert_exe.result()
            limb_student = limb_student_exe.result()
            exe.shutdown(wait=True)
            self.path, self.dist_mat, self.dist_lndmk_mat, self.cost_mat, self.cost = dtw(limb_expert,limb_student,cosine_similarity,
                                                                                            weight=weight_arr, MAW=self.MAW_var.get(), 
                                                                                            norm_value=int(self.norm.get()), 
                                                                                            windows=self.window_output, thresh=self.thresh_var.get(),
                                                                                            expo=self.expo_var.get())


        else:
            angle_expert_exe = exe.submit(find_angle, world_ladk_expert)
            angle_student_exe = exe.submit(find_angle, world_ladk_student)
            angle_expert = angle_expert_exe.result()
            angle_student = angle_student_exe.result()
            exe.shutdown(wait=True)
            self.path, self.dist_mat, self.dist_lndmk_mat, self.cost_mat, self.cost = dtw(angle_expert,angle_student,angle_similarity,
                                                                                            weight=weight_arr, MAW=self.MAW_var.get(), 
                                                                                            norm_value=int(self.norm.get()), 
                                                                                            windows=self.window_output, thresh=self.thresh_var.get(),
                                                                                            expo=self.expo_var.get())
        
        self.merge_img = display_error(self.path, self.dist_mat, self.dist_lndmk_mat, 
                                                all_frame_expert, all_frame_student, 
                                                cam_ladk_expert, cam_ladk_student,
                                                self.method.get().lower())
        
        t1 = time.time()
        print('เวลาในการคำนวณ: %f'%(t1-t0))
        
        self.score_text.config(text=f'Similarity score: {self.cost*100:.2f}%')
        

        # def display_result(cam_landmark, width, height, image, dist_lndmk_mat, sim_diff_function):
        def display_result():
            # draw_error(cam_landmark, width, height, image, dist_lndmk_mat, sim_diff_function)
            
            result_window = ttk.Toplevel()
            result_window.title('Video result')

            result_label = ttk.Label(result_window)
            result_label.grid(row=0,column=0, padx=5,pady=2)

            def video_stream(frame):
                if frame < len(self.merge_img):
                    cv2img = cv2.resize(self.merge_img[frame], (0, 0), fx = 0.5, fy = 0.5)
                    cv2img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2img)
                    imgtk = ImageTk.PhotoImage(image=img)
                    result_label.imgtk = imgtk
                    result_label.configure(image=imgtk)
                result_label.after(30, lambda: video_stream(frame+1))

            video_stream(0)

        show_result_button = ttk.Button(self.result_area, text = 'Show result', command=display_result) # command = link to onclick function 
        # show_result_button = ttk.Button(self.result_area, text = 'Show result', command=lambda: display_result(cam_ladk_expert, all_frame_expert[0].shape[1], all_frame_expert[0].shape[0], all_frame_expert[0], self.dist_lndmk_mat, self.method.get().lower())) # command = link to onclick function
        show_result_button.grid(row=1, column= 0, pady=5)
        

if __name__ == "__main__":
    App()
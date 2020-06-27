from image_procesor import ImageProcesor
import tkinter as tk
import numpy as np
from tkinter.filedialog import askopenfilename, asksaveasfile
import cv2


class Application(tk.Frame):
    def __init__(self, master=None, name="Prueba"):
        super().__init__(master)
        self.ip = ImageProcesor()
        self.master = master
        self.master.title(name)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.load_button = tk.Button(self,
                                     text="Load image",
                                     fg="white",
                                     bg="#1979a9",
                                     command=self.load_file)
        self.load_button.pack(side="left")

        self.create_button = tk.Button(self, text="Create image", fg="white", bg="#30C604",
                                       command=self.create)
        self.create_button.pack(side="left")

        self.quit_button = tk.Button(self, text="QUIT", fg="white", bg="#FF4633",
                                     command=self.master.destroy)
        self.quit_button.pack(side="right")

    def load_file(self):
        filename = askopenfilename()
        if filename:
            self.image = self.ip.read_image(filename)
            self.create_secundary()

    def save_file(self):
        f = asksaveasfile(mode='w', defaultextension=".png")
        if f:
            self.ip.save(f.name, self.image)

    def save_file_text(self):
        f = asksaveasfile(mode='w', defaultextension=".png")
        if f:
            self.ip.change_image_character(self.ip.get_only_B(self.image), f.name, savedata=True)

    def disable_buttons(self):
        self.create_button['state'] = tk.DISABLED
        self.load_button['state'] = tk.DISABLED

        self.create_button['bg'] = "#267110"
        self.load_button['bg'] = "#195876"

    def enable_buttons(self):
        self.create_button['state'] = tk.NORMAL
        self.load_button['state'] = tk.NORMAL

        self.create_button['bg'] = "#30C604"
        self.load_button['bg'] = "#1979a9"

    def create(self):
        self.disable_buttons()
        self.frame = tk.Frame(self.master)
        self.frame.pack(side="bottom")

        height_laber = tk.Label(self.frame, text="Height: ")
        height_laber.grid(column=0, row=1)

        height_image = tk.Spinbox(self.frame,
                                  width=3,
                                  from_=1,
                                  to=100,
                                  state="readonly")
        height_image.grid(column=1, row=1)

        width_laber = tk.Label(self.frame, text="Width: ")
        width_laber.grid(column=2, row=1)

        width_image = tk.Spinbox(self.frame,
                                 width=3,
                                 from_=1,
                                 to=100,
                                 state="readonly")
        width_image.grid(column=3, row=1)

        button_create = tk.Button(self.frame,
                                  text="Create",
                                  fg="white",
                                  bg="#30C604",
                                  command=lambda: self.create_image(int(height_image.get()),
                                                                    int(width_image.get()))
                                  )
        button_create.grid(column=4, row=1)

        button_cancel = tk.Button(self.frame,
                                  text="Cancel",
                                  fg="white",
                                  bg="#FF4633",
                                  command=self.cancel_create)
        button_cancel.grid(column=5, row=1)

    def show_image(self):
        self.ip.show_image_clic(self.image)
        self.frame.destroy()
        self.create_table()

    def cancel_create(self):
        self.enable_buttons()
        self.frame.destroy()

    def create_image(self, x, y):
        self.image = np.zeros((x, y, 3), np.uint8)
        self.image[:, :, :] = 255
        self.frame.destroy()
        self.create_secundary()

    def create_secundary(self):
        self.frame1 = tk.Frame(self.master)
        self.frame1.pack(side="bottom")

        show_image_button = tk.Button(self.frame1, text="Show image", fg="white", bg="#7332a6", command=self.show_image)
        show_image_button.grid(row=0, column=0)
        show_image_button = tk.Button(self.frame1, text="Export in text", fg="white", bg="#b5bf21",
                                      command=self.save_file_text)
        show_image_button.grid(row=0, column=1)
        self.create_table()

    def create_table(self):
        self.frame = tk.Frame(self.master)
        self.frame.pack(side="bottom")
        indicadorx = len(self.image) - 40 if len(self.image) > 40 else 0
        indicadory = len(self.image[0]) - 40 if len(self.image[0]) > 40 else 0
        for i in range(len(self.image) - indicadorx):
            for j in range(len(self.image[0]) - indicadory):
                var = tk.StringVar(root)
                var.set(int(self.image[i, j, 0] / 25))
                spin = tk.Spinbox(self.frame,
                                  width=3,
                                  from_=0,
                                  to=10,
                                  state="readonly",
                                  textvariable=var
                                  )
                spin["command"] = lambda x=i, y=j, value=spin: self.changeValue(x, y, int(value.get()))
                spin.grid(column=j, row=i)
        self.create_button['text'] = "Save image"
        self.create_button['command'] = self.save_file
        self.enable_buttons()

    def changeValue(self, x, y, value):
        self.image[x, y] = value * 25


root = tk.Tk()
app = Application(master=root)
app.mainloop()

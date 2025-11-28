"""
GUI Message Helper Module
=========================

Provides GUI dialogs and messages for all modules to use instead of terminal output.
This ensures everything is accessed via the GUI.
"""

import tkinter.messagebox as messagebox
from tkinter import Tk, Toplevel, Label, Button, Text, Scrollbar, Frame
import sys


class GUIMessages:
    """Helper class for showing GUI messages instead of terminal output."""
    
    @staticmethod
    def show_info(title, message):
        """Show an information dialog."""
        root = Tk()
        root.withdraw()  # Hide main window
        messagebox.showinfo(title, message, parent=root)
        root.destroy()
    
    @staticmethod
    def show_warning(title, message):
        """Show a warning dialog."""
        root = Tk()
        root.withdraw()
        messagebox.showwarning(title, message, parent=root)
        root.destroy()
    
    @staticmethod
    def show_error(title, message):
        """Show an error dialog."""
        root = Tk()
        root.withdraw()
        messagebox.showerror(title, message, parent=root)
        root.destroy()
    
    @staticmethod
    def ask_yesno(title, message):
        """Show a yes/no question dialog."""
        root = Tk()
        root.withdraw()
        result = messagebox.askyesno(title, message, parent=root)
        root.destroy()
        return result
    
    @staticmethod
    def show_progress_window(title, messages_list):
        """Show a progress window with messages."""
        root = Tk()
        root.title(title)
        root.geometry("600x400")
        root.resizable(False, False)
        
        # Create frame
        frame = Frame(root, padx=20, pady=20)
        frame.pack(fill='both', expand=True)
        
        # Title label
        title_label = Label(frame, text=title, font=('Helvetica', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Text area with scrollbar
        text_frame = Frame(frame)
        text_frame.pack(fill='both', expand=True)
        
        scrollbar = Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')
        
        text_widget = Text(text_frame, wrap='word', yscrollcommand=scrollbar.set, 
                          font=('Courier', 10))
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Add messages
        for msg in messages_list:
            text_widget.insert('end', msg + '\n')
        
        text_widget.config(state='disabled')
        
        # Close button
        close_button = Button(frame, text="Close", command=root.destroy, 
                             width=20, height=2)
        close_button.pack(pady=(10, 0))
        
        root.mainloop()


# Convenience functions
def show_info(title, message):
    """Show information dialog."""
    GUIMessages.show_info(title, message)

def show_warning(title, message):
    """Show warning dialog."""
    GUIMessages.show_warning(title, message)

def show_error(title, message):
    """Show error dialog."""
    GUIMessages.show_error(title, message)

def ask_yesno(title, message):
    """Show yes/no question."""
    return GUIMessages.ask_yesno(title, message)

def show_progress(title, messages):
    """Show progress window."""
    GUIMessages.show_progress_window(title, messages)


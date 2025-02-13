import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
import os, json, re

class Logview(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Log Viewer")
        self.geometry("800x600")

        self.fixed_columns = ["timestamp", "level", "module", "function", "event", "step", "data"]
        self.column_filters = {}

        # Top controls: Choose directory and search filter
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(control_frame, text="Choose Directory", command=self.choose_directory).pack(side=tk.LEFT)
        ttk.Label(control_frame, text="Global Search:").pack(side=tk.LEFT, padx=(10, 0))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(control_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, padx=(5, 5))
        search_entry.bind("<KeyRelease>", self.filter_table)

        # Treeview to display logs
        self.tree = ttk.Treeview(self, columns=self.fixed_columns, show="headings")
        self.tree.pack(expand=True, fill=tk.BOTH)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)

        for col in self.fixed_columns:
            self.tree.heading(col, text=col, command=lambda _col=col: self.sort_by(_col, False))
            self.tree.column(col, anchor=tk.W, width=120)
        self.tree.bind("<Button-3>", self.on_right_click)

        self.logs = []          # full list of logs (list of dicts)
        self.filtered_logs = [] # logs filtered by search

    def choose_directory(self):
        directory = filedialog.askdirectory()
        print(f"Chose directory: {directory}")
        if directory:
            self.load_logs(directory)

    def load_logs(self, directory):
        self.logs.clear()
        # Iterate over .log files in the directory
        pattern = re.compile(r'\.log(?:\.\d+)?$')
        for filename in os.listdir(directory):
            if pattern.search(filename):
                filepath = os.path.join(directory, filename)
                print(f"Noticed file: {filepath}")
                with open(filepath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                self.logs.append(entry)
                            except Exception as e:
                                print(f"Error parsing a line in {filename}: {e}")
        self.column_filters = {}
        self.filtered_logs = list(self.logs)
        self.populate_table(self.filtered_logs)
        print(f"Loaded {len(self.filtered_logs)} logs.")


    def setup_table(self):
        # Clear previous table content and columns
        for col in self.tree["columns"]:
            self.tree.heading(col, command=lambda: None)
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = ()

        # Determine all keys present in the logs
        columns = set()
        for log in self.logs:
            columns.update(log.keys())
        self.all_columns = sorted(list(columns))

        self.tree["columns"] = self.all_columns
        for col in self.all_columns:
            self.tree.heading(col, text=col, command=lambda _col=col: self.sort_by(_col, False))
            self.tree.column(col, anchor=tk.W, width=100)

        self.populate_table(self.filtered_logs)

    def populate_table(self, logs):
        self.tree.delete(*self.tree.get_children())
        for log in logs:
            values = [log.get(col, "") for col in self.fixed_columns]
            self.tree.insert("", tk.END, values=values)

    def sort_by(self, col, descending):
        try:
            self.filtered_logs.sort(key=lambda x: x.get(col, ""), reverse=descending)
        except Exception as e:
            print(f"Sorting error on column {col}: {e}")
        self.populate_table(self.filtered_logs)
        # Toggle sort order on next click
        self.tree.heading(col, command=lambda: self.sort_by(col, not descending))

    def filter_table(self, event=None):
        global_query = self.search_var.get().lower()
        filtered = []
        for log in self.logs:
            if global_query and not any(global_query in str(log.get(col, "")).lower() for col in self.fixed_columns):
                continue
            passes = True
            for col, ftext in self.column_filters.items():
                if ftext:
                    tokens = [token.strip().lower() for token in ftext.split(",") if token.strip()]
                    field_value = str(log.get(col, "")).lower()
                    if not any(token in field_value for token in tokens):
                        passes = False
                        break
            if passes:
                filtered.append(log)
        self.filtered_logs = filtered
        self.populate_table(self.filtered_logs)

    def on_right_click(self, event):
        region = self.tree.identify_region(event.x, event.y)
        if region == "heading":
            col_id = self.tree.identify_column(event.x)  # returns e.g., "#1"
            try:
                col_index = int(col_id.replace("#", "")) - 1
                if 0 <= col_index < len(self.fixed_columns):
                    column_name = self.fixed_columns[col_index]
                    self.show_column_menu(event, column_name)
            except Exception as e:
                print("Could not determine column:", e)

    def show_column_menu(self, event, column_name):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Sort Ascending", command=lambda: self.sort_by(column_name, False))
        menu.add_command(label="Sort Descending", command=lambda: self.sort_by(column_name, True))
        menu.add_separator()
        menu.add_command(label="Set Filter", command=lambda: self.set_filter(column_name))
        menu.add_command(label="Clear Filter", command=lambda: self.clear_filter(column_name))
        menu.tk_popup(event.x_root, event.y_root)

    def set_filter(self, column_name):
        current = self.column_filters.get(column_name, "")
        prompt = f"Enter filter text for '{column_name}':"
        filt = simpledialog.askstring("Set Filter", prompt, initialvalue=current)
        if filt is not None:
            self.column_filters[column_name] = filt.strip()
            self.filter_table()

    def clear_filter(self, column_name):
        if column_name in self.column_filters:
            self.column_filters[column_name] = ""
            self.filter_table()

if __name__ == "__main__":
    app = Logview()
    app.mainloop()

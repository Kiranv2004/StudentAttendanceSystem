import os
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AttendanceDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Attendance System")
        self.root.geometry("800x600")
        
        # Attendance data file
        self.attendance_file = os.path.join("data", "attendance.csv")
        
        # Initialize UI elements
        self.setup_ui()
        
        # Load attendance data
        self.load_attendance_data()
        
    def setup_ui(self):
        """Set up the user interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.attendance_tab = ttk.Frame(self.notebook)
        self.reports_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.attendance_tab, text="Attendance Records")
        self.notebook.add(self.reports_tab, text="Reports")
        self.notebook.add(self.settings_tab, text="Settings")
        
        # Set up Attendance Tab
        self.setup_attendance_tab()
        
        # Set up Reports Tab
        self.setup_reports_tab()
        
        # Set up Settings Tab
        self.setup_settings_tab()
        
        # Button to launch the live attendance system
        self.launch_btn = tk.Button(
            self.root, 
            text="Launch Live Attendance System",
            command=self.launch_live_system,
            bg="#4CAF50", 
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10
        )
        self.launch_btn.pack(pady=10)
        
    def setup_attendance_tab(self):
        """Setup the attendance records tab"""
        # Frame for date filter
        filter_frame = ttk.Frame(self.attendance_tab)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Date filter
        ttk.Label(filter_frame, text="Filter by date:").pack(side=tk.LEFT, padx=5)
        self.date_var = tk.StringVar()
        self.date_var.set("All dates")
        
        self.date_combo = ttk.Combobox(filter_frame, textvariable=self.date_var)
        self.date_combo.pack(side=tk.LEFT, padx=5)
        self.date_combo.bind("<<ComboboxSelected>>", self.filter_attendance)
        
        # Search field
        ttk.Label(filter_frame, text="Search:").pack(side=tk.LEFT, padx=20)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(filter_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, padx=5)
        self.search_var.trace("w", lambda name, index, mode: self.filter_attendance())
        
        # Clear filters button
        ttk.Button(filter_frame, text="Clear Filters", command=self.clear_filters).pack(side=tk.RIGHT, padx=5)
        
        # Create treeview for attendance data
        columns = ("Name", "Date", "Time")
        self.attendance_tree = ttk.Treeview(self.attendance_tab, columns=columns, show="headings")
        
        # Set column headings
        for col in columns:
            self.attendance_tree.heading(col, text=col, command=lambda c=col: self.sort_by(c))
            self.attendance_tree.column(col, width=100)
            
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.attendance_tab, orient=tk.VERTICAL, command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscroll=scrollbar.set)
        
        # Pack tree and scrollbar
        self.attendance_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export button
        export_button = ttk.Button(self.attendance_tab, text="Export to CSV", command=self.export_attendance)
        export_button.pack(pady=10)
        
    def setup_reports_tab(self):
        """Setup the reports tab"""
        # Controls frame
        controls_frame = ttk.Frame(self.reports_tab)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Report type selection
        ttk.Label(controls_frame, text="Report Type:").pack(side=tk.LEFT, padx=5)
        self.report_type = tk.StringVar()
        report_combo = ttk.Combobox(
            controls_frame, 
            textvariable=self.report_type,
            values=["Daily Attendance", "Weekly Summary", "Student Attendance Rate"]
        )
        report_combo.pack(side=tk.LEFT, padx=5)
        report_combo.current(0)
        
        # Generate report button
        ttk.Button(
            controls_frame, 
            text="Generate Report", 
            command=self.generate_report
        ).pack(side=tk.LEFT, padx=20)
        
        # Frame for the chart
        self.chart_frame = ttk.Frame(self.reports_tab)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def setup_settings_tab(self):
        """Setup the settings tab"""
        # Settings frame
        settings_frame = ttk.Frame(self.settings_tab, padding=20)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Face recognition threshold
        ttk.Label(settings_frame, text="Face Recognition Threshold:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.face_threshold = tk.DoubleVar(value=0.6)
        face_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.face_threshold)
        face_scale.grid(row=0, column=1, sticky=tk.EW, pady=5)
        ttk.Label(settings_frame, textvariable=self.face_threshold).grid(row=0, column=2, padx=5)
        
        # Anti-spoofing threshold
        ttk.Label(settings_frame, text="Anti-spoofing Threshold:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.spoof_threshold = tk.DoubleVar(value=0.5)
        spoof_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.spoof_threshold)
        spoof_scale.grid(row=1, column=1, sticky=tk.EW, pady=5)
        ttk.Label(settings_frame, textvariable=self.spoof_threshold).grid(row=1, column=2, padx=5)
        
        # Save settings button
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).grid(row=2, column=1, pady=20)
        
        # Reset database button (with confirmation)
        ttk.Button(
            settings_frame, 
            text="Reset Attendance Database", 
            command=self.confirm_reset_database,
            style="Danger.TButton"
        ).grid(row=3, column=1, pady=20)
        
        # Configure grid weights
        settings_frame.columnconfigure(1, weight=1)
        
    def load_attendance_data(self):
        """Load attendance data from CSV file"""
        try:
            if os.path.exists(self.attendance_file):
                # Read the CSV file
                with open(self.attendance_file, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    self.attendance_data = list(reader)
                
                # Update the treeview
                self.update_attendance_tree()
                
                # Update date filter options
                self.update_date_filter()
            else:
                # Create directory and empty file if not exists
                os.makedirs(os.path.dirname(self.attendance_file), exist_ok=True)
                with open(self.attendance_file, 'w') as f:
                    f.write("Name,Date,Time\n")
                self.attendance_data = []
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load attendance data: {str(e)}")
            self.attendance_data = []
    
    def update_attendance_tree(self):
        """Update the attendance treeview with filtered data"""
        # Clear the tree
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
            
        # Get filter values
        date_filter = self.date_var.get()
        search_text = self.search_var.get().lower()
        
        # Apply filters
        filtered_data = self.attendance_data
        if date_filter != "All dates":
            filtered_data = [row for row in filtered_data if row[1] == date_filter]
            
        if search_text:
            filtered_data = [row for row in filtered_data if search_text in row[0].lower()]
            
        # Add filtered data to tree
        for row in filtered_data:
            self.attendance_tree.insert("", "end", values=row)
    
    def update_date_filter(self):
        """Update the date filter dropdown with available dates"""
        dates = ["All dates"]
        if self.attendance_data:
            # Extract unique dates
            unique_dates = sorted(set(row[1] for row in self.attendance_data))
            dates.extend(unique_dates)
            
        # Update combobox values
        self.date_combo['values'] = dates
    
    def filter_attendance(self, event=None):
        """Filter attendance data based on selected filters"""
        self.update_attendance_tree()
    
    def clear_filters(self):
        """Clear all filters"""
        self.date_var.set("All dates")
        self.search_var.set("")
        self.update_attendance_tree()
    
    def sort_by(self, column):
        """Sort the attendance data by a specific column"""
        # Get column index
        columns = ["Name", "Date", "Time"]
        col_idx = columns.index(column)
        
        # Sort the data
        self.attendance_data.sort(key=lambda x: x[col_idx])
        
        # Update the tree
        self.update_attendance_tree()
    
    def export_attendance(self):
        """Export the current filtered attendance data to CSV"""
        try:
            # Get all visible items
            items = self.attendance_tree.get_children()
            if not items:
                messagebox.showinfo("Export", "No data to export")
                return
                
            # Get export file path
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile=f"attendance_export_{datetime.now().strftime('%Y%m%d')}.csv"
            )
            
            if not filename:
                return
                
            # Write data to CSV
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Time"])
                
                for item in items:
                    values = self.attendance_tree.item(item, 'values')
                    writer.writerow(values)
                    
            messagebox.showinfo("Export", f"Data exported successfully to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
    
    def generate_report(self):
        """Generate a report based on the selected type"""
        report_type = self.report_type.get()
        
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
            
        if not self.attendance_data:
            messagebox.showinfo("Report", "No attendance data available")
            return
            
        try:
            # Create a DataFrame from attendance data
            df = pd.DataFrame(self.attendance_data, columns=["Name", "Date", "Time"])
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if report_type == "Daily Attendance":
                # Count attendances per day
                daily_counts = df.groupby("Date").size()
                daily_counts.plot(kind="bar", ax=ax)
                ax.set_title("Daily Attendance Count")
                ax.set_ylabel("Number of Students")
                ax.set_xlabel("Date")
                
            elif report_type == "Weekly Summary":
                # Convert dates to datetime and get week numbers
                df["Date"] = pd.to_datetime(df["Date"])
                df["Week"] = df["Date"].dt.isocalendar().week
                weekly_counts = df.groupby("Week").size()
                weekly_counts.plot(kind="line", marker="o", ax=ax)
                ax.set_title("Weekly Attendance Summary")
                ax.set_ylabel("Number of Students")
                ax.set_xlabel("Week Number")
                ax.grid(True)
                
            elif report_type == "Student Attendance Rate":
                # Count attendance per student
                student_counts = df.groupby("Name").size().sort_values(ascending=False).head(10)
                student_counts.plot(kind="barh", ax=ax)
                ax.set_title("Top 10 Student Attendance Rates")
                ax.set_xlabel("Number of Days Present")
                ax.set_ylabel("Student")
                
            # Create canvas to display the chart
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Report Error", str(e))
    
    def save_settings(self):
        """Save the current settings"""
        # Get the values
        face_threshold = self.face_threshold.get()
        spoof_threshold = self.spoof_threshold.get()
        
        # Save settings to a file
        settings_file = os.path.join("data", "settings.csv")
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        
        with open(settings_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Setting", "Value"])
            writer.writerow(["face_threshold", face_threshold])
            writer.writerow(["spoof_threshold", spoof_threshold])
            
        messagebox.showinfo("Settings", "Settings saved successfully")
    
    def confirm_reset_database(self):
        """Confirm reset of attendance database"""
        confirm = messagebox.askyesno(
            "Confirm Reset",
            "Are you sure you want to reset the attendance database?\nThis action cannot be undone."
        )
        
        if confirm:
            self.reset_database()
    
    def reset_database(self):
        """Reset the attendance database"""
        try:
            # Create backup
            backup_file = os.path.join("data", f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            if os.path.exists(self.attendance_file):
                os.makedirs(os.path.dirname(backup_file), exist_ok=True)
                with open(self.attendance_file, 'r') as src, open(backup_file, 'w') as dst:
                    dst.write(src.read())
            
            # Reset the file
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Time"])
                
            # Clear data and update UI
            self.attendance_data = []
            self.update_attendance_tree()
            self.update_date_filter()
            
            messagebox.showinfo("Reset", f"Database reset successfully.\nBackup saved to {backup_file}")
            
        except Exception as e:
            messagebox.showerror("Reset Error", str(e))
    
    def launch_live_system(self):
        """Launch the live attendance system"""
        import subprocess
        import sys
        
        try:
            # Build the command to run the secure attendance system
            script_path = os.path.join(os.path.dirname(__file__), "secure_attendance_system.py")
            subprocess.Popen([sys.executable, script_path])
            
        except Exception as e:
            messagebox.showerror("Launch Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceDashboard(root)
    root.mainloop()
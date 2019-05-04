using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ProgressReporter
{
    public partial class MainForm : Form
    {
        private string ReportFileName;

        public static int Clamp(int value, int min, int max)
        {
            return (value < min) ? min : (value > max) ? max : value;
        }

        public MainForm(string[] args)
        {
            ReportFileName = args[0];

            InitializeComponent();
        }

        private void MainTimer_Tick(object sender, EventArgs e)
        {
            try
            {
                Process[] pname = Process.GetProcessesByName("UnrealLightmass");
                if(pname.Length == 0)
                    Application.Exit();

                string[] lines = System.IO.File.ReadAllLines(ReportFileName);
                CurrentTaskBox.Text = lines[0];
                CurrentProgressBar.Value = Clamp(int.Parse(lines[1]), 0, 100);
                OverallTaskBox.Text = lines[2];
                OverallProgressBar.Value = Clamp(int.Parse(lines[3]), 0, 100);
            }
            catch (FileNotFoundException)
            {
                Application.Exit();
            }
            catch(IOException)
            {

            }
        }
    }
}

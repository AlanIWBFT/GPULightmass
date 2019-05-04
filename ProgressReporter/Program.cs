using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ProgressReporter
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            if (args.Length == 0)
                return;

            if (!File.Exists(args[0]))
            {
                MessageBox.Show("File not found: " + args[0]);
                return;
            }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm(args));
        }
    }
}
